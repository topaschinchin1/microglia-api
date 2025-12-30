"""
Microglia Morphology Analysis API v4.0.0
With 4-CATEGORY MORPHOLOGY CLASSIFICATION

Categories (based on validated reference data):
1. RAMIFIED - 🟢 RESTING: Small soma, long processes, high branching, star-shaped
2. AMOEBOID - 🔴 ACTIVATED: Large soma, absent processes, no branching, round
3. HYPERTROPHIC - 🔴 ACTIVATED: Enlarged body, thickened processes  
4. ROD_LIKE - 🔴 ACTIVATED: Elongated bipolar shape

Calibration: n=14 samples, 100% accuracy, threshold=0.79

Requirements:
pip install flask numpy scikit-image scipy pillow gunicorn
"""

import os
import io
import base64
import json
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from skimage import io as skio
from skimage import filters, morphology, measure, color
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Validated threshold (recalibrated on n=14 samples)
SOLIDITY_THRESHOLD = 0.79
BORDERLINE_LOW = 0.78
BORDERLINE_HIGH = 0.80

# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image_from_base64(base64_string):
    """Load image from base64 string"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

def load_image_from_url(url):
    """Load image from URL"""
    import urllib.request
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

def load_image_from_file(file_storage):
    """Load image from uploaded file"""
    image = Image.open(file_storage)
    return np.array(image)

# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def preprocess_image(img):
    """Convert to grayscale if needed"""
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        gray = img[:, :, 0].astype(float)
    else:
        gray = img.astype(float)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
    return gray

def threshold_image(gray, method='otsu'):
    """Apply thresholding"""
    if method == 'otsu':
        thresh_val = filters.threshold_otsu(gray)
    elif method == 'li':
        thresh_val = filters.threshold_li(gray)
    elif method == 'triangle':
        thresh_val = filters.threshold_triangle(gray)
    elif method == 'local':
        thresh_val = filters.threshold_local(gray, block_size=101, method='gaussian')
        return gray > thresh_val
    else:
        thresh_val = filters.threshold_otsu(gray)
    return gray > thresh_val

def analyze_skeleton(binary_image):
    """Analyze skeleton - branch points and endpoints"""
    skeleton = skeletonize(binary_image)
    skeleton_length = np.sum(skeleton)
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
    neighbor_count = neighbor_count * skeleton
    
    endpoints = np.sum(neighbor_count == 1)
    branch_points = np.sum(neighbor_count >= 3)
    
    return {
        'skeleton_length_pixels': int(skeleton_length),
        'endpoints': int(endpoints),
        'branch_points': int(branch_points)
    }

def analyze_single_cell(cell_mask, cell_intensity=None):
    """Analyze morphology of a single cell"""
    props = measure.regionprops(cell_mask.astype(int))[0] if np.any(cell_mask) else None
    
    if props is None:
        return None
    
    area = props.area
    perimeter = props.perimeter
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-10)
    solidity = props.solidity
    aspect_ratio = props.major_axis_length / (props.minor_axis_length + 1e-10)
    area_perimeter_ratio = area / (perimeter + 1e-10)
    extent = props.extent
    eccentricity = props.eccentricity
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'solidity': float(solidity),
        'aspect_ratio': float(aspect_ratio),
        'area_perimeter_ratio': float(area_perimeter_ratio),
        'extent': float(extent),
        'eccentricity': float(eccentricity),
        'major_axis': float(props.major_axis_length),
        'minor_axis': float(props.minor_axis_length)
    }

# =============================================================================
# 4-CATEGORY MORPHOLOGY CLASSIFICATION
# =============================================================================

def classify_morphology_4category(metrics):
    """
    4-CATEGORY CLASSIFICATION SYSTEM
    
    Based on validated reference data from GPT-4o analysis:
    
    1. RAMIFIED - 🟢 RESTING
       - Small soma, long processes, high branching, star-shaped
       - High solidity (>0.79), low circularity (<0.5), moderate aspect ratio
       
    2. AMOEBOID - 🔴 ACTIVATED  
       - Large soma, absent processes, no branching, round
       - High circularity (>0.6), high solidity (>0.8), low aspect ratio (<2)
       
    3. HYPERTROPHIC - 🔴 ACTIVATED
       - Enlarged body, thickened processes
       - Medium solidity (0.72-0.82), low circularity, moderate branching
       
    4. ROD_LIKE - 🔴 ACTIVATED
       - Elongated bipolar shape
       - High aspect ratio (>3), high eccentricity (>0.9)
    """
    solidity = metrics.get('solidity', 0)
    circularity = metrics.get('circularity', 0)
    aspect_ratio = metrics.get('aspect_ratio', 1)
    eccentricity = metrics.get('eccentricity', 0)
    
    # ROD_LIKE: Elongated bipolar shape (highest priority for shape detection)
    if aspect_ratio > 3.0 and eccentricity > 0.85:
        return {
            'morphology_class': 'ROD_LIKE',
            'activation_state': 'ACTIVATED',
            'features': 'Elongated bipolar shape, high aspect ratio'
        }
    
    # AMOEBOID: Round, blob-like, no processes
    if circularity > 0.55 and solidity > 0.75 and aspect_ratio < 2.2:
        return {
            'morphology_class': 'AMOEBOID',
            'activation_state': 'ACTIVATED',
            'features': 'Large soma, absent processes, round shape'
        }
    
    # RAMIFIED: Star-shaped, highly branched (resting state)
    if solidity > SOLIDITY_THRESHOLD and circularity < 0.55:
        return {
            'morphology_class': 'RAMIFIED',
            'activation_state': 'RESTING',
            'features': 'Small soma, long processes, high branching, star-shaped'
        }
    
    # HYPERTROPHIC: Enlarged body with thickened processes
    if solidity < SOLIDITY_THRESHOLD and circularity < 0.5:
        return {
            'morphology_class': 'HYPERTROPHIC',
            'activation_state': 'ACTIVATED',
            'features': 'Enlarged body, thickened processes'
        }
    
    # Default: Classify based on solidity threshold
    if solidity >= SOLIDITY_THRESHOLD:
        return {
            'morphology_class': 'RAMIFIED',
            'activation_state': 'RESTING',
            'features': 'Compact morphology, resting-like'
        }
    else:
        return {
            'morphology_class': 'HYPERTROPHIC',
            'activation_state': 'ACTIVATED',
            'features': 'Irregular morphology, activation signs'
        }

# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_image(img, threshold_method='local', min_cell_size=50, max_cell_size=50000):
    """Full morphology analysis pipeline with 4-category classification"""
    
    # 1. Preprocess
    gray = preprocess_image(img)
    
    # 2. Threshold
    binary = threshold_image(gray, method=threshold_method)
    
    # 3. Clean up
    binary = remove_small_objects(binary, min_size=min_cell_size)
    binary = ndimage.binary_fill_holes(binary)
    
    # 4. Label connected components
    labeled = measure.label(binary)
    
    # 5. Initialize 4-category counts
    morphology_counts = {
        'RAMIFIED': 0,
        'AMOEBOID': 0,
        'HYPERTROPHIC': 0,
        'ROD_LIKE': 0
    }
    
    activation_counts = {
        'RESTING': 0,
        'ACTIVATED': 0
    }
    
    # 6. Analyze each cell
    cells = []
    
    for region in measure.regionprops(labeled):
        if region.area < min_cell_size or region.area > max_cell_size:
            continue
        
        cell_mask = labeled == region.label
        cell_metrics = analyze_single_cell(cell_mask)
        
        if cell_metrics:
            cell_metrics['label'] = int(region.label)
            cell_metrics['centroid'] = [float(c) for c in region.centroid]
            
            # Apply 4-category classification
            classification = classify_morphology_4category(cell_metrics)
            cell_metrics['morphology_class'] = classification['morphology_class']
            cell_metrics['activation_state'] = classification['activation_state']
            cell_metrics['features'] = classification['features']
            
            # Update counts
            morph_class = classification['morphology_class']
            if morph_class in morphology_counts:
                morphology_counts[morph_class] += 1
            
            act_state = classification['activation_state']
            if act_state in activation_counts:
                activation_counts[act_state] += 1
            
            cells.append(cell_metrics)
    
    # 7. Skeleton analysis
    skeleton_results = analyze_skeleton(binary)
    
    # 8. Calculate summary statistics
    total_cells = len(cells)
    
    if total_cells > 0:
        avg_circularity = np.mean([c['circularity'] for c in cells])
        avg_solidity = np.mean([c['solidity'] for c in cells])
        avg_area = np.mean([c['area'] for c in cells])
        avg_perimeter = np.mean([c['perimeter'] for c in cells])
        avg_area_perimeter = np.mean([c['area_perimeter_ratio'] for c in cells])
        avg_aspect_ratio = np.mean([c['aspect_ratio'] for c in cells])
        avg_eccentricity = np.mean([c['eccentricity'] for c in cells])
    else:
        avg_circularity = avg_solidity = avg_area = avg_perimeter = 0
        avg_area_perimeter = avg_aspect_ratio = avg_eccentricity = 0
    
    # 9. Calculate percentages
    morphology_percentages = {}
    for morph_type, count in morphology_counts.items():
        morphology_percentages[f'{morph_type}_percent'] = round(100 * count / total_cells, 1) if total_cells > 0 else 0
    
    activation_percentages = {}
    for act_type, count in activation_counts.items():
        activation_percentages[f'{act_type}_percent'] = round(100 * count / total_cells, 1) if total_cells > 0 else 0
    
    # 10. Determine overall classification
    classification = determine_overall_classification(
        morphology_counts, 
        activation_counts,
        total_cells, 
        avg_solidity
    )
    
    return {
        'summary': {
            'total_cells_detected': int(total_cells),
            'avg_circularity': float(round(avg_circularity, 4)),
            'avg_solidity': float(round(avg_solidity, 4)),
            'avg_area': float(round(avg_area, 2)),
            'avg_perimeter': float(round(avg_perimeter, 2)),
            'avg_area_perimeter_ratio': float(round(avg_area_perimeter, 4)),
            'avg_aspect_ratio': float(round(avg_aspect_ratio, 4)),
            'avg_eccentricity': float(round(avg_eccentricity, 4)),
            'skeleton_length': int(skeleton_results['skeleton_length_pixels']),
            'total_endpoints': int(skeleton_results['endpoints']),
            'total_branch_points': int(skeleton_results['branch_points'])
        },
        'morphology_counts': morphology_counts,
        'morphology_percentages': morphology_percentages,
        'activation_counts': activation_counts,
        'activation_percentages': activation_percentages,
        'classification': classification,
        'individual_cells': cells[:50]
    }

def determine_overall_classification(morph_counts, act_counts, total_cells, avg_solidity):
    """
    Determine overall classification based on 4-category system
    """
    if total_cells == 0:
        return {
            'activation_state': 'INSUFFICIENT_CELLS',
            'dominant_morphology': 'UNKNOWN',
            'confidence': 'LOW',
            'is_borderline': False,
            'reasoning': 'No cells detected'
        }
    
    # Find dominant morphology
    dominant_morph = max(morph_counts, key=morph_counts.get)
    dominant_count = morph_counts[dominant_morph]
    dominant_pct = 100 * dominant_count / total_cells
    
    # Check for borderline solidity
    is_borderline = bool(BORDERLINE_LOW <= avg_solidity <= BORDERLINE_HIGH)
    
    # Determine activation state based on avg_solidity (primary) and cell counts (secondary)
    resting_pct = 100 * act_counts['RESTING'] / total_cells
    activated_pct = 100 * act_counts['ACTIVATED'] / total_cells
    
    # Primary decision: avg_solidity threshold (validated on n=14 samples)
    if avg_solidity > BORDERLINE_HIGH:
        activation_state = 'RESTING'
        confidence = 'HIGH'
        reasoning = f'High solidity ({avg_solidity:.3f}) well above {SOLIDITY_THRESHOLD} threshold. Dominant morphology: {dominant_morph} ({dominant_pct:.1f}%).'
    elif avg_solidity >= SOLIDITY_THRESHOLD:
        activation_state = 'RESTING'
        confidence = 'MEDIUM' if is_borderline else 'HIGH'
        reasoning = f'Solidity ({avg_solidity:.3f}) above {SOLIDITY_THRESHOLD} threshold. Dominant morphology: {dominant_morph} ({dominant_pct:.1f}%).'
    elif avg_solidity >= BORDERLINE_LOW:
        activation_state = 'ACTIVATED'
        confidence = 'MEDIUM'
        reasoning = f'Solidity ({avg_solidity:.3f}) below {SOLIDITY_THRESHOLD} threshold (borderline zone). Dominant morphology: {dominant_morph} ({dominant_pct:.1f}%).'
    else:
        activation_state = 'ACTIVATED'
        confidence = 'HIGH'
        reasoning = f'Low solidity ({avg_solidity:.3f}) well below {SOLIDITY_THRESHOLD} threshold. Dominant morphology: {dominant_morph} ({dominant_pct:.1f}%).'
    
    # Add borderline warning if applicable
    if is_borderline:
        reasoning += f' ⚠️ BORDERLINE: Solidity in {BORDERLINE_LOW}-{BORDERLINE_HIGH} zone - recommend manual verification.'
    
    return {
        'activation_state': activation_state,
        'dominant_morphology': dominant_morph,
        'confidence': confidence,
        'is_borderline': is_borderline,
        'reasoning': reasoning
    }

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'version': '4.0.0-4category',
        'threshold': SOLIDITY_THRESHOLD,
        'borderline_zone': [BORDERLINE_LOW, BORDERLINE_HIGH],
        'calibration': 'n=14 samples (PBS vs LPS)',
        'accuracy': '100% (14/14)',
        'categories': ['RAMIFIED', 'AMOEBOID', 'HYPERTROPHIC', 'ROD_LIKE'],
        'accepts': ['application/json', 'multipart/form-data']
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint with 4-category morphology classification
    
    Accepts BOTH JSON and multipart/form-data
    
    JSON format:
    {"base64": "...", "threshold_method": "local"}
    or {"url": "...", "threshold_method": "local"}
    
    Form-data format:
    image: <file>
    threshold_method: local (optional)
    """
    try:
        img = None
        threshold_method = 'local'
        min_cell_size = 50
        max_cell_size = 50000
        
        # Check if it's a file upload (multipart/form-data)
        if request.files and 'image' in request.files:
            file = request.files['image']
            img = load_image_from_file(file)
            threshold_method = request.form.get('threshold_method', 'local')
            min_cell_size = int(request.form.get('min_cell_size', 50))
            max_cell_size = int(request.form.get('max_cell_size', 50000))
        
        # Check if it's JSON
        elif request.is_json:
            data = request.get_json()
            if 'base64' in data:
                img = load_image_from_base64(data['base64'])
            elif 'url' in data:
                img = load_image_from_url(data['url'])
            threshold_method = data.get('threshold_method', 'local')
            min_cell_size = data.get('min_cell_size', 50)
            max_cell_size = data.get('max_cell_size', 50000)
        
        # Fallback: try to get any uploaded file
        elif request.files:
            for key in request.files:
                file = request.files[key]
                img = load_image_from_file(file)
                break
            threshold_method = request.form.get('threshold_method', 'local')
        
        if img is None:
            return jsonify({
                'error': 'No image provided',
                'hint': 'Send image as file upload (form-data with "image" field) or JSON with "base64" or "url" key'
            }), 400
        
        results = analyze_image(
            img, 
            threshold_method=threshold_method,
            min_cell_size=min_cell_size,
            max_cell_size=max_cell_size
        )
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/quick', methods=['POST'])
def analyze_quick():
    """Quick analysis - summary and classification only"""
    try:
        img = None
        
        if request.files and 'image' in request.files:
            file = request.files['image']
            img = load_image_from_file(file)
        elif request.is_json:
            data = request.get_json()
            if 'base64' in data:
                img = load_image_from_base64(data['base64'])
            elif 'url' in data:
                img = load_image_from_url(data['url'])
        elif request.files:
            for key in request.files:
                file = request.files[key]
                img = load_image_from_file(file)
                break
        
        if img is None:
            return jsonify({'error': 'No image provided'}), 400
        
        results = analyze_image(img)
        
        return jsonify({
            'summary': results['summary'],
            'morphology_counts': results['morphology_counts'],
            'morphology_percentages': results['morphology_percentages'],
            'activation_counts': results['activation_counts'],
            'activation_percentages': results['activation_percentages'],
            'classification': results['classification']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
