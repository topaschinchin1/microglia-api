FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY microglia_morphology_api_v4.py .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "microglia_morphology_api_v4:app"]
