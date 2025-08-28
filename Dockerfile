FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app/patient_app_csv_testing


# deps for TLS, etc.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app (all of it)
COPY . /app

EXPOSE 8080
# CMD ["gunicorn","--workers","2","--threads","2","--timeout","60","--bind","0.0.0.0:8080","gunicorn.conf.py","app:app"]
# CMD ["gunicorn", "--workers", "2", "--threads", "2", "--timeout", "60", "--bind", "0.0.0.0:8080", "--config", "gunicorn.conf.py", "app:app"]
CMD ["gunicorn","-c","gunicorn.conf.py","app:app"]