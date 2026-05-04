FROM python:3.11-slim

LABEL description="Tesis Maestría · Gasto Social Colombia · Dash Multi-Page"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/src

EXPOSE 8050
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]