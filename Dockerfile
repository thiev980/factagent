FROM python:3.11-slim

# System-Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Dependencies installieren (Cache-Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren
COPY . .

# Verzeichnisse für Datenbank und Graphs erstellen
RUN mkdir -p /app/graphs /app/data

# Port für Chainlit (HF Spaces erwartet Port 7860)
EXPOSE 7860

# Chainlit starten
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
