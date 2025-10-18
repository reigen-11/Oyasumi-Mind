# ---- Base Image ----
FROM python:3.12.2-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    HOST=0.0.0.0 \
    PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy/pandas/sklearn etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

# Copy app code
COPY . .

# Expose port for Render
EXPOSE 5000

# Optional healthcheck (useful for Render)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the app with Gunicorn (using app.py -> app variable)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "wsgi:app"]

