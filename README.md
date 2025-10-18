# 💤 OYASUMI-MIND

*Empowering minds, transforming wellness through innovation.*

![Last Commit](https://img.shields.io/github/last-commit/reigen-11/Oyasumi-Mind?style=flat&color=blue)
![Languages](https://img.shields.io/github/languages/count/reigen-11/Oyasumi-Mind?color=yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)

---

## 🛠 Built With

![Flask](https://img.shields.io/badge/-Flask-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-orange?logo=scikitlearn)
![NumPy](https://img.shields.io/badge/-NumPy-blue?logo=numpy)
![pandas](https://img.shields.io/badge/-pandas-150458?logo=pandas)
![Gunicorn](https://img.shields.io/badge/-Gunicorn-green)
![GitHub Actions](https://img.shields.io/badge/-GitHub%20Actions-2088FF?logo=githubactions)

---

## 📑 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

**Oyasumi-Mind** is a toolkit for building, deploying, and interpreting mental-health prediction systems.  
It provides an end-to-end pipeline for data ingestion, cleaning, feature engineering, model training, and interpretation — plus a Flask web frontend for demoing models.

### 💡 Key Features

- Data pipeline: ingestion, cleaning, and feature engineering  
- Web deployment: Flask app with WSGI support (Gunicorn)  
- Model tooling: training, evaluation, and visualization for interpretability  
- Containerization: Docker image for reproducible deployments  
- Utilities: centralized logging and helpful development tools

---

## 🚀 Getting Started

### 🧱 Prerequisites

- Python 3.10 or newer
- Git
- (Optional) Docker, for containerized deployment

---

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/reigen-11/Oyasumi-Mind.git
cd Oyasumi-Mind

# Create and activate a virtual environment
python -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
# venv\Scripts\Activate.ps1
# Windows (CMD):
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### ▶️ Usage

Run the Flask app locally:

```bash
# Start the app
python app.py
```

Open your browser at: http://localhost:5000

---

## 🚢 Deployment

### Live (Render)
You can deploy the app to a hosting provider (Render, Heroku, etc.). Replace the URL below with your deployment link:

🔗 Oyasumi-Mind on Render — https://oyasumi-mind.onrender.com/

### Docker

Pull and run the published image:

```bash
# Pull the latest Docker image
docker pull mitsura11/oyasumi-mind:latest

# Run the container, mapping port 5000
docker run -p 5000:5000 mitsura11/oyasumi-mind:latest
```
after running docker image

The app will be available at: http://localhost:5000

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open a PR or an issue in the repository and follow the project's contribution guidelines.

---

## ⚖️ License

This project is licensed under the terms specified in the repository. See the LICENSE file for details.