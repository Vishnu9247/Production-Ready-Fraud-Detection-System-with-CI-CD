# Production-Ready Credit Card Fraud Detection System with CI/CD

A production-ready machine learning system for real-time credit card fraud detection built using Python, FastAPI, Docker, GitHub Actions, and Azure Container Apps.

This project automates the complete ML lifecycle including:

* data preprocessing
* feature engineering
* model training and hyperparameter tuning
* model versioning
* Docker image creation
* deployment to Azure Container Apps
* CI/CD automation using GitHub Actions

---

# Project Architecture

```text
Raw CSV Data
      ↓
Preprocessing Pipeline
      ↓
Feature Engineering
      ↓
Model Training + Hyperparameter Tuning
      ↓
Best Model Selection
      ↓
Model Versioning & Archiving
      ↓
FastAPI Inference Service
      ↓
Docker Image
      ↓
Azure Container Registry
      ↓
Azure Container Apps
      ↓
Public REST API
```

---

# Tech Stack

## Machine Learning

* Python
* Scikit-learn
* XGBoost
* SMOTE
* Pandas
* NumPy

## API & Deployment

* FastAPI
* Docker
* Azure Container Registry
* Azure Container Apps

## Automation & MLOps

* GitHub Actions
* CI/CD Pipelines
* Model Versioning

---

# Key Features

* Automated preprocessing pipeline
* Advanced feature engineering
* Fraud-focused model optimization using Recall
* Hyperparameter tuning with RandomizedSearchCV
* Automatic model versioning
* Automatic archival of old models
* REST API deployment using FastAPI
* Docker containerization
* Automated Azure deployment pipeline
* CI/CD workflow triggered by new staging data

---

# Dataset

The dataset contains simulated credit card transactions with fraud labels.

## Sample Features

| Feature                | Description           |
| ---------------------- | --------------------- |
| amt                    | Transaction amount    |
| city_pop               | City population       |
| category               | Merchant category     |
| lat / long             | Customer coordinates  |
| merch_lat / merch_long | Merchant coordinates  |
| trans_date_trans_time  | Transaction timestamp |
| is_fraud               | Fraud label           |

---

# Feature Engineering

The preprocessing pipeline generates additional fraud-focused features:

* transaction hour
* weekend indicator
* late-night transaction indicator
* log-transformed transaction amount
* z-score normalization
* high amount transaction flag
* customer age
* geographic transaction distance
* smoothed target encoding
* log-transformed distance features

---

# Model Training

The system trains and tunes multiple models:

* XGBoost Classifier
* HistGradientBoostingClassifier

Hyperparameter tuning is performed using:

```python
RandomizedSearchCV
```

with:

```python
StratifiedKFold
```

The best model is selected based on:

```text
Recall Score
```

because fraud detection prioritizes minimizing false negatives.

---

# Model Versioning

When a new model is generated:

* existing model is archived
* new model version is created automatically

Example:

```text
best_model_v1.pkl
best_model_v2.pkl
best_model_v3.pkl
```

Archived models are stored inside:

```text
Models/archive/
```

---

# FastAPI REST API

The trained model is exposed through a FastAPI inference API.

## Endpoint

```http
POST /predict
```

## Example Request

```json
{
  "amt": 120.5,
  "hour": 3,
  "distance_from_home": 150.2
}
```

## Example Response

```json
{
  "prediction": 1
}
```

---

# Docker Deployment

The FastAPI application is containerized using Docker.

## Build Image

```bash
docker build -t creditcard_fraud_mlmodel ./App
```

## Run Container

```bash
docker run -p 8000:8000 creditcard_fraud_mlmodel
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# Azure Deployment

The Docker image is deployed to:

* Azure Container Registry (ACR)
* Azure Container Apps

The deployed application is publicly accessible through a REST API endpoint.

---

# CI/CD Pipeline

GitHub Actions automates the workflow.

## Workflow Trigger

The pipeline automatically starts when a new file is added to:

```text
Data Files/Staging/
```

## Automated Pipeline

```text
New Data File
    ↓
Preprocessing
    ↓
Model Training
    ↓
Hyperparameter Tuning
    ↓
Model Replacement
    ↓
Docker Build
    ↓
Push to Azure Container Registry
    ↓
Deploy to Azure Container Apps
```

---

# Project Structure

```text
Production-Ready Fraud Detection System with CI CD
│
├── App/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── *.pkl
│
├── Data Files/
│   ├── Raw Files/
│   ├── Processed Files/
│   └── Staging/
│
├── Models/
│   └── archive/
│
├── Python_Scripts/
│   ├── preprocess.py
│   └── train.py
│
├── .github/
│   └── workflows/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Future Improvements

* MLflow experiment tracking
* Drift detection
* Automated monitoring
* Batch inference pipelines
* Kubernetes deployment
* Feature store integration
* Real-time streaming inference with Kafka

---

# Author

Vishnu Vardhan Reddy Alla

```text
Data Science | Machine Learning | MLOps | Azure
```
