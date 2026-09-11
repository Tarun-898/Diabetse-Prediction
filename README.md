# Diabetes Prediction MLOps Project

An end-to-end Machine Learning and MLOps project for predicting diabetes risk from patient health and lifestyle information.

The project goes beyond simply training a machine learning model. It packages the application with Docker, deploys it on Kubernetes using Helm, adds Horizontal Pod Autoscaling, and integrates Prometheus and Grafana for monitoring.

The main goal of this project was to understand how a machine learning application can be taken from a trained model to a containerized and monitored production-style deployment.

---

## Project Overview

This project provides a web application where users can enter patient-related information and receive a diabetes risk prediction.

The application uses a trained machine learning pipeline that includes preprocessing and prediction.

The complete deployment flow is:

```text
Dataset
   |
   v
Data Preprocessing
   |
   v
Model Training
   |
   v
Saved ML Pipeline
   |
   v
Flask Web Application
   |
   v
Docker Container
   |
   v
Docker Hub
   |
   v
Kubernetes
   |
   +-------------------+
   |                   |
   v                   v
Deployment           Service
   |
   v
Horizontal Pod Autoscaler
   |
   v
Prometheus
   |
   v
Grafana





Features:

Diabetes risk prediction using Machine Learning
Dataset-aligned input fields
Saved preprocessing pipeline
Flask-based web application
Dockerized application
Docker image available through Docker Hub
Kubernetes deployment
Kubernetes NodePort service
Helm chart for Kubernetes deployment
Horizontal Pod Autoscaling (HPA)
CPU and memory resource requests and limits
Prometheus monitoring
Grafana monitoring dashboards
Kubernetes metrics using Metrics Server
Kubernetes workload monitoring using kube-state-metrics


Tech Stack:

Machine Learning
Python
Pandas
NumPy
Scikit-learn
Joblib


Backend / Application:

Flask
HTML
CSS


Containerization:

Docker
Docker Hub


Kubernetes:

Kubernetes
kind
kubectl
Helm


Monitoring:

Prometheus
Grafana
Metrics Server
kube-state-metrics


Development Environment:

Windows
PowerShell
Docker Desktop
