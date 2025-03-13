import os
import time
import json
import locust
import logging
import pandas as pd
import requests
import numpy as np
from locust import HttpUser, task, between
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import psutil
import matplotlib.pyplot as plt
import threading
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from flask import Flask, jsonify, request
import socket
import seaborn as sns
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StormScaleUser(HttpUser):
    wait_time = between(1, 5)
    
    def on_start(self):
        self.login()
    
    def login(self):
        payload = {"username": "normaluser1", "password": "admin1"}
        response = self.client.post("/api/login", json=payload)
        if response.status_code == 200:
            self.token = response.json().get("token")
            logging.info("Login successful")
        else:
            logging.error("Login failed")
            self.token = None
    
    @task(1)
    def create_dataverse(self):
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            data = {"name": "TestDataverse", "description": "Performance Testing Dataverse"}
            response = self.client.post("/api/dataverse", headers=headers, json=data)
            if response.status_code == 201:
                logging.info("Dataverse created successfully")
            else:
                logging.error("Failed to create Dataverse")
    
    @task(2)
    def create_dataset(self):
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            data = {"title": "TestDataset", "description": "AI-powered Performance Test Dataset"}
            response = self.client.post("/api/dataset", headers=headers, json=data)
            if response.status_code == 201:
                logging.info("Dataset created successfully")
            else:
                logging.error("Failed to create Dataset")

# AI-Based Performance Prediction
class StormScalePerformanceAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = RandomForestRegressor()
    
    def train_model(self):
        df = pd.read_csv(self.dataset)
        X = df[['requests_per_second', 'cpu_usage', 'memory_usage', 'response_time']]
        y = df['success_rate']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        logging.info("AI Model Trained for Predicting Performance Bottlenecks")
    
    def predict(self, X_new):
        return self.model.predict([X_new])

# Advanced Deep Learning Model for Performance Prediction
def build_lstm_model():
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(10, 4)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# System Resource Monitoring
def monitor_system_resources():
    while True:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        network = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        logging.info(f"CPU: {cpu}% | Memory: {memory}% | Disk: {disk}% | Network: {network} bytes")
        time.sleep(5)

# API for Real-Time Performance Monitoring
app = Flask(__name__)

@app.route('/status', methods=['GET'])
def get_system_status():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    return jsonify({"cpu": cpu, "memory": memory, "disk": disk})

@app.route('/advanced-metrics', methods=['GET'])
def get_advanced_metrics():
    df = pd.read_csv("performance_data.csv")
    stats = {
        "avg_response_time": df['response_time'].mean(),
        "peak_cpu_usage": df['cpu_usage'].max(),
        "peak_memory_usage": df['memory_usage'].max(),
        "total_requests": len(df)
    }
    return jsonify(stats)

# Auto-scaling performance testing
def auto_scale_tests():
    while True:
        current_cpu = psutil.cpu_percent()
        if current_cpu < 50:
            os.system("locust --users=100 --spawn-rate=10 --run-time=2m")
        elif current_cpu < 80:
            os.system("locust --users=50 --spawn-rate=5 --run-time=2m")
        time.sleep(60)

# Network Monitoring
def get_local_ip():
    return "127.0.0.1"

if __name__ == "__main__":
    logging.info("Starting StormScale - AI-Powered Performance Testing Tool")
    threading.Thread(target=monitor_system_resources, daemon=True).start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000), daemon=True).start()
    threading.Thread(target=auto_scale_tests, daemon=True).start()
    logging.info("System monitoring API available at http://127.0.0.1:5000/status")
    os.system("locust -f stormscale.py")
