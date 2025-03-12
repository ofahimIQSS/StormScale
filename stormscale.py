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
from flask import Flask, jsonify
import socket

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StormScaleUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(1)
    def homepage(self):
        response = self.client.get("/")
        if response.status_code == 200:
            logging.info("Homepage loaded successfully")
        else:
            logging.error("Homepage load failed")
    
    @task(2)
    def search(self):
        response = self.client.get("/search?q=performance")
        if response.status_code == 200:
            logging.info("Search completed successfully")
        else:
            logging.error("Search failed")

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

# Network Monitoring
def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

if __name__ == "__main__":
    logging.info("Starting StormScale - AI-Powered Performance Testing Tool")
    threading.Thread(target=monitor_system_resources, daemon=True).start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000), daemon=True).start()
    logging.info(f"System monitoring API available at http://{get_local_ip()}:5000/status")
    os.system("locust -f stormscale.py")
