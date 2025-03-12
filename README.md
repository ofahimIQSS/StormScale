# StormScale - AI-Powered Performance Testing Tool

StormScale is a powerful AI-driven performance testing tool that integrates **load testing, real-time system monitoring, deep learning-based performance prediction, and network traffic analysis**.

## Features
- **AI-Driven Load Testing** using Locust
- **Real-Time System Monitoring API** (CPU, Memory, Disk)
- **Deep Learning Performance Prediction**
- **Live API for Performance Insights** (`/status`)
- **Advanced Data Visualization** (Graphs CPU, memory, response times)
- **CI/CD Integration** for Automated Testing with GitHub Actions

## Installation Guide
1. Install dependencies:
```sh
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow pandas numpy locust scikit-learn psutil matplotlib flask requests
```
2. Clone the Repository:
```sh
git clone https://github.com/yourusername/StormScale.git
cd StormScale
```
3. Run StormScale:
```sh
python3 stormscale.py
```
Open in Browser:
- **Load Testing UI:** [http://localhost:8089](http://localhost:8089)
- **System Monitoring API:** [http://localhost:5000/status](http://localhost:5000/status)
