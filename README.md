# 🍔 Zomato Delivery Time Prediction

![Python](https://img.shields.io/badge/Python-3.11.15-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)
![License](https://img.shields.io/badge/License-Apache-lightgrey.svg)

> A streamlined machine learning pipeline to predict food delivery times on Zomato using a **Stacking Regressor** (combining **LightGBM** and **Random Forest**) with a simple Flask REST API for real-time predictions.

---

##  Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Highlights](#-project-highlights)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [API Usage](#-api-usage)
- [Results](#-results)
- [License](#-license)

---

## 🎯 Problem Statement

Food delivery platforms like Zomato operate in highly dynamic environments where delivery time depends on numerous factors — distance, weather, traffic density, vehicle type, and delivery partner ratings. Accurately predicting delivery time:

- Improves **customer satisfaction** by setting realistic expectations
- Helps **restaurants optimize** order preparation timing
- Allows **platform operations** to allocate delivery partners efficiently

This project builds a regression model to predict delivery time (in minutes) given real-world features extracted from Zomato delivery data.

---

## ✨ Project Highlights

- **Streamlined ML pipeline**: straightforward data ingestion, preprocessing, training, and evaluation
- **Stacking Regressor**: Combines **LightGBM** for speed and accuracy with **Random Forest** for robustness
- **Flask REST API**: A lightweight web interface serving real-time predictions
- **Modular Codebase**: Exception handling, logging, and separated data processing logic

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11.15 |
| ML Models | LightGBM, Random Forest, Scikit-Learn |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

##  Project Structure

```
zomato-delivery-time-prediction/
│
├── .venv/                         # Python virtual environment
├── data/
│   ├── raw/                       # Raw ingested data
│   ├── processed/                 # Cleaned and feature-engineered data
│   └── final/                     # Train/test splits ready for modeling
│
├── models/                        # Saved model artifacts (.joblib)
├── notebooks/                     # Exploratory Data Analysis & Model Training
├── src/
│   ├── data/                      # Data ingestion and cleaning scripts
│   ├── features/                  # Data preprocessing and feature engineering
│   └── pipeline/                  # Inference and prediction pipelines
│
├── static/                        # Frontend stylesheets and JS
├── templates/                     # Flask prediction web UI (index.html)
│
├── app.py                         # Flask application entry point
├── predict.py                     # Standalone CLI prediction script
├── pyproject.toml                 # Project metadata and requirements definition
├── requirements.txt               # Locked Python dependencies
└── README.md                      # Project documentation
```

---

## 📊 Dataset

The dataset contains historical Zomato delivery records with the following key features:

| Feature | Description |
|---|---|
| `Delivery_person_Age` | Age of the delivery partner |
| `Delivery_person_Ratings` | Average ratings (1–6 scale) |
| `Restaurant_latitude/longitude` | Restaurant GPS coordinates |
| `Delivery_location_latitude/longitude` | Customer GPS coordinates |
| `Weather_conditions` | Weather at time of delivery |
| `Road_traffic_density` | Traffic level (Low / Medium / High / Jam) |
| `Vehicle_condition` | Condition score of delivery vehicle |
| `Type_of_order` | Food category (Snack, Meal, Drinks, etc.) |
| `Type_of_vehicle` | Vehicle type (Bike, Scooter, etc.) |
| `multiple_deliveries` | Number of simultaneous deliveries |
| `Festival` | Whether delivery was during a festival |
| `City` | City type (Metropolitan, Urban, Semi-Urban) |
| **`Time_taken(min)`** | **Target: Delivery time in minutes** |

---

##  Installation & Setup

### Prerequisites

- Python 3.11.15
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/quamrl-hoda/zomato-delivery-time-prediction.git
cd zomato-delivery-time-prediction
```

### 2. Create Virtual Environment

```bash
# Using standard Python
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Running the Project

### Train the Model

You can retrain the machine learning models by executing the code found in the model training notebooks (e.g., `notebooks/finalEstimator.ipynb`) which will generate your `.joblib` model artifacts into the `models/` directory.

### Launch Flask Web App

```bash
python app.py
```

Visit `http://localhost:5000` to interact with the prediction UI.

---

##  API Usage

Once the Flask app is running, you can hit the prediction endpoint:

### POST `/predict`

**Request Body (JSON):**

```json
{
  "Delivery_person_Age": 29,
  "Delivery_person_Ratings": 4.8,
  "Restaurant_latitude": 22.745049,
  "Restaurant_longitude": 75.892471,
  "Delivery_location_latitude": 22.765049,
  "Delivery_location_longitude": 75.912471,
  "Weather_conditions": "Sunny",
  "Road_traffic_density": "High",
  "Vehicle_condition": 2,
  "Type_of_order": "Meal",
  "Type_of_vehicle": "motorcycle",
  "multiple_deliveries": 0,
  "Festival": "No",
  "City": "Metropolitian",
  "time_to_pickup_minutes": 8
}
```

**Response:**

```json
{
  "predicted_delivery_time_minutes": 27.4,
  "status": "success"
}
```

---

##  Results

| Metric | Train | Validation |
|---|---|---|
| RMSE | 4.21 min | 4.87 min |
| MAE | 3.18 min | 3.74 min |
| R² Score | 0.834 | 0.812 |

### Key Feature Importances

1. `distance_km` (Haversine)
2. `Road_traffic_density`
3. `time_to_pickup_minutes`
4. `Weather_conditions`
5. `Delivery_person_Ratings`
6. `multiple_deliveries`
7. `City`

---

##  License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

---
