# Bangladesh-air-quality-forecasting


An end-to-end data science pipeline for short-term air quality forecasting across major Bangladeshi cities using classical time-series models and rolling-origin validation.

---

## Overview

Air pollution is a major environmental and public health challenge in South Asia. Fine particulate matter (PM2.5) is particularly harmful due to its ability to penetrate deep into the respiratory system.

This project implements a **reproducible forecasting system** designed to generate short-horizon predictions of PM2.5 concentrations across major Bangladeshi cities.

The system integrates:

- Automated environmental data ingestion
- Structured database storage
- Time-series forecasting models
- Walk-forward validation
- Multi-city performance comparison

The goal is to evaluate whether interpretable statistical models can provide reliable short-term pollution forecasts in urban environments.

---

## Cities Analyzed

The forecasting pipeline was applied to eight major Bangladeshi cities.

| City | Latitude | Longitude |
|-----|-----|-----|
| Dhaka | 23.8103 | 90.4125 |
| Chattogram | 22.3569 | 91.7832 |
| Khulna | 22.8456 | 89.5403 |
| Rajshahi | 24.3745 | 88.6042 |
| Sylhet | 24.8949 | 91.8687 |
| Barishal | 22.7010 | 90.3535 |
| Rangpur | 25.7439 | 89.2752 |
| Mymensingh | 24.7471 | 90.4203 |

---

## System Architecture

The forecasting system follows a structured data pipeline.


Open-Meteo Air Quality API
           ↓
Python Data Ingestion Pipeline
           ↓
Normalization & Validation
           ↓
UTC Timestamp Standardization
           ↓
SQLite Database Storage
           ↓
Time-Series Dataset Extraction
           ↓
Forecasting Models (ARIMA / SARIMA)
           ↓
Rolling-Origin Validation
           ↓
Multi-City Model Evaluation
           ↓
Visualization & Results


This architecture ensures the forecasting workflow is **reproducible and modular**.

---

## Data Source

Air-quality observations were retrieved from:

**Open-Meteo Environmental API**

The following pollutants are collected:

- PM2.5  
- PM10  
- Carbon monoxide  
- Nitrogen dioxide  
- Ozone  

Data is retrieved as **hourly observations** for each city.

---

## Database Design

Observations are stored in a structured SQLite database.

**Table**


air_quality_data


**Primary Key**


(city_name, timestamp_utc)


This composite key ensures:

- Idempotent ingestion
- Duplicate prevention
- Chronological consistency

---

## Dataset Preparation

The forecasting dataset is constructed by:

1. Extracting PM2.5 observations from the database  
2. Sorting observations chronologically  
3. Removing missing values  
4. Creating a time-indexed series  

This produces a **univariate time series suitable for autoregressive forecasting models**.

---

## Forecasting Models

Two classical statistical models were evaluated.

### ARIMA

Autoregressive Integrated Moving Average


ARIMA(p,d,q)


Used to capture short-term temporal dependence.

Baseline configuration:


ARIMA(2,1,2)


---

### SARIMA

Seasonal Autoregressive Integrated Moving Average


SARIMA(p,d,q)(P,D,Q)s


Because the dataset contains hourly observations:


s = 24 hours


Baseline specification:


SARIMA(1,1,1)(1,1,1)24


---

## Validation Strategy

Model performance is evaluated using **rolling-origin (walk-forward) validation**.

This approach simulates real-time forecasting conditions.

Procedure:

1. Train model on historical observations  
2. Forecast next time step  
3. Reveal true observation  
4. Expand training window  
5. Repeat

This prevents **future data leakage**, which is critical in time-series evaluation.

---

## Evaluation Metrics

Forecast accuracy was measured using:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

These metrics quantify the magnitude of prediction error.

---

## Results

Forecast accuracy varies across cities.

| City | Best Model | RMSE | MAE |
|-----|-----|-----|-----|
| Barishal | SARIMA | 3.27 | 2.54 |
| Chattogram | SARIMA | 1.53 | 1.09 |
| Dhaka | ARIMA | 6.53 | 4.55 |
| Khulna | ARIMA | 5.12 | 2.99 |
| Mymensingh | SARIMA | 7.18 | 5.02 |
| Rajshahi | ARIMA | 11.40 | 8.23 |
| Rangpur | ARIMA | 5.00 | 4.08 |
| Sylhet | SARIMA | 1.92 | 1.33 |

Key finding:

No single model consistently outperformed across all cities.

Urban environments with stronger daily pollution cycles benefited from **seasonal models**, while cities with more irregular variation favored **ARIMA models**.

---

## Example Forecast

Example rolling-origin forecast for Dhaka.

*(Insert forecast figure here)*

---

## Multi-City Model Comparison

Model performance differences across cities.

*(Insert RMSE comparison figure here)*

These results highlight how pollution dynamics differ across geographic locations.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Statsmodels
- SQLite
- Matplotlib

---

## How to Run

Clone the repository


git clone https://github.com/yourusername/air-quality-forecast-bangladesh


Install dependencies


pip install -r requirements.txt


Run ingestion pipeline


python scripts/ingest_air_quality_data.py


Run forecasting models


python scripts/forecast_model_arima.py
python scripts/forecast_model_sarima.py


Run validation


python scripts/forecast_validation_multicity.py


Generate comparison plots


python scripts/multicity_metrics_plot.py


---

## Project Contributions

This project was independently designed and implemented as a complete data science pipeline.

Key contributions include:

- Automated environmental data ingestion
- Database schema design
- Implementation of ARIMA and SARIMA forecasting models
- Rolling-origin validation framework
- Multi-city forecasting evaluation
- Visualization of model performance

---

## Limitations

Several limitations remain:

- Short historical observation window  
- Meteorological variables not included  
- Pollution sources not explicitly modeled  
- Forecasts limited to short time horizons  

Future work may incorporate **multivariate forecasting models and meteorological predictors**.

---

## Author

**Meherab Hossain Shafin**  
Daffodil International University  
Dhaka, Bangladesh

---

## License

This project is released under the MIT License.
