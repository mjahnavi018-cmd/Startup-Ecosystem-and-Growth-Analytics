# Indian-Startup-Ecosystem-Analytics
# India Startup Ecosystem Analytics

### How is India's startup ecosystem growing, and where are the emerging opportunities?

An end-to-end analytics project using **10 years of official DPIIT / Startup India data (2016–2025)** to understand the growth, geographic distribution, sectoral trends, state specialisation, and emerging state–sector hotspots of India's startup ecosystem.

## Problem

India's startup ecosystem has expanded rapidly, but growth is not uniform across states and sectors.

The project answers:

* Which states are growing fastest?
* Is startup activity becoming geographically decentralised?
* Which sectors are emerging?
* Which states have distinctive sector specialisations?
* Which state–sector combinations are potential hotspots?
* What could national startup recognitions look like in 2026–27?

> **Data note:** DPIIT recognitions represent registrations under the Startup India recognition scheme. They should not be interpreted as funding, revenue, profitability, or startup survival.

## Data

The analysis uses official **DPIIT / Startup India** recognition data covering:

* **2016–2025**
* **36 States / UTs**
* **56 Industries**
* **207,134 recognitions**
* **9,932 original records**

The raw data was cleaned and converted into a complete **state × industry × year panel**, expanding it to **20,160 records**.

## Approach

```text
DPIIT Data
     ↓
Data Cleaning & Preparation
     ↓
Python + SQL Analysis
     ↓
Growth & Geographic Concentration
     ↓
Sector Trends & Specialisation
     ↓
State Clustering
     ↓
State–Sector Hotspot Analysis
     ↓
Forecasting
     ↓
Interactive Dashboard
```

The analysis uses **CAGR, absolute growth, momentum, HHI, Top-5 share, Spearman correlation, Location Quotients, KMeans clustering, and time-series forecasting**.

A hotspot index combines **scale, momentum, and specialisation**, with five weighting schemes used to test ranking stability.

## Key Results

* DPIIT recognitions increased approximately **9×**, from **5,473 in 2017 to 49,430 in 2025**.
* The Top-5 state share declined from **63.6% to 53.1%**, indicating increasing geographic decentralisation.
* Geographic concentration showed a strong downward trend (**Spearman ρ = −0.97**).
* Smaller startup ecosystems generally showed higher relative growth (**ρ = −0.51, p = 0.02**).
* **Bihar** recorded the highest CAGR (~43%).
* **Maharashtra** recorded the highest absolute growth (+5,838 recognitions).
* AI recognitions increased from **309 in 2022 to 2,360 in 2025**.

### Leading State × Sector Hotspots

1. Karnataka × AI
2. Telangana × AI
3. Telangana × IT Services
4. Andhra Pradesh × IT Services
5. Rajasthan × Renewable Energy

## Forecasting

Four forecasting methods were compared using a **2023–25 holdout period**.

The best-performing model achieved **15% MAPE**, producing a 2026–27 recognition estimate of:

**48.7K–68.3K recognitions**

This is presented as a planning estimate because only nine annual observations are available.

## Interactive Dashboard

The complete analysis was developed into a **7-page Streamlit + Plotly dashboard** covering:

* Overview
* State Explorer
* Sector Explorer
* Growth Map
* Hotspot Finder
* Forecast
* Data & Method

The dashboard allows users to explore state growth, sector trends, specialisations, clusters, hotspots, and forecasts interactively.

## Limitations

DPIIT recognitions do not measure startup success, funding, or survival. Sector classifications are self-reported and may change over time. The registered state may not represent the actual operating location. Forecasting is also limited by the small number of annual observations.

## Conclusion

The project transforms official DPIIT startup-recognition data into a complete analytical framework covering **growth, geographic concentration, sector trends, state specialisation, clustering, hotspot identification, and forecasting**.

It demonstrates how a large public dataset can be converted into actionable insights on **where India's startup ecosystem is growing, how it is changing, and which state–sector combinations are emerging as potential hotspots**.
