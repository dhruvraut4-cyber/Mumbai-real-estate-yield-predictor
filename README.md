# 🏢 Mumbai Real Estate Rental Yield & Investment ROI Evaluator

A Master's-grade Data Science portfolio project designed to predict rental yields, map spatial transit networks, compute intrinsic asset valuations, and forecast multi-year investment returns across Mumbai micro-markets.

## 🚀 Key Features
- **LightGBM Rental Predictor:** Accurately estimates monthly rental yields based on carpet area, BHK, bathroom count, property age, furnishing status, and spatial proximity.
- **Comprehensive Transit Mapping:** Calculates Haversine distances against the complete Mumbai Suburban Railway network (Western, Central, Harbour lines) and Mumbai Metro corridors (Lines 1, 2A, 3 Aqua, and 7).
- **Auto-Micro-Market Growth Profiling:** Dynamically fetches locality-tier appreciation rates and rental escalations based on infrastructure expansions (e.g., MTHL, Navi Mumbai Airport, Metro lines).
- **Feasible Value Deal Assessment:** Computes intrinsic fair market pricing using the Income Approach to evaluate whether an asking price is undervalued, fairly priced, or overvalued.
- **Multi-Year ROI & CAGR Simulation:** Projects cumulative cash flows and future resale values over user-defined investment horizons (1–20 years).
- **Explainable AI (XAI):** Uses native TreeSHAP (`pred_contrib`) to provide exact feature contribution breakdowns in Indian Rupees for every valuation.

## 🛠️ Tech Stack
- **Python / Machine Learning:** `LightGBM`, `Scikit-Learn`, `Pandas`, `NumPy`, `Joblib`
- **Web Application Framework:** `Streamlit`
- **Spatial Analytics:** Haversine great-circle distance computations

## ⚙️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/dhruvraut4-cyber/Mumbai-real-estate-yield-predictor.git](https://github.com/dhruvraut4-cyber/Mumbai-real-estate-yield-predictor.git)
   cd Mumbai-real-estate-yield-predictor
   
