# ✈️ Flight Price Predictor

A Machine Learning web app to predict Indian domestic flight prices, built with **Streamlit** and **ExtraTreesRegressor**.

---

## 🚀 Live Demo
Deploy on [Streamlit Cloud](https://share.streamlit.io) for free!

---

## 📁 Project Structure
```
flight_price_app/
│
├── app.py                  ← Streamlit web app
├── flight_price_model.pkl  ← Trained ML model (generate from notebook)
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## ⚙️ Setup & Run Locally

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/flight-price-app.git
cd flight-price-app
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add your model file
Run your Kaggle notebook and save the model:
```python
import pickle
model_package = {
    'model'  : modelETR,
    'scaler' : mmscaler,
    'features': list(df.drop('price', axis=1).columns)
}
with open('flight_price_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)
```
Place `flight_price_model.pkl` in the same folder as `app.py`.

### Step 4 — Run the app
```bash
streamlit run app.py
```
Opens at: `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New App** → select your repo → `main` → `app.py`
5. Click **Deploy** ✅

---

## 🧠 Model Info

| Item | Detail |
|---|---|
| Algorithm | ExtraTreesRegressor |
| Dataset | Indian Domestic Flights (Kaggle) |
| Features | Airline, Cities, Class, Stops, Duration, Days Left |
| Metrics | R² ≈ 0.985, MAPE ≈ 7.67% |

---

## 📊 Features Used
- Airline
- Source City & Destination City
- Departure Time & Arrival Time
- Number of Stops
- Flight Duration
- Days Left for Departure
- Class (Economy / Business)

---

## 🛠️ Tech Stack
- Python
- Streamlit
- scikit-learn
- XGBoost
- Pandas / NumPy
