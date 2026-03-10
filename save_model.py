# save_model.py
# ─────────────────────────────────────────────────────────────
# Run this script in your Kaggle notebook AFTER training models
# to save the model package for the Streamlit app.
# ─────────────────────────────────────────────────────────────

import pickle

# Save trained model + scaler together
model_package = {
    'model'   : modelETR,       # your best model
    'scaler'  : mmscaler,       # your MinMaxScaler
    'features': list(df.drop('price', axis=1).columns)  # feature names
}

with open('flight_price_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("✅ Model saved as flight_price_model.pkl")
print(f"   Features: {model_package['features']}")

# ── Verify it loads correctly ─────────────────────────────────
with open('flight_price_model.pkl', 'rb') as f:
    loaded = pickle.load(f)

test_pred = loaded['model'].predict(
    loaded['scaler'].transform(x_test[:1])
)
print(f"   Test prediction: ₹{test_pred[0]:,.0f}  ✅ Working!")
