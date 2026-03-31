import pandas as pd
import numpy as np
import streamlit as st
import joblib


kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛒 Customer Segmentation App")
st.write("Wprowadź dane klienta, aby przypisać go do odpowiedniego segmentu marketingowego.")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Wiek (Age)", min_value=18, max_value=100, value=35)
    income = st.number_input("Roczny dochód (Income)", min_value=0, max_value=200000, value=50000)
    total_spending = st.number_input("Całkowite wydatki (Total Spending)", min_value=0, max_value=5000, value=1000)

with col2:
    num_web_purchases = st.number_input("Zakupy online (Web Purchases)", min_value=0, max_value=100, value=10)
    num_store_purchases = st.number_input("Zakupy stacjonarne (Store Purchases)", min_value=0, max_value=100, value=10)
    num_web_visits = st.number_input("Wizyty na stronie/msc (Web Visits)", min_value=0, max_value=50, value=3)
    recency = st.number_input("Dni od ostatnich zakupów (Recency)", min_value=0, max_value=365, value=30)


input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})


if st.button("Przewidź Segment", type="primary"):
    
    input_scaled = scaler.transform(input_data)
    
    cluster = kmeans.predict(input_scaled)[0]
    
    st.success(f"🎉 Przypisany segment: Klaster {cluster}")
    
    
    st.subheader("Profil Segmentu:")
    if cluster == 0:
        st.info("Klaster 0: Wysoki budżet, częste wizyty na stronie (Klienci Premium Web).")
    elif cluster == 1:
        st.info("Klaster 1: Średni dochód, rzadkie zakupy (Wymagają kampanii aktywizującej).")
    elif cluster == 2:
        st.info("Klaster 2: Klienci stacjonarni o wysokich wydatkach.")
    elif cluster == 3:
        st.info("Klaster 3: Młodzi klienci, niskie wydatki, wysoka aktywność online.")
    elif cluster == 4:
        st.info("Klaster 4: 'Łowcy okazji' - rzadkie zakupy stacjonarne.")
    else:
        st.info("Klaster 5: Lojalni klienci zrównoważeni (kupują i tu, i tu).")
