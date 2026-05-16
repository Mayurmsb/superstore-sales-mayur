import streamlit as st
import pickle
import numpy as np

# ============================================================
# CAR PRICE PREDICTION APP
# Submitted By : Mayur Bavaskar
# ============================================================

# Load the trained model
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App Title
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)
st.title('🚗 Car Price Prediction App')
st.markdown('#### Internship Task 9 | Submitted By: Mayur Bavaskar')
st.markdown('---')

st.markdown('### Enter Car Details Below:')

# Input Fields
col1, col2 = st.columns(2)

with col1:
    location     = st.selectbox('Location', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  format_func=lambda x: ['Ahmedabad','Bangalore','Chennai','Coimbatore','Delhi','Hyderabad','Jaipur','Kochi','Kolkata','Mumbai'][x])
    fuel_type    = st.selectbox('Fuel Type', [0, 1, 2, 3, 4],
                                  format_func=lambda x: ['CNG','Diesel','Electric','LPG','Petrol'][x])
    transmission = st.selectbox('Transmission', [0, 1],
                                  format_func=lambda x: ['Automatic','Manual'][x])
    owner_type   = st.selectbox('Owner Type', [0, 1, 2, 3],
                                  format_func=lambda x: ['First','Fourth & Above','Second','Third'][x])

with col2:
    km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=1000000, value=50000, step=1000)
    mileage   = st.number_input('Mileage (kmpl)', min_value=0.0, max_value=50.0, value=18.0, step=0.1)
    engine    = st.number_input('Engine (CC)', min_value=500.0, max_value=6000.0, value=1200.0, step=100.0)
    power     = st.number_input('Power (bhp)', min_value=30.0, max_value=500.0, value=90.0, step=5.0)

seats     = st.slider('Number of Seats', 2, 10, 5)
new_price = st.number_input('New Price (Lakhs)', min_value=0.0, max_value=200.0, value=10.0, step=0.5)
car_age   = st.slider('Car Age (Years)', 0, 30, 5)

st.markdown('---')

# Predict Button
if st.button('Predict Car Price 🚀'):
    input_data = np.array([[location, km_driven, fuel_type, transmission, owner_type,
                             mileage, engine, power, seats, new_price, car_age]])

    predicted_price = model.predict(input_data)[0]

    st.success(f'### 💰 Estimated Car Price: ₹ {predicted_price:.2f} Lakhs')

st.markdown('---')
st.markdown('##### Powered by Random Forest Regressor | Task 9 - Synent Internship')


