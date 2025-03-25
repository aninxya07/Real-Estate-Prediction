import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('real_estate_model.joblib')  # Ensure the filename matches the saved model
scaler = joblib.load('scaler.joblib')

def main():
    st.set_page_config(
        page_title="Real Estate Price Prediction App",  # Title of the page
        page_icon="🏣",  # Graduation Cap emoji
        # You can use emoji or provide a path to an image file for the favicon
        layout="wide",  # You can set layout as 'wide' or 'centered'
    )

    # Sidebar for instructions
    with st.sidebar:
        st.markdown("<p style='color: red; font-size:36px;'><b>About App</b></p>", unsafe_allow_html=True)
        st.markdown("<p style=font-size:25px;'>📌 Instructions</p>", unsafe_allow_html=True)
        # st.markdown("## 📌 Instructions")
        st.sidebar.markdown(
            "- Fill in all fields accurately.\n"
            "- The model will predict the estimated house price.\n"
            "- Click **Predict Price** to see the result."
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:19px;'>A machine learning project to predict house prices per unit area based on key real estate factors. The model is trained using Random Forest Regressor, considering features like house age, distance to MRT, convenience stores, and location data.</p>", unsafe_allow_html=True)
        st.markdown("<br><hr><p style='text-align: center; font-size:18px;'>© 2025 Anindya Dolui & Pragati Das<br>All rights reserved.</p>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: red; font-size:55px;'>Real Estate Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])  # Adjust ratio if needed
    with col2:
        st.image("house.jpg", width=500)



    st.markdown("<p style='text-align: center; font-size:28px;'>Estimate the price of a house based on different factors</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # Input fields with corrected column names
    st.markdown("<p style='font-size:25px;'>🏡 House Features", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        X2_house_age = st.number_input("🧱 House Age (Years)", min_value=0, value=10)
        X3_distance_to_MRT = st.number_input("🚇 Distance to Nearest MRT (m)", min_value=0.0, value=500.0)
    
    with col2:
        X4_number_of_convenience_stores = st.number_input("🏪 Number of Convenience Stores", min_value=0, value=5)
        X5_latitude = st.number_input("🌍 Latitude", min_value=0.0, value=25.0)
        X6_longitude = st.number_input("🌍 Longitude", min_value=0.0, value=121.0)
    
    # Create DataFrame with correct feature names
    input_data = pd.DataFrame({
        'X2 house age': [X2_house_age],
        'X3 distance to the nearest MRT station': [X3_distance_to_MRT],
        'X4 number of convenience stores': [X4_number_of_convenience_stores],
        'X5 latitude': [X5_latitude],
        'X6 longitude': [X6_longitude]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Prediction button
    if st.button("🚀 Predict Price"):
        predicted_price = model.predict(input_scaled)
        st.success(f"💰 Estimated House Price: {predicted_price[0]:,.2f} per unit area")

if __name__ == '__main__':
    main()
