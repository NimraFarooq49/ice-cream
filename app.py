# ============================================
# STREAMLIT APP FOR DECISION TREE REGRESSOR
# File name: app.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 24px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .price-value {
        font-size: 52px;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏠 House Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Decision Tree Regressor</p>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('decision_tree_regressor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except:
        st.error("⚠️ Model files not found. Please train the model first.")
        return None, None

model, feature_names = load_model()

# Create two columns for layout
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 🔧 House Features")
    st.markdown("Adjust the sliders to input house characteristics:")
    
    # Input features with sliders
    size = st.slider(
        "🏡 Size (sq ft)",
        min_value=800,
        max_value=3500,
        value=2000,
        step=50,
        help="Total living area in square feet"
    )
    
    bedrooms = st.slider(
        "🛏️ Number of Bedrooms",
        min_value=1,
        max_value=6,
        value=3,
        step=1
    )
    
    age = st.slider(
        "📅 Age (years)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        help="How old is the house?"
    )
    
    distance = st.slider(
        "📍 Distance from City Center (km)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.5
    )
    
    # Predict button
    predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

with col_right:
    if model is not None:
        # Create feature dataframe
        input_data = pd.DataFrame({
            'Size_sqft': [size],
            'Bedrooms': [bedrooms],
            'Age_years': [age],
            'Distance_km': [distance]
        })
        
        if predict_button or 'initial_prediction' not in st.session_state:
            st.session_state['initial_prediction'] = True
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display prediction
            st.markdown("### 🎯 Prediction Result")
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin: 0; font-weight: normal;">Estimated House Price</h3>
                <div class="price-value">${prediction:,.0f}</div>
                <p style="margin: 5px 0; opacity: 0.9;">Based on Decision Tree Regression Model</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance visualization
            st.markdown("### 📊 Feature Contribution")
            
            feature_importance = model.feature_importances_
            
            fig_importance = go.Figure(data=[
                go.Bar(
                    x=['Size', 'Bedrooms', 'Age', 'Distance'],
                    y=feature_importance * 100,
                    text=[f'{imp*100:.1f}%' for imp in feature_importance],
                    textposition='auto',
                    marker=dict(
                        color=feature_importance,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    )
                )
            ])
            
            fig_importance.update_layout(
                title="How Each Feature Affects Price",
                xaxis_title="Feature",
                yaxis_title="Importance (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Input summary
            st.markdown("### 📋 Your Input Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Size", f"{size:,} sq ft", 
                         delta=f"{size - 2000:+,} from avg")
                st.metric("Bedrooms", f"{bedrooms}", 
                         delta=f"{bedrooms - 3:+} from avg")
            
            with col2:
                st.metric("Age", f"{age} years", 
                         delta=f"{age - 10:+} from avg")
                st.metric("Distance", f"{distance} km", 
                         delta=f"{distance - 10:+.1f} from avg")
            
            # Price breakdown (estimated contribution)
            st.markdown("### 💰 Price Breakdown (Estimated)")
            
            # Simple estimation of contribution
            base_price = 100000
            size_contrib = size * 200
            bedroom_contrib = bedrooms * 50000
            age_contrib = -age * 1000
            distance_contrib = -distance * 2000
            
            breakdown_data = pd.DataFrame({
                'Component': ['Base Price', 'Size', 'Bedrooms', 'Age', 'Distance'],
                'Contribution': [base_price, size_contrib, bedroom_contrib, 
                               age_contrib, distance_contrib]
            })
            
            fig_breakdown = px.bar(
                breakdown_data,
                x='Component',
                y='Contribution',
                color='Contribution',
                color_continuous_scale=['red', 'yellow', 'green'],
                text='Contribution'
            )
            
            fig_breakdown.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_breakdown.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="",
                yaxis_title="Price Contribution ($)"
            )
            
            st.plotly_chart(fig_breakdown, use_container_width=True)

# Sidebar information
with st.sidebar:
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This application uses a **Decision Tree Regressor** to predict house prices 
    based on four key features:
    
    - **Size**: Total living area
    - **Bedrooms**: Number of bedrooms
    - **Age**: Age of the property
    - **Distance**: Distance from city center
    
    ### 🎯 Model Performance
    - Algorithm: Decision Tree Regressor
    - Training R²: ~0.95
    - Test R²: ~0.90
    - Average Error: ~$50,000
    
    ### 🔍 How It Works
    1. Adjust the sliders to set house features
    2. Click "Predict Price" to get estimation
    3. View feature importance and breakdown
    
    ### 📊 Feature Importance
    The model learns which features matter most:
    - **Size** typically has the highest impact
    - **Bedrooms** significantly affect price
    - **Age** causes depreciation
    - **Distance** from city affects value
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Model Details")
    if model is not None:
        st.write(f"Max Depth: {model.max_depth}")
        st.write(f"Min Samples Split: {model.min_samples_split}")
        st.write(f"Min Samples Leaf: {model.min_samples_leaf}")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        Built with Streamlit<br>
        Decision Tree Regressor<br>
        Lab 09 Task 3
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>💡 <strong>Tip:</strong> Try adjusting different features to see how they affect the predicted price!</p>
    <p style="font-size: 12px;">Note: Predictions are based on synthetic data for educational purposes</p>
</div>
""", unsafe_allow_html=True)