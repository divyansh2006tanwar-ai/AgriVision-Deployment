import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import os
import time
import gdown

# --- 1. SYSTEM CONFIGURATION & THEME ---
st.set_page_config(page_title="AgriVision AI Pro", layout="wide", page_icon="🌿")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1a1c24; padding: 20px; border-radius: 12px; border: 1px solid #2e313d; }
    .stSidebar { background-color: #0e1117; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VIRTUAL FARM DATABASE (MEMORY) ---
# This ensures the dashboard remembers the data between page clicks
if 'farm_df' not in st.session_state:
    # Create 100 baseline healthy sectors. Max yield is 50kg per sector (5000kg total)
    st.session_state['farm_df'] = pd.DataFrame({
        'Sector': [f"SEC-{i:03d}" for i in range(1, 101)],
        'Vitality': np.random.uniform(92, 100, 100),
        'Diagnosis': ['Baseline Healthy'] * 100,
        'Yield_kg': 50.0
    })
    st.session_state['scans_completed'] = 0

# --- 3. THE "BRAIN" ENGINE: WEIGHTS-BASED LOADER ---
@st.cache_resource
def load_resnet_engine():
    weights_path = 'plant_weights.weights.h5'
    
    # 🌟 ENTER YOUR GOOGLE DRIVE FILE ID HERE 🌟
    file_id = '1AlJ1MldX9Y1Mol18lFGIeYC1dAIHBCx2' 
    
    # If the file isn't on the laptop/server, download it from Google Drive!
    if not os.path.exists(weights_path):
        st.info("Downloading AI Engine from Cloud Storage... (This takes 1 minute)")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, weights_path, quiet=False)

    # Rebuild ResNet50
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    predictions = tf.keras.layers.Dense(38, activation='softmax') 
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions(x))

    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            return model
        except Exception as e:
            st.error(f"Incompatibility detected in weights file: {e}")
            return None
    return None

model = load_resnet_engine()

# --- 4. FULL 38-CLASS DATA MAPPING ---
CLASS_NAMES = [
    'Apple_Scab', 'Apple_Black_Rot', 'Cedar_Apple_Rust', 'Apple_Healthy', 'Blueberry_Healthy',
    'Cherry_Powdery_Mildew', 'Cherry_Healthy', 'Corn_Gray_Leaf_Spot', 'Corn_Common_Rust',
    'Corn_Northern_Leaf_Blight', 'Corn_Healthy', 'Grape_Black_Rot', 'Grape_Esca', 
    'Grape_Leaf_Blight', 'Grape_Healthy', 'Orange_Haunglongbing', 'Peach_Bacterial_Spot',
    'Peach_Healthy', 'Pepper_Bell_Bacterial_Spot', 'Pepper_Bell_Healthy', 'Potato_Early_Blight',
    'Potato_Late_Blight', 'Potato_Healthy', 'Raspberry_Healthy', 'Soybean_Healthy',
    'Squash_Powdery_Mildew', 'Strawberry_Leaf_Scorch', 'Strawberry_Healthy', 
    'Tomato_Bacterial_Spot', 'Tomato_Early_Blight', 'Tomato_Late_Blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_Leaf_Spot', 'Tomato_Spider_Mites', 'Tomato_Target_Spot', 
    'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Mosaic_Virus', 'Tomato_Healthy'
]

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("AgriVision v2.0")
    st.markdown("---")
    view = st.radio("Control Panel", ["🛰️ Dashboard", "🔬 Deep Learning Scanner"])
    st.info("B.Tech AIML: Divyansh Tanwar & Priyanshu")

# --- 6. MAIN INTERFACE ---
if model is None:
    st.error("⚠️ System Offline: `plant_weights.weights.h5` not detected or shape mismatch.")
    st.stop()

if view == "🛰️ Dashboard":
    st.header("🛰️ Agricultural Command Center")
    st.markdown("Global metrics and environmental vitality tracking (Live Data).")
    
    # Calculate Live Metrics from the Database
    df = st.session_state['farm_df']
    avg_vitality = df['Vitality'].mean()
    total_yield = df['Yield_kg'].sum()
    scans = st.session_state['scans_completed']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Global Farm Vitality", f"{avg_vitality:.1f}%")
    col2.metric("Sectors Updated by AI", f"{scans}/100")
    col3.metric("Projected Yield", f"{total_yield:,.0f} kg")
    
    # Live Dynamic Graph: Show the sectors with the lowest vitality (Highest Risk)
    st.subheader("⚠️ High-Risk Sectors Identified by AI")
    risk_df = df.sort_values(by='Vitality').head(10) # Get top 10 worst sectors
    
    fig = px.bar(risk_df, x='Sector', y='Vitality', color='Vitality', 
                 title="Sector Vitality Levels (Lower is worse)",
                 color_continuous_scale="RdYlGn") # Red to Green color scale
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Raw Farm Database"):
        st.dataframe(df, use_container_width=True)

elif view == "🔬 Deep Learning Scanner":
    st.header("🔬 Neural Network Batch Inference")
    st.markdown("Submit multiple field samples for high-precision, large-scale pathology detection.")
    
    uploaded_files = st.file_uploader("Upload Leaf Samples", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="main_scanner")
    
    if uploaded_files:
        st.info(f"📦 Batch Processing Initiated: {len(uploaded_files)} samples queued.")
        st.divider()
        
        for file in uploaded_files:
            c1, c2 = st.columns([1, 2])
            
            with c1:
                img = Image.open(file).convert('RGB')
                st.image(img, caption=file.name, use_container_width=True)
                
            with c2:
                with st.spinner(f"ResNet50 Analyzing {file.name}..."):
                    # Preprocessing
                    resized = img.resize((224, 224))
                    arr = np.array(resized) / 255.0
                    arr = np.expand_dims(arr, axis=0)
                    
                    # Live Inference
                    preds = model.predict(arr)
                    idx = np.argmax(preds)
                    conf = np.max(preds) * 100
                    diagnosis = CLASS_NAMES[idx].replace('_', ' ')

                # --- UPDATE THE DATABASE ---
                # We update the next available sector in the virtual farm
                target_idx = st.session_state['scans_completed'] % 100 
                st.session_state['farm_df'].at[target_idx, 'Diagnosis'] = diagnosis
                
                # EVS Logic: If diseased, health drops and yield is destroyed
                if "Healthy" in diagnosis:
                    st.session_state['farm_df'].at[target_idx, 'Vitality'] = conf # High health
                    st.success(f"✅ Diagnosis: {diagnosis} (Confidence: {conf:.2f}%)")
                else:
                    health_score = 100 - conf # High confidence in disease = Low health
                    if health_score < 10: health_score = 10 # Floor it at 10%
                    
                    st.session_state['farm_df'].at[target_idx, 'Vitality'] = health_score
                    # Destroy 60% of the yield for this diseased sector
                    st.session_state['farm_df'].at[target_idx, 'Yield_kg'] = 50.0 * 0.4 
                    
                    st.error(f"⚠️ Biological Stress Found: {diagnosis} (Confidence: {conf:.2f}%)")
                
                st.session_state['scans_completed'] += 1
            
            st.divider()