import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components

# Title
st.title("🌾 Smart Irrigation Predictor")
st.write("Predict whether to water the crop based on environmental and crop-specific conditions")

# Load dataset
df = pd.read_csv("cropdata_updated.csv")
df.columns = df.columns.str.strip()

# Display dataset summary with optional preview
with st.expander("📊 Dataset Overview"):
    st.write(df.head())
    st.write("Shape:", df.shape)

# Encode categorical variables
categorical_cols = ['soil_type', 'Seedling Stage', 'crop ID']
df_encoded = df.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Feature matrix and target variable
X = df_encoded[['crop ID', 'soil_type', 'Seedling Stage', 'MOI', 'temp', 'humidity']]
y = df_encoded['result']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting model for better results
model = GradientBoostingClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Sidebar for inputs
st.sidebar.header("🌿 Input Crop Conditions")
crop_options = df['crop ID'].unique().tolist()
soil_options = df['soil_type'].unique().tolist()
stage_options = df['Seedling Stage'].unique().tolist()

selected_crop = st.sidebar.selectbox("Crop Type", crop_options)
selected_soil = st.sidebar.selectbox("Soil Type", soil_options)
selected_stage = st.sidebar.selectbox("Seedling Stage", stage_options)

moi = st.sidebar.slider("Soil Moisture Index (MOI)", int(df['MOI'].min()), int(df['MOI'].max()), int(df['MOI'].mean()))
temp = st.sidebar.slider("Temperature (°C)", int(df['temp'].min()), int(df['temp'].max()), int(df['temp'].mean()))
humidity = st.sidebar.slider("Humidity (%)", float(df['humidity'].min()), float(df['humidity'].max()), float(df['humidity'].mean()))

# Prepare input for prediction
input_data = pd.DataFrame([[
    le_dict['crop ID'].transform([selected_crop])[0],
    le_dict['soil_type'].transform([selected_soil])[0],
    le_dict['Seedling Stage'].transform([selected_stage])[0],
    moi, temp, humidity
]], columns=X.columns)

# Prediction
prediction = model.predict(input_data)[0]

# Custom HTML/CSS for water droplet effect
if prediction == 1:
    st.markdown("""
        <style>
        .droplet {
            width: 10px;
            height: 10px;
            background: #00BFFF;
            border-radius: 50%;
            position: absolute;
            animation: fall 2s infinite;
            opacity: 0.7;
        }

        @keyframes fall {
            0% { top: 0; opacity: 0.7; }
            100% { top: 100vh; opacity: 0; }
        }

        .rain-container {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: 9999;
            pointer-events: none;
        }

        </style>
        <div class="rain-container">
            """ +
            "".join([f'<div class="droplet" style="left:{i}%; animation-delay: {i*0.05}s;"></div>' for i in range(0, 100, 5)]) +
        "</div>",
        unsafe_allow_html=True
    )

# Display result
st.subheader("🔍 Prediction Result")
if prediction == 1:
    st.success("💧 The model recommends watering the crop.")
else:
    st.info("✅ No irrigation needed currently.")

# Show model evaluation
with st.expander("📈 Model Evaluation"):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    st.json(report)
    st.write(f"Model Accuracy: {accuracy:.2f}")

# Feature Importance (optional)
with st.expander("📊 Feature Importance"):
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("Importance")
    ax.set_title("Gradient Boosting Feature Importance")
    st.pyplot(fig)
