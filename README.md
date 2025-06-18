# Smart Irrigation Predictor

## Overview
The Smart Irrigation Predictor is a web application built using Streamlit and machine learning techniques to predict whether to water crops based on environmental and crop-specific conditions. The application utilizes a Gradient Boosting model to analyze various factors and provide recommendations for irrigation.

## Features
- **User Input**: Users can input crop conditions such as soil type, seedling stage, soil moisture index (MOI), temperature, and humidity.
- **Prediction**: The model predicts whether irrigation is needed based on the input conditions.
- **Visualization**: The application provides a visual representation of the prediction results and feature importance.
- **Water Droplet Effect**: A fun visual effect of falling water droplets is displayed when the model recommends watering.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

## Installation
To run this application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Soham-Kamathi/smart-irrigation-predictor.git
   cd smart-irrigation-predictor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the application in your web browser.
2. Use the sidebar to input the crop conditions.
3. Click on the "Predict" button to see the recommendation.
4. The application will display whether to water the crop and show a visual effect if watering is recommended.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.



## Acknowledgments
- Thanks to the contributors and the community for their support and feedback.
