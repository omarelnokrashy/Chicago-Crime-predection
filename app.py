import streamlit as st
import requests
import joblib
from streamlit_lottie import st_lottie # type: ignore
import requests
from PIL import Image  # Import PIL for image handling
import os
from Models import Model
import json
import visualization
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

#load data frame 
df=pd.read_csv("Data\\Crime Prediction in Chicago_Dataset.csv")

# load classifiers 
RandomForest = joblib.load("Models\\best_rf_classifier.pkl")
KNN = joblib.load('Models\\best_knn_classifier.pkl')
Logistic = joblib.load('Models\\best_lr_classifier.pkl')
XGB = joblib.load('Models\\best_xgb_classifier.pkl')
SVM = joblib.load('Models\\svm_classifier.pkl')
DecisionTree = joblib.load('Models\\best_dt_classifier.pkl')

models ={
    "SVM":SVM,
    "KNN":KNN,
    "Logistic":Logistic,
    "Random Forest":RandomForest,
    "XGBoost":XGB,
    "Decision Tree":DecisionTree
}

# Load the scaler
scaler = joblib.load('Models\\scaler.pkl')

def get_unique(column_name):
    return df[column_name].unique().tolist()

def create_and_save_encoder(data_column, encoder_filename):
    encoder = LabelEncoder()
    encoder.fit(data_column)
    
    # Save the encoder to a file
    joblib.dump(encoder, encoder_filename)
    print(f"Encoder saved as {encoder_filename}")

    return encoder

def preprocess(Block, IUCR, Primary_Type, Description, Location_Description, 
                                 Beat, FBI_Code, Latitude, Longitude, hour, day_of_week, month):
    """
    Preprocess the input features using newly created LabelEncoders.
    
    Parameters:
        Block, IUCR, Primary_Type, Description, Location_Description, FBI_Code: Categorical data.
        Beat, Latitude, Longitude, hour, day_of_week, month: Numerical data.
        
    Returns:
        np.ndarray: Preprocessed feature vector ready for prediction.
    """
    
    # Load or create new encoders for each categorical feature
    block_encoder = create_and_save_encoder(df['Block'], 'Models\\Block_encoder.pkl')  # Example: df['Block']
    iucr_encoder = create_and_save_encoder(df['IUCR'], 'Models\\IUCR_encoder.pkl')  # Example: df['IUCR']
    primary_type_encoder = create_and_save_encoder(df['Primary Type'], 'Models\\Primary_Type_encoder.pkl')  # Example: df['Primary Type']
    description_encoder = create_and_save_encoder(df['Description'], 'Models\\Description_encoder.pkl')  # Example: df['Description']
    location_desc_encoder = create_and_save_encoder(df['Location Description'], 'Models\\Location_Description_encoder.pkl')  # Example: df['Location Description']
    fbi_code_encoder = create_and_save_encoder(df['FBI Code'], 'Models\\FBI_Code_encoder.pkl')  # Example: df['FBI Code']
    
    # Encode categorical features
    Block = block_encoder.transform([Block])[0]
    IUCR = iucr_encoder.transform([IUCR])[0]
    Primary_Type = primary_type_encoder.transform([Primary_Type])[0]
    Description = description_encoder.transform([Description])[0]
    Location_Description = location_desc_encoder.transform([Location_Description])[0]
    FBI_Code = fbi_code_encoder.transform([FBI_Code])[0]
    
    # Combine all features into a list
    features = [
        Block, IUCR, Primary_Type, Description, Location_Description,
        Beat, FBI_Code, Latitude, Longitude, hour, day_of_week, month
    ]
    
    # Load the scaler (assuming it's already trained)
    scaler = joblib.load('Models\\scaler.pkl')
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    return features_scaled

def predict_classification(model,features):
    """
    Predicts the class label and probabilities for the given features.
    
    Parameters:
        features (np.ndarray): Preprocessed feature vector for prediction.
        
    Returns:
        tuple: Predicted class label and corresponding probabilities.
    """
    # Predict the class
    predicted = model.predict(features)[0]
    
    # Predict probabilities for each class
    probabilities = model.predict_proba(features)[0]  # [0] to get probabilities for the first input
    
    return predicted, probabilities

st.set_page_config(
    page_title="Crime Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)    


def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Lottie file: {e}")
        return None


with st.sidebar:
    choose = option_menu(None, ["Home", "Visualization","Classification Insights","Important Insights"],
                         icons=['house', 'kanban',
                                'book', 'person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

from streamlit_lottie import st_lottie
import streamlit as st
import requests

def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Lottie file: {e}")
        return None



if choose == 'Home':
    # Path to your Lottie JSON file
    lottie_filepath = r"Assets\Detective.json"
    # Load Lottie animation
    lottie_animation = load_lottie_file(lottie_filepath)
    if lottie_animation:
        # Display the Lottie animation
        st_lottie(lottie_animation, height=400, width=400, loop=True, quality="high")
    else:
        st.error("Failed to load Lottie animation.")

    st.write('# Chicago Crime Predictor')

    st.write('---')
    st.subheader('Enter the details to predict')
    
    # input fields 
    block = st.selectbox("Block",[None]+sorted(get_unique('Block')))
    iucr = st.selectbox("IUCR",[None]+sorted(get_unique('IUCR')))
    primary_type = st.selectbox("Primary Type",[None]+sorted(get_unique('Primary Type')))
    description = st.selectbox("Description",[None]+sorted(get_unique('Description')))
    location_description = st.selectbox("Location Description",[None]+get_unique('Location Description'))
    Beat =st.selectbox("Beat",[None]+sorted(get_unique('Beat')))
    fbi_code = st.selectbox("FBI Code",[None]+sorted(get_unique('FBI Code')))
    latitude = st.number_input("Latitude", format="%1f")
    longitude = st.number_input("Longitude", format="%1f")
    hour = st.number_input("Hour", min_value=0, max_value=23, step=1)
    day_of_week = st.selectbox("Day of the Week",[None]+["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    month = st.selectbox("Month",[None]+list(range(1, 13)))
    model=st.selectbox("Model",[None]+["SVM","KNN","Logistic","XGBoost","Random Forest","Decision Tree"])

    day_of_week_mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }
   
    if st.button("Predict"):
        SelectedModel=models[model]
        features=preprocess(block,iucr,primary_type,description,location_description,Beat,fbi_code,latitude,longitude,hour,day_of_week_mapping[day_of_week],month)
        prediction,propability=predict_classification(SelectedModel,features)
        st.write("The Prediction :",prediction)
        st.write("The Probability :",propability)
        if prediction == 0:
            st.success("Predicted No crime")
            st.write("Aman")
            st.balloons()    
        elif prediction == 1:
            st.warning("Predicted Crime")
            st.write("2msek Harami")
         
if choose == 'Visualization':
    column = st.selectbox("Select Column for Bar Chart",df.columns)
    visualization_type = st.selectbox("Select Visualization Type",["Bar Chart","Donate Chart"])
    if st.button("Visualize"):
        if visualization_type == "Bar Chart":
            visualization.bar_chart(df, column)
        elif visualization_type == "Donate Chart":
            visualization.DonateChart(df, column)



if choose == 'Classification Insights':
    st.subheader('Classification Insights')
    st.write('---')
    model_name=st.selectbox("Model",[None]+["SVM","KNN","Logistic","XGBoost","Random Forest","Decision Tree"])
    if st.button("Get Insights"):
        obj=Model(models[model_name])
        obj.display_model_information()

if choose == 'Important Insights':
    st.header("Important Insights")
    st.write('---')
    
    insights = {
        'Biased Data':"Assets\\before upsamplin.png",  # Replace with the actual file path or generated chart
        'After UpSampling':"Assets\\after upsampling.png",
        'Heat Map Correlation':"Assets\\Heat map.png",
        'Feature Importance':"Assets\\feature importance.png",
        'Model Accuracies':"Assets\\Accuracy of Classification models.png",
        'ROC':"Assets\\ROC.png"
    }


    # Create tabs for each insight
    tabs = st.tabs(list(insights.keys()))

    # Display each insight in its respective tab
    for tab, (title, image_path) in zip(tabs, insights.items()):
        with tab:
            st.image(image_path, caption=title, use_container_width=True)
