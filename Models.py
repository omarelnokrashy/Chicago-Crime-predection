import streamlit as st
import requests
import joblib
import seaborn as sns
from streamlit_lottie import st_lottie # type: ignore
import requests
from PIL import Image  # Import PIL for image handling
import os
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
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, model):
        self.model = model
        # Load data frame
        self.X = pd.read_csv("X.csv")
        self.y_df = pd.read_csv("y.csv")        

        # Load the scaler
        self.scaler = joblib.load('scaler.pkl')

        # Flatten the target variable (if it's not already)
        self.y = self.y_df.to_numpy().ravel()

        # Scale the features
        self.X_scaled = self.scaler.transform(self.X)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y)

        # Get predictions using the model
        self.y_pred = self.model.predict(self.X_test)

    def predict(self,X_test):
        predicted = self.model.predict(X_test)[0]
        return predicted
    
    def predict_prob(self):
        prob = self.model.predict_proba(self.features)
        return prob
    
    def predict_log_prob(self):
        log_prob = self.model.predict_log_proba(self.features)
        return log_prob
    
    def classification_report(self):
        report = classification_report(self.y_test,self.y_pred)
        return report
    
    def confusion_matrix(self):
        matrix = confusion_matrix(self.y_test, self.y_pred)
        return matrix
    
    def accuracy(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        return acc
    
    def feature_importance(self):
        importance = self.model.feature_importances_
        return importance
    
    def plot_feature_importance(self):
        plt.figure(figsize=(10, 6))
        plt.barh(self.X.columns, self.model.feature_importances_)
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Feature Importance', fontsize=16)
        st.pyplot(plt)
    
    def plot_confusion_matrix(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            confusion_matrix(self.y_test, self.y_pred),
            annot=True,
            fmt='d',
            cmap='viridis'
        )
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        st.pyplot(plt)

    def plot_classification_report(self):
        report = classification_report(self.y_test, self.y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.table(df)

    def display_model_information (self):
        st.write("Model Accuracy Score : ",self.accuracy())
        st.write("Model Classification Report : ")
        self.plot_classification_report()
        self.plot_confusion_matrix()