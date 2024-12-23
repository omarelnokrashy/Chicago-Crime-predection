import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Function for Bar Chart
def bar_chart(df, column):
    plt.figure(figsize=(10, 6))
    df[column].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Bar Chart of {column}', fontsize=16)
    plt.ylabel('Count', fontsize=14)
    plt.xlabel(column, fontsize=14)
    st.pyplot(plt)

# Function for Donate Chart
def DonateChart(df, column):
    counts = df[column].value_counts()
    labels = [f'{label} ({count})' for label, count in zip(counts.index, counts.values)]

    # Normalize color indices to distribute colors evenly
    color_indices = [i / len(counts) for i in range(len(counts))]
    colors = plt.cm.viridis(color_indices)  # Generate distinct colors

    plt.figure(figsize=(12, 8))  # Adjust figure size
    wedges, texts = plt.pie(
        counts,
        #labels=labels,
        # autopct='%1.1f%%',
        startangle=90,
        colors=colors,  
        wedgeprops={'width': 0.4}
    )

    # Style the percentage text
    # for autotext in autotexts:
    #     autotext.set_bbox(dict(facecolor='white', edgecolor='none'))

    # Add legend outside the pie chart
    plt.legend(
        wedges,
        labels,
        title=f"{column} Distribution",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=11
    )
    plt.title(f'Distribution of {column}', fontsize=16)
    st.pyplot(plt)

# Function for Line Chart
def line_chart(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        plt.figure(figsize=(10, 6))
        plt.plot(df[column].dropna(), marker='o', color='orange')
        plt.title(f'Line Chart of {column}', fontsize=16)
        plt.ylabel(column, fontsize=14)
        plt.xlabel('Index', fontsize=14)
        st.pyplot(plt)
    else:
        st.warning(f"The column '{column}' is not numeric or contains invalid values. Please select a numeric column.")

# Function for Grouped Bar Chart
def grouped_bar_chart(df, group_column, value_column):
    if pd.api.types.is_numeric_dtype(df[value_column]):
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df, 
            x=group_column, 
            y=value_column, 
            hue=group_column, 
            palette='viridis'
        )
        plt.title(f'Grouped Bar Chart: {value_column} by {group_column}', fontsize=16)
        plt.xlabel(group_column, fontsize=14)
        plt.ylabel(value_column, fontsize=14)
        plt.legend(title=group_column)
        st.pyplot(plt)
    else:
        st.warning(f"The column '{value_column}' must be numeric to create a grouped bar chart.")

