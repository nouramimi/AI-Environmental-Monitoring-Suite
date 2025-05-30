import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load data function
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

# Cache the data
@st.cache_data
def cache_data(uploaded_file):
    return load_data(uploaded_file)

# Data Exploration
def explore_data(df):
    st.write("Exploring Data...")
    st.write("Data Preview:", df.head())
    st.write("Data Info:", df.info())
    st.write("Descriptive Statistics:", df.describe())

# Data Cleaning
def clean_data(df):
    # Vérification des types de données
    print(df.dtypes)
    
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Remplir les valeurs manquantes avec la moyenne des colonnes numériques
    df_filled = df.copy()  # Créer une copie pour ne pas modifier df original
    df_filled[numeric_df.columns] = numeric_df.fillna(numeric_df.mean())
    
    # Nettoyage des valeurs textuelles (si nécessaire)
    df_filled = df_filled.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    return df_filled


# Visualization options
def visualize_data(df):
    st.write("Choose a visualization:")
    options = [
        "Spread of AQI Categories",
        "Mean AQI by Country",
        "Heatmap of Correlations",
        "Histogram of AQI and PM2.5",
        "Regression: AQI vs PM2.5",
        "Top 10 Most/Least Polluted Countries",
        "AQI Value Distribution by Country",
        "Choropleth Map of AQI by Country",  # Nouvelle option
    ]
    choice = st.selectbox("Select a visualization", options)

    # Spread of AQI Categories
    if choice == "Spread of AQI Categories":
        st.write("Spread of AQI Categories")
        fig, ax = plt.subplots()
        df['AQI Category'].value_counts().plot.bar(ax=ax)
        st.pyplot(fig)

    # Mean AQI by Country
    elif choice == "Mean AQI by Country":
        st.write("Mean AQI by Country")
        mean_aqi = df.groupby("Country")["AQI Value"].mean().reset_index()
        fig = px.bar(mean_aqi, x="Country", y="AQI Value", title="Mean AQI Value by Country")
        st.plotly_chart(fig)

    # Heatmap of Correlations
    elif choice == "Heatmap of Correlations":
        st.write("Correlation Heatmap")
        corr_matrix = df[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    # Histogram of AQI and PM2.5
    elif choice == "Histogram of AQI and PM2.5":
        st.write("Histogram of AQI and PM2.5 Values")
        fig, ax = plt.subplots()
        ax.hist(df['AQI Value'], bins=30, color='blue', edgecolor='black', label='AQI Values')
        ax.hist(df['PM2.5 AQI Value'], bins=30, color='orange', alpha=0.6, edgecolor='black', label='PM2.5 AQI Values')
        ax.set_title("Histogram of AQI Values")
        ax.set_xlabel("AQI Values")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    # Regression: AQI vs PM2.5
    elif choice == "Regression: AQI vs PM2.5":
        st.write("Regression: AQI Value vs PM2.5 AQI Value")
        x = df['AQI Value'].values.reshape(-1, 1)
        y = df['PM2.5 AQI Value'].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['AQI Value'], y=df['PM2.5 AQI Value'], ax=ax)
        ax.plot(x, model.predict(x), color='red')
        ax.set_title("AQI vs PM2.5 with Regression Line")
        st.pyplot(fig)

    # Top 10 Most/Least Polluted Countries
    elif choice == "Top 10 Most/Least Polluted Countries":
        avg_aqi_by_country = df.groupby('Country')['AQI Value'].mean().sort_values()
        top_10_least = avg_aqi_by_country.head(10)
        top_10_most = avg_aqi_by_country.tail(10)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        top_10_least.plot(kind='bar', ax=axes[0], title='Top 10 Least Polluted Countries')
        top_10_most.plot(kind='bar', ax=axes[1], title='Top 10 Most Polluted Countries')
        pd.concat([top_10_least, top_10_most]).plot(kind='bar', ax=axes[2], title='Combined: Least and Most Polluted')
        st.pyplot(fig)

    # AQI Value Distribution by Country
    elif choice == "AQI Value Distribution by Country":
        st.write("Distribution of AQI Values by Country")
        fig, ax = plt.subplots(figsize=(10, 14))
        sns.barplot(x='AQI Value', y='Country', data=df, ax=ax)
        ax.set_title("AQI Value Distribution by Country")
        ax.set_xlabel("AQI Value")
        ax.set_ylabel("Country")
        st.pyplot(fig)

    # Choropleth Map of AQI by Country (Nouvelle option)
    elif choice == "Choropleth Map of AQI by Country":
        st.write("Choropleth Map of AQI by Country")
        # Créez le DataFrame nécessaire pour la carte
        pollute_country = df.groupby("Country")["AQI Value"].mean().reset_index()
        # Créez la carte choroplèthe
        fig = px.choropleth(pollute_country, locations="Country", locationmode="country names",
                            color="AQI Value", hover_name="Country",
                            color_continuous_scale="Turbo",
                            title="Average Air Quality Index by Country")
        # Affichez la carte
        st.plotly_chart(fig)

# Encode categorical data
def encode_categorical_data(df):
    label_encoder = LabelEncoder()
    # Encode categorical columns (e.g., 'Country', 'City')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

# Model Selection and Evaluation
def model_selection(df):
    st.header('Model Selection')
    model_option = st.selectbox("Choose a model", ['Linear Regression', 'Decision Tree', 'Random Forest'])
    target_column = st.selectbox('Choose the target variable', df.columns)
    features = df.drop(columns=[target_column])

    # Encoding categorical columns before model training
    features_encoded = encode_categorical_data(features)

    X = features_encoded
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_option == 'Linear Regression':
        model = LinearRegression()
    elif model_option == 'Decision Tree':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
    elif model_option == 'Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Model Results")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    st.write(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")

    st.write("Predicted vs Actual Values")
    st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

    return model

# Prediction for a specific country or city
def make_prediction(df, model):
    st.header('Make Prediction')
    location = st.text_input("Enter a country or city")
    if location:
        st.write(f"Predicting AQI for {location}...")
        location_data = df[df['Country'] == location] if 'Country' in df.columns else df[df['City'] == location]
        if not location_data.empty:
            features = location_data.drop(columns=['AQI Value'])  # Use the relevant columns for prediction
            features_encoded = encode_categorical_data(features)  # Ensure the input features are encoded
            predicted_aqi = model.predict(features_encoded)
            st.write(f"Predicted AQI for {location}: {predicted_aqi[0]}")

            # Provide a recommendation based on predicted AQI value
            st.write("Recommendation based on AQI:")
            if predicted_aqi[0] <= 50:
                st.write("Air quality is good. No immediate action required.")
            elif 51 <= predicted_aqi[0] <= 100:
                st.write("Air quality is moderate. Sensitive individuals may experience mild health effects.")
            elif 101 <= predicted_aqi[0] <= 150:
                st.write("Air quality is unhealthy for sensitive individuals. Members of sensitive groups may experience health effects.")
            elif 151 <= predicted_aqi[0] <= 200:
                st.write("Air quality is unhealthy. Everyone may begin to experience health effects.")
            elif 201 <= predicted_aqi[0] <= 300:
                st.write("Air quality is very unhealthy. Health alert: everyone may experience more serious health effects.")
            else:
                st.write("Air quality is hazardous. Health warning of emergency conditions. The entire population is likely to be affected.")
        else:
            st.write("Location not found in the dataset.")

# Main Streamlit app structure
st.title('Air Pollution Data Analysis and Prediction')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = cache_data(uploaded_file)
    if df is not None:
        task_option = st.radio("Select task", ['Data Exploration', 'Data Cleaning', 'Data Visualization', 'Model Training & Prediction'])

        if task_option == 'Data Exploration':
            explore_data(df)
        elif task_option == 'Data Cleaning':
            df_filled = clean_data(df)
            st.write("Cleaned Data (after filling missing values):", df_filled.head())
        elif task_option == 'Data Visualization':
            visualize_data(df)
        elif task_option == 'Model Training & Prediction':
            model = model_selection(df)
            make_prediction(df, model)
else:
    st.write("Please upload a CSV file to begin.")
