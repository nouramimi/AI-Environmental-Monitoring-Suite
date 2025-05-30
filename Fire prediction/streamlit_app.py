import streamlit as st
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import month_plot
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series, plot_lags
from statsmodels.graphics.tsaplots import plot_acf, month_plot, seasonal_plot, plot_predict
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sktime.split import temporal_train_test_split
from statsmodels.tsa.stattools import adfuller

pio.templates.default = 'plotly_white'


def visualize_geo(df):
    geo_plot = px.scatter_geo(
        df.sort_values(by=["acq_date"]),
        lat='latitude',
        lon='longitude',
        animation_frame='acq_date',
        title='Scatter geo plot of fire incidents',
        projection='natural earth'
    )
    geo_plot.update_layout(height=600)
    return geo_plot

def process_dataframe(df):
    # Convert 'acq_date' to datetime
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    # Set 'acq_date' as the index
    df.set_index('acq_date', inplace=True)
    
    # Extract month and day from the index
    df['month'] = df.index.month
    df['day'] = df.index.day
    
    # Drop specified columns
    df.drop(['version', 'satellite', 'type', 'instrument', 'version'], axis=1, inplace=True)
    
    return df

def do_feature_engineering(df):
    confidence_mapping = {'l': 0, 'n': 1, 'h': 2} # Define mapping for 'confidence' feature
    df['confidence_encoded'] = df['confidence'].map(confidence_mapping)
    daynight_mapping = {'D': 1, 'N': 0} # Define mapping for 'daynight' feature
    df['daynight_encoded'] = df['daynight'].map(daynight_mapping) # Encode 'daynight' feature
    # Drop unnecessary or unrelated columns
    columns_to_drop = ['confidence', 'daynight', 'month', 'day']
    df = df.drop(columns=columns_to_drop, axis = 1) 
    return df

def decompose_data(data):
    result = adfuller(data['confidence_encoded'])
    if result[1] > 0.05:  # If p-value > 0.05, apply differencing
        data['confidence_encoded_diff'] = data['confidence_encoded'].diff().dropna()
    return data


def resample_data(df, freq='D'):
    resampled_df = df.resample(freq).mean()
    resampled_df = resampled_df.bfill()
    return resampled_df


def optimize_arima_and_forecast(resampled_df, target_column, seasonal=False, m=1, test_size=0.2):
    """
    Optimize ARIMA hyperparameters and forecast the target column.

    Parameters:
        resampled_df (pd.DataFrame): Time series data with a DateTime index.
        target_column (str): Column to forecast ('confidence_encoded' or 'confidence_encoded_diff').
        test_size (float): Proportion of the dataset to include in the test split.
        seasonal (bool): Whether to consider seasonality in ARIMA.
        m (int): Seasonal periodicity (used if seasonal=True).
    
    Returns:
        dict: Dictionary containing model, evaluation metrics, and forecasts.
    """

    # Split into train and test sets
    y_train, y_test = temporal_train_test_split(resampled_df, test_size=test_size)
    train, test = y_train[target_column], y_test[target_column]

    # Auto ARIMA to find optimal (p, d, q) parameters
    model = auto_arima(
        train,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=True
    )

    # Fit the ARIMA model
    model.fit(train)

    # Forecast the test set
    n_periods = len(test)  # Convert test_size to an integer
    forecasts = model.predict(n_periods=n_periods)

    # Evaluate the model
    mae = mean_absolute_error(test, forecasts)
    mse = mean_squared_error(test, forecasts)
    rmse = np.sqrt(mse)

    # Output results
    return {
        "model": model,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "forecasts": forecasts,
        "actuals": test
    }
# Streamlit app
st.title("Wild Fire Forecasting")

# File upload
uploaded_file = st.file_uploader("Upload your time series dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)


    st.subheader("Geo Plot of Fire Recordings")
    geo_chart = visualize_geo(data)
    st.plotly_chart(geo_chart, use_container_width=True)

    data = process_dataframe(data)

    data = do_feature_engineering(data)


    data = decompose_data(data) 

    data = resample_data(data)

    st.write("Dataset Preview after preprocessing:")
    st.write(data.head())


    # Select target column
    target_column = "confidence_encoded"

    # Forecasting parameters
    #test_size = st.slider("Test size (proportion of data)", 0.1, 0.5, 0.2, step=0.05)
    test_size = 0.2
    # seasonal = st.checkbox("Seasonal data?", value=False)
    # m = st.number_input("Seasonal periodicity (m)", min_value=1, value=1, step=1)

    # Run ARIMA optimization
    if st.button("Run ARIMA Forecast"):
        with st.spinner("Optimizing ARIMA model..."):
            result = optimize_arima_and_forecast(data, target_column, test_size)

        st.success("Model training and forecasting complete!")

        # Display model details and metrics
        st.subheader("Best ARIMA Model:")
        st.text(result["model"])
        st.metric("Mean Absolute Error (MAE)", round(result["mae"], 4))
        st.metric("Mean Squared Error (MSE)", round(result["mse"], 4))
        st.metric("Root Mean Squared Error (RMSE)", round(result["rmse"], 4))

        # Plot actual vs forecasted values using plot_series
        st.subheader("Actual vs Forecasted Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_series(
            result["actuals"], 
            result["forecasts"], 
            labels=["Actual", "Forecast"], 
            ax=ax,
            title="Actual vs Forecasted Values"
        )
        st.pyplot(fig)
    

        # Diagnostics plot
        st.subheader("Model Diagnostics")
        diagnostics_fig = result["model"].plot_diagnostics(figsize=(10, 8))
        st.pyplot(diagnostics_fig)