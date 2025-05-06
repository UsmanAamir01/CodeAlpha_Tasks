import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set plotly as not available to use matplotlib instead
PLOTLY_AVAILABLE = False

# =====================================================
# Car Price Prediction with Machine Learning
# =====================================================
st.title("Car Price Prediction")
st.markdown("""
This app predicts the selling price of a car based on various features like the year,
present price, kilometers driven, fuel type, and more.
""")

# ---------------------------
# Data Loading using st.cache_data
# ---------------------------
@st.cache_data
def load_data():
    # Get the directory of the current script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the data file
    data_path = os.path.join(script_dir, "car data.csv")
    # Load the data
    data = pd.read_csv(data_path)
    return data

data = load_data()

# Feature engineering
# Calculate car age from year
current_year = datetime.now().year
data['Car_Age'] = current_year - data['Year']

# ---------------------------
# Data Preview and Column Selection
# ---------------------------
st.subheader("Dataset Preview")
st.write("Columns in the dataset:", data.columns.tolist())
st.dataframe(data.head())

# ---------------------------
# Data Exploration and Visualization
# ---------------------------
st.subheader("Data Exploration")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Correlation Analysis", "Feature Relationships", "Year Analysis"])

with tab1:
    st.write("### Car Price Distribution")

    # Histogram of car prices
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='Selling_Price', kde=True, ax=ax)
    ax.set_title('Distribution of Car Selling Prices')
    ax.set_xlabel('Selling Price (in lakhs)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Box plot of prices by fuel type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='Fuel_Type', y='Selling_Price', ax=ax)
    ax.set_title('Selling Price by Fuel Type')
    ax.set_xlabel('Fuel Type')
    ax.set_ylabel('Selling Price (in lakhs)')
    st.pyplot(fig)

with tab2:
    st.write("### Correlation Analysis")

    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Compute correlation matrix
    corr_matrix = numeric_data.corr()

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features')
    st.pyplot(fig)

    # Highlight strongest correlations with Selling_Price
    if 'Selling_Price' in corr_matrix.columns:
        price_corr = corr_matrix['Selling_Price'].sort_values(ascending=False)
        st.write("### Features Correlation with Selling Price")

        # Create a bar chart of correlations
        fig, ax = plt.subplots(figsize=(10, 6))
        price_corr.drop('Selling_Price').plot(kind='bar', ax=ax)
        ax.set_title('Feature Correlation with Selling Price')
        ax.set_ylabel('Correlation Coefficient')
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        st.pyplot(fig)

with tab3:
    st.write("### Feature Relationships")

    # Scatter plot of Present Price vs Selling Price
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Present_Price', y='Selling_Price', hue='Fuel_Type', ax=ax)
    ax.set_title('Present Price vs Selling Price by Fuel Type')
    ax.set_xlabel('Present Price (in lakhs)')
    ax.set_ylabel('Selling Price (in lakhs)')
    st.pyplot(fig)

    # Scatter plot of Car Age vs Selling Price
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Car_Age', y='Selling_Price', hue='Transmission', ax=ax)
    ax.set_title('Car Age vs Selling Price by Transmission Type')
    ax.set_xlabel('Car Age (years)')
    ax.set_ylabel('Selling Price (in lakhs)')
    st.pyplot(fig)

    # Kilometers Driven vs Selling Price
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='Driven_kms', y='Selling_Price', hue='Owner', ax=ax)
    ax.set_title('Kilometers Driven vs Selling Price by Owner')
    ax.set_xlabel('Kilometers Driven')
    ax.set_ylabel('Selling Price (in lakhs)')
    st.pyplot(fig)

with tab4:
    st.write("### Year Analysis")

    # Group by Year and calculate mean selling price
    year_data = data.groupby('Year')['Selling_Price'].mean().reset_index()

    # Plot year vs average selling price
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=year_data, x='Year', y='Selling_Price', marker='o', ax=ax)
    ax.set_title('Average Selling Price by Year of Manufacture')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Selling Price (in lakhs)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Count of cars by year
    year_count = data['Year'].value_counts().sort_index()

    # Plot count of cars by year
    fig, ax = plt.subplots(figsize=(12, 6))
    year_count.plot(kind='bar', ax=ax)
    ax.set_title('Number of Cars by Year of Manufacture')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# ---------------------------
# Define Column Names for Features and Target
# ---------------------------
# Standardize column names
column_mapping = {
    'selling_price': 'Selling_Price',
    'present_price': 'Present_Price',
    'car_name': 'Car_Name',
    'year': 'Year',
    'driven_kms': 'Driven_kms',
    'fuel_type': 'Fuel_Type',
    'selling_type': 'Selling_type',
    'transmission': 'Transmission',
    'owner': 'Owner'
}

# Rename columns if needed
for old_col, new_col in column_mapping.items():
    if old_col in data.columns and new_col not in data.columns:
        data.rename(columns={old_col: new_col}, inplace=True)

# Check for required columns
required_columns = ['Selling_Price', 'Present_Price', 'Year', 'Driven_kms', 'Fuel_Type', 'Transmission', 'Owner']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# ---------------------------
# Data Preparation
# ---------------------------
# Ensure numeric columns are properly formatted
numeric_columns = ['Selling_Price', 'Present_Price', 'Year', 'Driven_kms', 'Owner']
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Check for missing values
if data[numeric_columns].isnull().any().any():
    st.warning("Some numeric columns contain missing values. These rows will be dropped.")
    data = data.dropna(subset=numeric_columns)

# Feature engineering was already done after loading the data

# Define features and target
# Target is the selling price
target_col = 'Selling_Price'
y = data[target_col]

# Features are everything except the target and Car_Name
feature_cols = [col for col in data.columns if col not in [target_col, 'Car_Name']]
X = data[feature_cols]

# Display selected features
st.write("**Target Variable:** Selling_Price")
st.write("**Selected Features:**", feature_cols)

# ---------------------------
# Split Data for Training and Testing
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Preprocessing Pipeline
# ---------------------------
# Identify categorical and numerical columns
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Create preprocessing steps
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

# ---------------------------
# Model Training
# ---------------------------
# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# ---------------------------
# Model Evaluation
# ---------------------------
y_pred = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

# Create metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("MSE", f"{mse:.2f}")
with col2:
    st.metric("RMSE", f"{rmse:.2f}")
with col3:
    st.metric("MAE", f"{mae:.2f}")
with col4:
    st.metric("R² Score", f"{r2:.2f}")

# Create tabs for different evaluation visualizations
eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Prediction Analysis", "Residual Analysis", "Feature Importance"])

with eval_tab1:
    st.write("### Actual vs Predicted Prices")

    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Scatter plot with perfect prediction line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Actual vs Predicted Car Prices')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Interactive scatter plot with Plotly if available
    if PLOTLY_AVAILABLE:
        import plotly.express as px
        import plotly.graph_objects as go
        fig = px.scatter(results_df, x='Actual', y='Predicted',
                        labels={'Actual': 'Actual Price', 'Predicted': 'Predicted Price'},
                        title='Interactive Actual vs Predicted Car Prices')

        # Add perfect prediction line
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))

        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

with eval_tab2:
    st.write("### Residual Analysis")

    # Calculate residuals
    residuals = y_test - y_pred

    # Add residuals to results DataFrame
    results_df['Residuals'] = residuals

    # Histogram of residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title('Distribution of Residuals')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Frequency')
    ax.axvline(x=0, color='r', linestyle='--')
    st.pyplot(fig)

    # Residuals vs Predicted values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('Residuals vs Predicted Values')
    ax.set_xlabel('Predicted Price')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # QQ plot for residuals
    from scipy import stats
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, plot=ax)
    ax.set_title('Q-Q Plot of Residuals')
    st.pyplot(fig)

with eval_tab3:
    st.write("### Feature Importance")

    # Get feature importance from the model (RandomForest has feature_importances_)
    feature_names = X.columns

    # Get feature importances
    importances = model[-1].feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': [importances[i] for i in indices]
    })

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

    # Display feature importance table
    st.write("### Feature Importance Table")
    st.dataframe(importance_df)

# ---------------------------
# Interactive Prediction Section
# ---------------------------
st.subheader("Predict Car Price")
st.write("Enter the details of the car to predict its selling price.")

# Create input form for all features
input_data = {}

# Year input
input_data['Year'] = st.number_input(
    "Year of Manufacture:",
    min_value=int(data['Year'].min()),
    max_value=int(data['Year'].max()),
    value=int(data['Year'].median())
)

# Calculate car age
input_data['Car_Age'] = current_year - input_data['Year']

# Present Price input
input_data['Present_Price'] = st.number_input(
    "Present Price (in lakhs):",
    min_value=float(data['Present_Price'].min()),
    max_value=float(data['Present_Price'].max()),
    value=float(data['Present_Price'].median())
)

# Kilometers Driven input
input_data['Driven_kms'] = st.number_input(
    "Kilometers Driven:",
    min_value=int(data['Driven_kms'].min()),
    max_value=int(data['Driven_kms'].max()),
    value=int(data['Driven_kms'].median())
)

# Fuel Type selection
fuel_types = data['Fuel_Type'].unique().tolist()
input_data['Fuel_Type'] = st.selectbox("Fuel Type:", fuel_types)

# Selling Type selection
if 'Selling_type' in data.columns:
    selling_types = data['Selling_type'].unique().tolist()
    input_data['Selling_type'] = st.selectbox("Selling Type:", selling_types)
else:
    input_data['Selling_type'] = "Dealer"  # Default value

# Transmission selection
transmission_types = data['Transmission'].unique().tolist()
input_data['Transmission'] = st.selectbox("Transmission:", transmission_types)

# Owner input
if 'Owner' in data.columns:
    owner_options = sorted(data['Owner'].unique().tolist())
    input_data['Owner'] = st.selectbox("Owner (Previous owners count):", owner_options)
else:
    input_data['Owner'] = 0  # Default value

# Create a DataFrame with the input data
input_df = pd.DataFrame([input_data])

# Ensure all columns match the training data
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Default value for missing columns

# Reorder columns to match training data
input_df = input_df[X.columns]

# Make prediction
if st.button("Predict Price"):
    predicted_price = model.predict(input_df)[0]
    st.success(f"### Predicted Selling Price: ₹{predicted_price:.2f} lakhs")

    # Create a gauge chart for the predicted price if plotly is available
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_price,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Price (in lakhs)"},
            gauge={
                'axis': {'range': [0, max(20, predicted_price * 1.5)]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 10], 'color': "gray"},
                    {'range': [10, 15], 'color': "lightblue"},
                    {'range': [15, 20], 'color': "royalblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_price
                }
            }
        ))

        fig.update_layout(width=600, height=400)
        st.plotly_chart(fig)
    else:
        # Alternative visualization using matplotlib
        fig, ax = plt.subplots(figsize=(10, 2))

        # Create a simple horizontal bar to represent the price
        max_price = max(20, predicted_price * 1.5)
        ax.barh(['Price'], [predicted_price], color='royalblue')
        ax.barh(['Price'], [max_price], color='lightgray', alpha=0.3)

        # Add the price value as text
        ax.text(predicted_price, 0, f'₹{predicted_price:.2f} lakhs',
                va='center', ha='left', fontweight='bold')

        # Set the limits and remove y-axis
        ax.set_xlim(0, max_price)
        ax.set_yticks([])
        ax.set_xlabel('Price (in lakhs)')
        ax.set_title('Predicted Car Price')

        st.pyplot(fig)

    # Show similar cars from the dataset
    st.subheader("Similar Cars in the Dataset")

    # Find similar cars based on features
    # Calculate Euclidean distance for numeric features
    numeric_features = ['Year', 'Present_Price', 'Driven_kms', 'Car_Age']

    # Normalize the features for distance calculation
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Select only numeric columns that exist in both input_df and data
    common_numeric_features = [col for col in numeric_features if col in input_df.columns and col in data.columns]

    if common_numeric_features:
        # Scale the features
        data_scaled = scaler.fit_transform(data[common_numeric_features])
        input_scaled = scaler.transform(input_df[common_numeric_features])

        # Calculate distances
        distances = np.sqrt(((data_scaled - input_scaled) ** 2).sum(axis=1))

        # Get indices of 5 most similar cars
        similar_indices = distances.argsort()[:5]

        # Display similar cars
        similar_cars = data.iloc[similar_indices].copy()

        # Add a column for similarity score (inverse of distance)
        similar_cars['Similarity'] = 1 / (1 + distances[similar_indices])

        # Display the similar cars with their details
        st.write("These cars have similar characteristics to your input:")
        st.dataframe(similar_cars[['Car_Name', 'Year', 'Selling_Price', 'Present_Price',
                                  'Driven_kms', 'Fuel_Type', 'Transmission', 'Similarity']])

        # Create a comparison chart
        comparison_data = pd.DataFrame({
            'Your Car': [predicted_price],
            'Similar Car 1': [similar_cars.iloc[0]['Selling_Price']],
            'Similar Car 2': [similar_cars.iloc[1]['Selling_Price']],
            'Similar Car 3': [similar_cars.iloc[2]['Selling_Price']],
            'Similar Car 4': [similar_cars.iloc[3]['Selling_Price']],
            'Similar Car 5': [similar_cars.iloc[4]['Selling_Price']]
        })

        # Plot the comparison
        st.write("### Price Comparison with Similar Cars")
        st.bar_chart(comparison_data.T)
    else:
        st.write("Could not find similar cars due to missing features.")

# ---------------------------
# Future Enhancements
# ---------------------------
st.markdown("""
**Next Steps / Enhancements:**

- **Model Tuning:** Experiment with hyperparameter tuning to optimize the Random Forest model or try other regression models like XGBoost, Gradient Boosting, or Neural Networks.
- **Feature Engineering:** Create more advanced features like interaction terms or polynomial features to capture non-linear relationships.
- **Data Visualization:** Add more visualizations to explore relationships between features and car prices.
- **Cross-Validation:** Implement k-fold cross-validation for more robust model evaluation.
- **Feature Importance:** Display feature importance to understand which factors most influence car prices.
- **Price Trends:** Add analysis of how car prices change over time or with different features.
- **Deployment:** Package the application for production deployment with proper error handling and logging.
""")
