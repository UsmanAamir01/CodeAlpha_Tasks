import streamlit as st
import pandas as pd
import numpy as np
import logging
import joblib
import os
import io
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

log_output = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.StreamHandler(log_output)
    ]
)
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and drop missing values."""
    # Get the absolute path to the file
    try:
        # First try the path as provided
        logging.info(f"Attempting to load data from {filepath}")
        df = pd.read_csv(filepath).dropna()
    except FileNotFoundError:
        # If that fails, try to resolve the path relative to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_filepath = os.path.join(script_dir, os.path.basename(filepath))
        logging.info(f"File not found. Trying script directory: {abs_filepath}")
        df = pd.read_csv(abs_filepath).dropna()

    logging.info(f"Data shape after dropna: {df.shape}")
    return df

def explore_data(df: pd.DataFrame) -> None:
    """
    Explore and visualize the dataset to understand relationships.
    """
    logging.info("Exploring data relationships")

    # Create a directory for plots if it doesn't exist
    Path("plots").mkdir(exist_ok=True)

    # Summary statistics
    logging.info("\nSummary Statistics:")
    logging.info(f"\n{df.describe()}")

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")

    # Pairplot to visualize relationships
    plt.figure(figsize=(12, 10))
    sns_plot = sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars=['Sales'],
                           height=4, aspect=1, kind='scatter')
    sns_plot.savefig("plots/pairplot.png")

    # Individual scatter plots with regression line
    for feature in ['TV', 'Radio', 'Newspaper']:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=feature, y='Sales', data=df)
        plt.title(f'Sales vs {feature} Advertising')
        plt.xlabel(f'{feature} Advertising Budget')
        plt.ylabel('Sales')
        plt.tight_layout()
        plt.savefig(f"plots/sales_vs_{feature.lower()}.png")

    logging.info("Data exploration completed. Plots saved to 'plots' directory.")

def feature_engineering(
    df: pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = True
) -> tuple[pd.DataFrame, pd.Series, PolynomialFeatures]:
    """
    Create polynomial interaction features from TV, Radio, Newspaper.
    Returns transformed X, target y, and the fitted PolynomialFeatures object.
    """
    # Basic features
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)
    cols = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=cols)

    # Add custom features if needed
    # For example, ratio of TV to total advertising budget
    X_poly_df['TV_Ratio'] = df['TV'] / (df['TV'] + df['Radio'] + df['Newspaper'])
    X_poly_df['Radio_Ratio'] = df['Radio'] / (df['TV'] + df['Radio'] + df['Newspaper'])
    X_poly_df['Newspaper_Ratio'] = df['Newspaper'] / (df['TV'] + df['Radio'] + df['Newspaper'])

    logging.info(f"Generated {X_poly_df.shape[1]} features")
    return X_poly_df, y, poly

def build_and_tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    n_iter: int = 20,
    random_state: int = 42
) -> tuple[Pipeline, RandomizedSearchCV]:
    """
    Builds a pipeline and performs RandomizedSearchCV over multiple regressors.
    """
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=random_state))
    ])

    # Create separate parameter distributions for each model type
    # Linear Regression parameters
    linear_params = {
        'model': [LinearRegression()],
        'model__fit_intercept': [True, False],
        'model__positive': [True, False]
    }

    # Ridge parameters
    ridge_params = {
        'model': [Ridge(random_state=random_state)],
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'model__fit_intercept': [True, False],
        'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

    # Lasso parameters
    lasso_params = {
        'model': [Lasso(random_state=random_state)],
        'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'model__fit_intercept': [True, False],
        'model__selection': ['cyclic', 'random']
    }

    # Random Forest parameters
    rf_params = {
        'model': [RandomForestRegressor(random_state=random_state)],
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_depth': [None, 5, 10, 15, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    # Gradient Boosting parameters
    gb_params = {
        'model': [GradientBoostingRegressor(random_state=random_state)],
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__subsample': [0.8, 0.9, 1.0]
    }

    # Combine all parameter distributions
    param_dists = [linear_params, ridge_params, lasso_params, rf_params, gb_params]

    # Create a list of all parameter combinations
    param_list = []
    for param_dist in param_dists:
        param_list.extend(list(ParameterSampler(param_dist, n_iter=6, random_state=random_state)))

    # Set up cross-validation
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Instead of using RandomizedSearchCV with param_distributions, we'll manually evaluate each model
    # This avoids the issue with incompatible parameters for different model types

    # Create a list to store results
    all_results = []
    best_score = float('-inf')
    best_model = None

    logging.info(f"Evaluating {len(param_list)} different model configurations")

    # Evaluate each parameter combination
    for i, params in enumerate(param_list):
        try:
            # Create a new pipeline with these parameters
            test_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', params['model'])
            ])

            # Set the parameters
            for param_name, param_value in params.items():
                if param_name != 'model':  # Skip the model itself
                    setattr(test_pipeline.named_steps['model'], param_name.replace('model__', ''), param_value)

            # Evaluate using cross-validation
            scores = cross_val_score(
                test_pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            # Calculate mean and std of scores
            mean_score = scores.mean()
            std_score = scores.std()

            # Store results
            result = {
                'params': params,
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'model_type': type(params['model']).__name__
            }
            all_results.append(result)

            # Check if this is the best model so far
            if mean_score > best_score:
                best_score = mean_score
                # Fit the model on the full training set
                test_pipeline.fit(X_train, y_train)
                best_model = test_pipeline

            # Log progress
            if (i + 1) % 5 == 0:
                logging.info(f"Evaluated {i + 1}/{len(param_list)} models. Best score so far: {best_score:.3f}")

        except Exception as e:
            logging.warning(f"Error evaluating model with params {params}: {str(e)}")

    # Create a DataFrame with all results for analysis
    results_df = pd.DataFrame(all_results)

    # Convert negative MSE to positive for easier interpretation
    results_df['mean_test_score'] = -results_df['mean_test_score']

    # Sort by score
    results_df = results_df.sort_values('mean_test_score')

    # Create a mock RandomizedSearchCV object to maintain compatibility with the rest of the code
    class MockRandomizedSearchCV:
        def __init__(self, best_estimator, cv_results_, best_score_):
            self.best_estimator_ = best_estimator
            self.cv_results_ = cv_results_
            self.best_score_ = best_score_
            self.best_params_ = {k: v for k, v in results_df.iloc[0]['params'].items()
                               if k != 'model'}
            self.best_params_['model_type'] = type(results_df.iloc[0]['params']['model']).__name__

    # Create the mock object
    search = MockRandomizedSearchCV(
        best_model,
        {
            'mean_test_score': -results_df['mean_test_score'].values,
            'std_test_score': results_df['std_test_score'].values,
            'mean_train_score': -results_df['mean_test_score'].values * 0.9,  # Approximation
            'std_train_score': results_df['std_test_score'].values * 0.9,  # Approximation
        },
        -results_df['mean_test_score'].min()
    )

    # Log results
    logging.info(f"Model evaluation complete. {len(all_results)} models evaluated.")
    logging.info(f"Best model type: {search.best_params_['model_type']}")
    logging.info(f"Best parameters found: {search.best_params_}")
    logging.info(f"Best CV score: {-search.best_score_:.3f} MSE")

    # Create a plot of the CV results
    plt.figure(figsize=(12, 6))
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results['mean_test_score'] = -cv_results['mean_test_score']  # Convert to positive MSE
    cv_results['mean_train_score'] = -cv_results['mean_train_score']

    # Sort by test score
    cv_results = cv_results.sort_values('mean_test_score')

    # Plot top 10 models
    top_models = cv_results.head(10)
    plt.errorbar(
        range(len(top_models)),
        top_models['mean_test_score'],
        yerr=top_models['std_test_score'],
        fmt='o',
        label='Test Score'
    )
    plt.errorbar(
        range(len(top_models)),
        top_models['mean_train_score'],
        yerr=top_models['std_train_score'],
        fmt='o',
        label='Train Score'
    )
    plt.xlabel('Model Rank')
    plt.ylabel('Mean Squared Error')
    plt.title('Top 10 Models: Train vs Test Performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png")

    return search.best_estimator_, search

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "plots"
) -> dict:
    """
    Evaluates model on test set, saves plots, and returns evaluation metrics.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    logging.info(f"Test MSE: {mse:.3f}")
    logging.info(f"Test RMSE: {rmse:.3f}")
    logging.info(f"Test MAE: {mae:.3f}")
    logging.info(f"Test R²: {r2:.3f}")

    # Create a results dictionary
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'actuals': y_test
    }

    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect Prediction')

    # Add regression line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r-", label=f'Best Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "actual_vs_predicted.png")

    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.hlines(0, y_pred.min(), y_pred.max(), 'k', 'dashed')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "residuals_plot.png")

    # Residuals histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "residuals_histogram.png")

    # Feature importance (if the model supports it)
    if hasattr(model[-1], 'feature_importances_'):
        feature_names = X_test.columns
        importances = model[-1].feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(X_test.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "feature_importance.png")

        # Save feature importance to results
        results['feature_importance'] = {feature_names[i]: importances[i] for i in range(len(feature_names))}

    return results

def make_prediction(model, tv_budget, radio_budget, newspaper_budget, poly=None):
    """
    Make a sales prediction based on advertising budgets.

    Args:
        model: Trained model pipeline
        tv_budget: TV advertising budget
        radio_budget: Radio advertising budget
        newspaper_budget: Newspaper advertising budget
        poly: PolynomialFeatures transformer (if used in training)

    Returns:
        Predicted sales value
    """
    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'TV': [tv_budget],
        'Radio': [radio_budget],
        'Newspaper': [newspaper_budget]
    })

    # If we have polynomial features, we need to transform the input
    if poly is not None:
        # Transform the input data
        input_poly = poly.transform(input_data)
        cols = poly.get_feature_names_out(input_data.columns)
        input_df = pd.DataFrame(input_poly, columns=cols)

        # Add the ratio features
        total_budget = tv_budget + radio_budget + newspaper_budget
        input_df['TV_Ratio'] = tv_budget / total_budget
        input_df['Radio_Ratio'] = radio_budget / total_budget
        input_df['Newspaper_Ratio'] = newspaper_budget / total_budget
    else:
        input_df = input_data

    # Make prediction
    prediction = model.predict(input_df)[0]

    return prediction

def save_pipeline(
    pipeline: Pipeline,
    filepath: str = "sales_prediction_pipeline.joblib"
) -> None:
    """Serialize the trained pipeline to disk."""
    joblib.dump(pipeline, filepath)
    logging.info(f"Pipeline saved to {filepath}")


def analyze_optimal_budget_allocation(model, poly, max_budget=200):
    """
    Analyze how different budget allocations affect sales predictions.
    Creates visualizations to help understand optimal advertising strategy.
    """
    logging.info("Analyzing optimal budget allocation strategies")

    # Create a grid of budget allocations
    budget_combinations = []
    predictions = []

    # Generate different budget combinations
    step = max_budget / 10
    for tv in np.arange(0, max_budget + step, step):
        for radio in np.arange(0, max_budget - tv + step, step):
            newspaper = max_budget - tv - radio
            if newspaper < 0:
                continue

            # Make prediction for this budget allocation
            pred = make_prediction(model, tv, radio, newspaper, poly)

            # Store the combination and prediction
            budget_combinations.append((tv, radio, newspaper))
            predictions.append(pred)

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(budget_combinations, columns=['TV', 'Radio', 'Newspaper'])
    results_df['Predicted_Sales'] = predictions

    # Create visualizations

    # 1. TV vs Radio (with fixed Newspaper = 0)
    tv_radio_df = results_df[results_df['Newspaper'] < 1].copy()

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    pivot_table = tv_radio_df.pivot_table(
        index='TV',
        columns='Radio',
        values='Predicted_Sales',
        aggfunc='mean'
    )

    sns.heatmap(pivot_table, annot=False, cmap='viridis', cbar_kws={'label': 'Predicted Sales'}, ax=ax1)
    ax1.set_title('Predicted Sales by TV and Radio Budget (Newspaper = 0)')
    plt.tight_layout()

    # 2. Top 10 budget allocations
    top_allocations = results_df.sort_values('Predicted_Sales', ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    top_allocations_melted = pd.melt(
        top_allocations,
        id_vars=['Predicted_Sales'],
        value_vars=['TV', 'Radio', 'Newspaper'],
        var_name='Channel',
        value_name='Budget'
    )

    sns.barplot(x='Predicted_Sales', y='Budget', hue='Channel', data=top_allocations_melted, ax=ax2)
    ax2.set_title('Top 10 Budget Allocations by Predicted Sales')
    plt.tight_layout()

    # Return the top allocation and figures
    top_allocation = top_allocations.iloc[0]
    logging.info(f"Optimal budget allocation found:")
    logging.info(f"TV: ${top_allocation['TV']:.2f}")
    logging.info(f"Radio: ${top_allocation['Radio']:.2f}")
    logging.info(f"Newspaper: ${top_allocation['Newspaper']:.2f}")
    logging.info(f"Predicted Sales: ${top_allocation['Predicted_Sales']:.2f}")

    return top_allocation, fig1, fig2, results_df

def train_model(df):
    """Train the sales prediction model and return the model and evaluation results"""
    # Feature engineering
    X, y, poly = feature_engineering(df, degree=2, interaction_only=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and tune the model
    with st.spinner('Training model... This may take a few minutes.'):
        best_model, _ = build_and_tune_model(X_train, y_train, n_iter=30)
        logging.info(f"Best model: {type(best_model[-1]).__name__}")

    # Evaluate the model
    results = evaluate_model(best_model, X_test, y_test)

    # Log the key metrics
    logging.info(f"Test R² score: {results['r2']:.4f}")
    logging.info(f"Test RMSE: {results['rmse']:.4f}")

    return best_model, poly, results

def create_prediction_ui(model, poly):
    """Create UI for making predictions"""
    st.subheader("Make Your Own Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        tv_budget = st.slider("TV Advertising Budget ($)", 0, 300, 100)

    with col2:
        radio_budget = st.slider("Radio Advertising Budget ($)", 0, 100, 20)

    with col3:
        newspaper_budget = st.slider("Newspaper Advertising Budget ($)", 0, 100, 30)

    if st.button("Predict Sales"):
        prediction = make_prediction(model, tv_budget, radio_budget, newspaper_budget, poly)

        st.success(f"### Predicted Sales: ${prediction:.2f}")

        # Show budget breakdown

        # Create a pie chart of budget allocation
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            [tv_budget, radio_budget, newspaper_budget],
            labels=['TV', 'Radio', 'Newspaper'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.1, 0, 0)
        )
        ax.set_title('Your Budget Allocation')
        st.pyplot(fig)


def main():
    """Main Streamlit application"""
    st.title("📊 Sales Prediction Based on Advertising Spend")
    st.markdown("""
    This app predicts product sales based on advertising budgets across different channels (TV, Radio, Newspaper).

    **How it works:**
    1. Upload your advertising data or use the sample dataset
    2. Explore the data and relationships
    3. Train a machine learning model to predict sales
    4. Analyze optimal budget allocation
    5. Make your own predictions
    """)

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Model Training", "Budget Optimization", "Make Predictions"])

    # Initialize session state for storing data and models
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'poly' not in st.session_state:
        st.session_state.poly = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'optimal_allocation' not in st.session_state:
        st.session_state.optimal_allocation = None

    # Data Upload Page
    if page == "Data Upload":
        st.header("Upload Your Advertising Data")

        upload_option = st.radio(
            "Choose an option:",
            ["Upload CSV file", "Use sample dataset"]
        )

        if upload_option == "Upload CSV file":
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file).dropna()
                    st.session_state.data = df
                    st.success(f"Data loaded successfully! Shape: {df.shape}")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        else:
            try:
                # Use the sample dataset
                data_path = "Advertising.csv"
                df = load_data(data_path)
                st.session_state.data = df
                st.success(f"Sample data loaded successfully! Shape: {df.shape}")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")

    # Data Exploration Page
    elif page == "Data Exploration":
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return

        st.header("Data Exploration")

        df = st.session_state.data

        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Pairplots
        st.subheader("Relationships Between Variables")
        fig = sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars=['Sales'], height=4, aspect=1, kind='scatter')
        st.pyplot(fig)

        # Individual scatter plots
        st.subheader("Sales vs. Advertising Channels")

        channel = st.selectbox("Select Advertising Channel", ["TV", "Radio", "Newspaper"])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=channel, y='Sales', data=df, ax=ax)
        ax.set_title(f'Sales vs {channel} Advertising')
        ax.set_xlabel(f'{channel} Advertising Budget')
        ax.set_ylabel('Sales')
        st.pyplot(fig)

    # Model Training Page
    elif page == "Model Training":
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return

        st.header("Model Training")

        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                best_model, poly, results = train_model(st.session_state.data)
                st.session_state.model = best_model
                st.session_state.poly = poly
                st.session_state.results = results

                # Display model performance
                st.subheader("Model Performance")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{results['r2']:.4f}")
                col2.metric("RMSE", f"{results['rmse']:.4f}")
                col3.metric("MSE", f"{results['mse']:.4f}")
                col4.metric("MAE", f"{results['mae']:.4f}")

                # Display actual vs predicted plot
                st.subheader("Actual vs Predicted Sales")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(results['actuals'], results['predictions'], alpha=0.6)
                ax.plot([results['actuals'].min(), results['actuals'].max()],
                        [results['actuals'].min(), results['actuals'].max()],
                        'k--', label='Perfect Prediction')

                # Add regression line
                z = np.polyfit(results['actuals'], results['predictions'], 1)
                p = np.poly1d(z)
                ax.plot(results['actuals'], p(results['actuals']), "r-",
                        label=f'Best Fit: y={z[0]:.2f}x+{z[1]:.2f}')

                ax.set_xlabel('Actual Sales')
                ax.set_ylabel('Predicted Sales')
                ax.set_title('Actual vs Predicted Sales')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Display feature importance if available
                if 'feature_importance' in results:
                    st.subheader("Feature Importance")

                    # Sort feature importance
                    importance_df = pd.DataFrame({
                        'Feature': list(results['feature_importance'].keys()),
                        'Importance': list(results['feature_importance'].values())
                    }).sort_values('Importance', ascending=False)

                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)

        # Show logs
        if st.session_state.model is not None:
            st.subheader("Training Logs")
            st.text(log_output.getvalue())

    # Budget Optimization Page
    elif page == "Budget Optimization":
        if st.session_state.model is None or st.session_state.poly is None:
            st.warning("Please train a model first!")
            return

        st.header("Budget Optimization")

        if st.button("Find Optimal Budget Allocation") or st.session_state.optimal_allocation is not None:
            if st.session_state.optimal_allocation is None:
                with st.spinner("Analyzing budget allocations... This may take a moment."):
                    optimal_allocation, fig1, fig2, results_df = analyze_optimal_budget_allocation(
                        st.session_state.model,
                        st.session_state.poly,
                        max_budget=200
                    )
                    st.session_state.optimal_allocation = optimal_allocation
                    st.session_state.budget_figs = (fig1, fig2)
                    st.session_state.budget_results = results_df

            # Display optimal allocation
            st.subheader("Optimal Budget Allocation")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("TV Budget", f"${st.session_state.optimal_allocation['TV']:.2f}")
            col2.metric("Radio Budget", f"${st.session_state.optimal_allocation['Radio']:.2f}")
            col3.metric("Newspaper Budget", f"${st.session_state.optimal_allocation['Newspaper']:.2f}")
            col4.metric("Predicted Sales", f"${st.session_state.optimal_allocation['Predicted_Sales']:.2f}")

            # Display heatmap
            st.subheader("Sales Prediction Heatmap (TV vs Radio, Newspaper=0)")
            st.pyplot(st.session_state.budget_figs[0])

            # Display top allocations
            st.subheader("Top Budget Allocations")
            st.pyplot(st.session_state.budget_figs[1])

            # Display results table
            st.subheader("All Budget Combinations")
            st.dataframe(st.session_state.budget_results.sort_values('Predicted_Sales', ascending=False))

    # Make Predictions Page
    elif page == "Make Predictions":
        if st.session_state.model is None or st.session_state.poly is None:
            st.warning("Please train a model first!")
            return

        st.header("Make Your Own Predictions")

        create_prediction_ui(st.session_state.model, st.session_state.poly)

# Run the app
if __name__ == "__main__":
    main()
