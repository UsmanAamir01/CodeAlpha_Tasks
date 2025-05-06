import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Unemployment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("📊 Unemployment Analysis During COVID-19")
st.markdown("""
This application analyzes unemployment rates before, during, and after the COVID-19 pandemic.
Explore the data, visualize trends, and understand the impact of the pandemic on employment.
""")

# Function to load and prepare data
@st.cache_data
def load_data():
    try:
        # Try multiple possible locations for the data file
        possible_paths = [
            "Unemployment_Rate_upto_11_2020.csv",  # Current directory
            "Task5/Unemployment_Rate_upto_11_2020.csv",  # Task5 subdirectory
            "../Task5/Unemployment_Rate_upto_11_2020.csv",  # Parent directory's Task5
            os.path.join(os.path.dirname(__file__), "Unemployment_Rate_upto_11_2020.csv")  # Same directory as script
        ]

        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"Data loaded successfully from {path}")
                return df
            except (FileNotFoundError, IOError):
                continue

        # If we get here, none of the paths worked
        st.error("Data file not found. Please upload a CSV file.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to clean and preprocess data
def preprocess_data(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()

    # Check for and handle missing values
    if data.isnull().sum().sum() > 0:
        st.write("Missing values found in the dataset:")
        st.write(data.isnull().sum())
        data = data.dropna()
        st.write("Missing values have been dropped.")

    # Convert date columns to datetime if they exist
    date_columns = [col for col in data.columns if 'date' in col.lower() or 'month' in col.lower()]
    for col in date_columns:
        try:
            data[col] = pd.to_datetime(data[col])
            st.write(f"Converted {col} to datetime format.")
        except:
            st.write(f"Could not convert {col} to datetime.")

    # Rename columns for consistency if needed
    if ' Date' in data.columns:
        data.rename(columns={' Date': 'Date'}, inplace=True)

    if ' Estimated Unemployment Rate (%)' in data.columns:
        data.rename(columns={' Estimated Unemployment Rate (%)': 'Unemployment_Rate'}, inplace=True)

    if ' Estimated Employed' in data.columns:
        data.rename(columns={' Estimated Employed': 'Employed'}, inplace=True)

    if ' Estimated Labour Participation Rate (%)' in data.columns:
        data.rename(columns={' Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'}, inplace=True)

    if ' Region' in data.columns:
        data.rename(columns={' Region': 'Region'}, inplace=True)

    if ' Area' in data.columns:
        data.rename(columns={' Area': 'Area'}, inplace=True)

    return data

# Function to create visualizations
def create_visualizations(data):
    st.header("Unemployment Analysis Visualizations")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Unemployment Trends", "Regional Analysis", "Urban vs Rural", "COVID-19 Impact"])

    with tab1:
        st.subheader("Unemployment Rate Trends Over Time")

        # Check if Date column exists
        if 'Date' in data.columns and 'Unemployment_Rate' in data.columns:
            # Group by date and calculate mean unemployment rate
            monthly_data = data.groupby('Date')['Unemployment_Rate'].mean().reset_index()

            # Create a line chart using Plotly
            fig = px.line(
                monthly_data,
                x='Date',
                y='Unemployment_Rate',
                title='Monthly Unemployment Rate Trend',
                labels={'Unemployment_Rate': 'Unemployment Rate (%)', 'Date': 'Month'},
                markers=True
            )

            # Add a vertical line for COVID-19 start (March 2020)
            if monthly_data['Date'].min() <= pd.to_datetime('2020-03-01') <= monthly_data['Date'].max():
                fig.add_vline(
                    x=pd.to_datetime('2020-03-01'),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="COVID-19 Start",
                    annotation_position="top right"
                )

            st.plotly_chart(fig, use_container_width=True)

            # Show the data in a table
            st.write("Monthly Unemployment Data:")
            st.dataframe(monthly_data)
        else:
            st.write("Required columns not found in the dataset.")

    with tab2:
        st.subheader("Regional Unemployment Analysis")

        if 'Region' in data.columns and 'Unemployment_Rate' in data.columns:
            # Group by region and calculate statistics
            region_data = data.groupby('Region')['Unemployment_Rate'].agg(['mean', 'min', 'max']).reset_index()
            region_data.columns = ['Region', 'Average Unemployment Rate', 'Minimum Unemployment Rate', 'Maximum Unemployment Rate']

            # Create a bar chart for regional comparison
            fig = px.bar(
                region_data,
                x='Region',
                y='Average Unemployment Rate',
                color='Average Unemployment Rate',
                title='Average Unemployment Rate by Region',
                labels={'Average Unemployment Rate': 'Unemployment Rate (%)'},
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Create a map visualization if possible
            st.write("Regional Unemployment Statistics:")
            st.dataframe(region_data)

            # Time series by region
            st.subheader("Unemployment Trends by Region")

            # Allow user to select regions to display
            regions = sorted(data['Region'].unique())
            selected_regions = st.multiselect("Select Regions to Display", regions, default=regions[:3])

            if selected_regions:
                # Filter data for selected regions
                filtered_data = data[data['Region'].isin(selected_regions)]

                # Group by region and date
                region_time_data = filtered_data.groupby(['Region', 'Date'])['Unemployment_Rate'].mean().reset_index()

                # Create line chart for each region
                fig = px.line(
                    region_time_data,
                    x='Date',
                    y='Unemployment_Rate',
                    color='Region',
                    title='Unemployment Rate Trends by Region',
                    labels={'Unemployment_Rate': 'Unemployment Rate (%)', 'Date': 'Month'}
                )

                # Add COVID-19 marker
                if region_time_data['Date'].min() <= pd.to_datetime('2020-03-01') <= region_time_data['Date'].max():
                    fig.add_vline(
                        x=pd.to_datetime('2020-03-01'),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="COVID-19 Start",
                        annotation_position="top right"
                    )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Required columns not found in the dataset.")

    with tab3:
        st.subheader("Urban vs Rural Unemployment")

        if 'Area' in data.columns and 'Unemployment_Rate' in data.columns:
            # Group by area and calculate statistics
            area_data = data.groupby('Area')['Unemployment_Rate'].agg(['mean', 'min', 'max']).reset_index()
            area_data.columns = ['Area', 'Average Unemployment Rate', 'Minimum Unemployment Rate', 'Maximum Unemployment Rate']

            # Create a bar chart for area comparison
            fig = px.bar(
                area_data,
                x='Area',
                y='Average Unemployment Rate',
                color='Area',
                title='Average Unemployment Rate by Area (Urban vs Rural)',
                labels={'Average Unemployment Rate': 'Unemployment Rate (%)'},
                barmode='group'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Time series by area
            st.subheader("Unemployment Trends by Area")

            # Group by area and date
            area_time_data = data.groupby(['Area', 'Date'])['Unemployment_Rate'].mean().reset_index()

            # Create line chart for each area
            fig = px.line(
                area_time_data,
                x='Date',
                y='Unemployment_Rate',
                color='Area',
                title='Unemployment Rate Trends by Area (Urban vs Rural)',
                labels={'Unemployment_Rate': 'Unemployment Rate (%)', 'Date': 'Month'}
            )

            # Add COVID-19 marker
            if area_time_data['Date'].min() <= pd.to_datetime('2020-03-01') <= area_time_data['Date'].max():
                fig.add_vline(
                    x=pd.to_datetime('2020-03-01'),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="COVID-19 Start",
                    annotation_position="top right"
                )

            st.plotly_chart(fig, use_container_width=True)

            # Compare urban and rural by region
            if 'Region' in data.columns:
                st.subheader("Urban vs Rural Unemployment by Region")

                # Group by region and area
                region_area_data = data.groupby(['Region', 'Area'])['Unemployment_Rate'].mean().reset_index()

                # Create a grouped bar chart
                fig = px.bar(
                    region_area_data,
                    x='Region',
                    y='Unemployment_Rate',
                    color='Area',
                    title='Urban vs Rural Unemployment Rate by Region',
                    labels={'Unemployment_Rate': 'Unemployment Rate (%)'},
                    barmode='group'
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Required columns not found in the dataset.")

    with tab4:
        st.subheader("COVID-19 Impact Analysis")

        if 'Date' in data.columns and 'Unemployment_Rate' in data.columns:
            # Define pre-COVID and COVID periods
            try:
                # Convert Date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                    data['Date'] = pd.to_datetime(data['Date'])

                # Define the COVID start date (March 2020)
                covid_start = pd.to_datetime('2020-03-01')

                # Create period column
                data['Period'] = data['Date'].apply(lambda x: 'Pre-COVID' if x < covid_start else 'During-COVID')

                # Group by period
                period_data = data.groupby('Period')['Unemployment_Rate'].mean().reset_index()

                # Create a bar chart comparing pre and during COVID
                fig = px.bar(
                    period_data,
                    x='Period',
                    y='Unemployment_Rate',
                    color='Period',
                    title='Average Unemployment Rate: Pre-COVID vs During-COVID',
                    labels={'Unemployment_Rate': 'Unemployment Rate (%)'},
                    barmode='group',
                    color_discrete_map={'Pre-COVID': 'blue', 'During-COVID': 'red'}
                )

                st.plotly_chart(fig, use_container_width=True)

                # Monthly trend with COVID highlight
                monthly_data = data.groupby(['Date', 'Period'])['Unemployment_Rate'].mean().reset_index()

                fig = px.line(
                    monthly_data,
                    x='Date',
                    y='Unemployment_Rate',
                    color='Period',
                    title='Unemployment Rate Trend with COVID-19 Period Highlighted',
                    labels={'Unemployment_Rate': 'Unemployment Rate (%)', 'Date': 'Month'},
                    color_discrete_map={'Pre-COVID': 'blue', 'During-COVID': 'red'}
                )

                # Add COVID-19 marker
                fig.add_vline(
                    x=covid_start,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="COVID-19 Start",
                    annotation_position="top right"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Impact by region
                if 'Region' in data.columns:
                    st.subheader("COVID-19 Impact by Region")

                    # Group by region and period
                    region_period_data = data.groupby(['Region', 'Period'])['Unemployment_Rate'].mean().reset_index()

                    # Create a grouped bar chart
                    fig = px.bar(
                        region_period_data,
                        x='Region',
                        y='Unemployment_Rate',
                        color='Period',
                        title='COVID-19 Impact on Unemployment by Region',
                        labels={'Unemployment_Rate': 'Unemployment Rate (%)'},
                        barmode='group',
                        color_discrete_map={'Pre-COVID': 'blue', 'During-COVID': 'red'}
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Impact by area (Urban vs Rural)
                if 'Area' in data.columns:
                    st.subheader("COVID-19 Impact: Urban vs Rural")

                    # Group by area and period
                    area_period_data = data.groupby(['Area', 'Period'])['Unemployment_Rate'].mean().reset_index()

                    # Create a grouped bar chart
                    fig = px.bar(
                        area_period_data,
                        x='Area',
                        y='Unemployment_Rate',
                        color='Period',
                        title='COVID-19 Impact on Urban vs Rural Unemployment',
                        labels={'Unemployment_Rate': 'Unemployment Rate (%)'},
                        barmode='group',
                        color_discrete_map={'Pre-COVID': 'blue', 'During-COVID': 'red'}
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Calculate percentage increase
                pre_covid = period_data[period_data['Period'] == 'Pre-COVID']['Unemployment_Rate'].values[0]
                during_covid = period_data[period_data['Period'] == 'During-COVID']['Unemployment_Rate'].values[0]
                percent_increase = ((during_covid - pre_covid) / pre_covid) * 100

                # Display metrics
                st.subheader("COVID-19 Impact Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Pre-COVID Unemployment Rate", f"{pre_covid:.2f}%")
                col2.metric("During-COVID Unemployment Rate", f"{during_covid:.2f}%")
                col3.metric("Percentage Increase", f"{percent_increase:.2f}%", f"{percent_increase:.2f}%")

            except Exception as e:
                st.error(f"Error in COVID-19 analysis: {str(e)}")
        else:
            st.write("Required columns not found in the dataset.")

# Function to show data statistics and summary
def show_data_summary(data):
    st.header("Data Summary")

    # Display basic information
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Records", f"{data.shape[0]}")
    col2.metric("Number of Features", f"{data.shape[1]}")
    col3.metric("Time Period", f"{data['Date'].min().strftime('%b %Y')} - {data['Date'].max().strftime('%b %Y')}")

    # Display the first few rows
    st.subheader("Sample Data")
    st.dataframe(data.head())

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(data.describe())

    # Display correlation matrix if there are numerical columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numerical_cols) > 1:
        st.subheader("Correlation Matrix")
        corr = data[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# Main function
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Analysis", "About"])

    # Load data
    df = load_data()

    if df is not None:
        # Preprocess data
        data = preprocess_data(df)

        if page == "Data Overview":
            show_data_summary(data)

        elif page == "Visualizations":
            create_visualizations(data)

        elif page == "Analysis":
            st.header("Unemployment Analysis")

            # Allow user to select specific analysis
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Unemployment Rate Distribution", "Time Series Decomposition", "Regional Comparison", "Custom Analysis"]
            )

            if analysis_type == "Unemployment Rate Distribution":
                st.subheader("Unemployment Rate Distribution")

                # Create histogram
                fig = px.histogram(
                    data,
                    x='Unemployment_Rate',
                    nbins=20,
                    title='Distribution of Unemployment Rates',
                    labels={'Unemployment_Rate': 'Unemployment Rate (%)'},
                    color_discrete_sequence=['blue']
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary statistics
                st.write("Summary Statistics for Unemployment Rate:")
                st.dataframe(data['Unemployment_Rate'].describe())

            elif analysis_type == "Time Series Decomposition":
                st.subheader("Time Series Decomposition")
                st.write("This analysis decomposes the unemployment time series into trend, seasonal, and residual components.")

                try:
                    # Group by date to get monthly data
                    monthly_data = data.groupby('Date')['Unemployment_Rate'].mean().reset_index()
                    monthly_data.set_index('Date', inplace=True)

                    # Check if we have enough data for decomposition
                    data_length = len(monthly_data)
                    st.info(f"Available data points: {data_length} months")

                    if data_length >= 24:  # Ideal case: 2 complete yearly cycles
                        period = 12
                        st.success("Using standard yearly seasonality (period=12)")
                    elif data_length >= 14:  # At least 1 year + 2 months
                        # Use a smaller period for decomposition
                        period = min(6, data_length // 4)
                        st.warning(f"Not enough data for full yearly decomposition. Using shorter period ({period}) for analysis.")
                    else:
                        period = None
                        st.error("Not enough data for meaningful time series decomposition. Need at least 14 months of data.")

                    if period:
                        try:
                            from statsmodels.tsa.seasonal import seasonal_decompose

                            # Perform decomposition with adjusted period if necessary
                            decomposition = seasonal_decompose(monthly_data['Unemployment_Rate'], model='additive', period=period)

                            # Create figure with subplots
                            fig = make_subplots(
                                rows=4,
                                cols=1,
                                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                                vertical_spacing=0.1
                            )

                            # Add traces
                            fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
                            fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                            fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)

                            # Update layout
                            fig.update_layout(height=800, title_text=f"Time Series Decomposition of Unemployment Rate (Period={period})")
                            st.plotly_chart(fig, use_container_width=True)

                            # Add explanation
                            if period != 12:
                                st.info("""
                                **Note on decomposition:** The standard approach uses a period of 12 for monthly data,
                                but this requires at least 24 months of data. Since we have fewer observations,
                                a shorter period was used. This may affect the seasonal component interpretation.
                                """)
                        except Exception as e:
                            st.error(f"Error in time series decomposition: {str(e)}")
                            st.info("""
                            **Possible solutions:**
                            - Try a different period value
                            - Use more data if available
                            - Consider alternative analysis methods for short time series
                            """)
                    else:
                        # Show alternative analysis when decomposition isn't possible
                        st.subheader("Alternative Analysis: Month-to-Month Changes")
                        monthly_data['Change'] = monthly_data['Unemployment_Rate'].diff()

                        fig = px.line(
                            monthly_data.dropna(),
                            x=monthly_data.dropna().index,
                            y='Change',
                            title='Month-to-Month Changes in Unemployment Rate',
                            labels={'Change': 'Change in Unemployment Rate (%)', 'index': 'Month'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in time series decomposition: {str(e)}")

            elif analysis_type == "Regional Comparison":
                st.subheader("Regional Unemployment Comparison")

                if 'Region' in data.columns:
                    # Allow user to select regions to compare
                    regions = sorted(data['Region'].unique())
                    selected_regions = st.multiselect("Select Regions to Compare", regions, default=regions[:3])

                    if selected_regions:
                        # Filter data for selected regions
                        filtered_data = data[data['Region'].isin(selected_regions)]

                        # Group by region
                        region_stats = filtered_data.groupby('Region')['Unemployment_Rate'].agg(['mean', 'min', 'max', 'std']).reset_index()
                        region_stats.columns = ['Region', 'Average', 'Minimum', 'Maximum', 'Std Dev']

                        # Display statistics
                        st.write("Unemployment Statistics by Region:")
                        st.dataframe(region_stats)

                        # Create visualization
                        fig = px.box(
                            filtered_data,
                            x='Region',
                            y='Unemployment_Rate',
                            color='Region',
                            title='Unemployment Rate Distribution by Region',
                            labels={'Unemployment_Rate': 'Unemployment Rate (%)'}
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # If we have date data, show time trends
                        if 'Date' in filtered_data.columns:
                            # Group by region and date
                            region_time = filtered_data.groupby(['Region', 'Date'])['Unemployment_Rate'].mean().reset_index()

                            # Create line chart
                            fig = px.line(
                                region_time,
                                x='Date',
                                y='Unemployment_Rate',
                                color='Region',
                                title='Unemployment Rate Trends by Region',
                                labels={'Unemployment_Rate': 'Unemployment Rate (%)', 'Date': 'Month'}
                            )

                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Please select at least one region to compare.")
                else:
                    st.write("Region column not found in the dataset.")

            elif analysis_type == "Custom Analysis":
                st.subheader("Custom Analysis")

                # Allow user to select variables for analysis
                available_columns = [col for col in data.columns if col not in ['Date', 'Period']]

                x_axis = st.selectbox("Select X-axis Variable", available_columns)
                y_axis = st.selectbox("Select Y-axis Variable", available_columns, index=min(1, len(available_columns)-1))

                # Select chart type
                chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Violin Plot"])

                # Optional color variable
                color_var = st.selectbox("Select Color Variable (Optional)", ["None"] + available_columns)
                color_column = None if color_var == "None" else color_var

                # Create the selected chart
                if chart_type == "Scatter Plot":
                    fig = px.scatter(
                        data,
                        x=x_axis,
                        y=y_axis,
                        color=color_column,
                        title=f'{y_axis} vs {x_axis}',
                        labels={x_axis: x_axis, y_axis: y_axis}
                    )

                elif chart_type == "Line Chart":
                    # For line chart, we need to sort by x-axis
                    plot_data = data.sort_values(by=x_axis)
                    fig = px.line(
                        plot_data,
                        x=x_axis,
                        y=y_axis,
                        color=color_column,
                        title=f'{y_axis} vs {x_axis}',
                        labels={x_axis: x_axis, y_axis: y_axis}
                    )

                elif chart_type == "Bar Chart":
                    fig = px.bar(
                        data,
                        x=x_axis,
                        y=y_axis,
                        color=color_column,
                        title=f'{y_axis} by {x_axis}',
                        labels={x_axis: x_axis, y_axis: y_axis}
                    )

                elif chart_type == "Box Plot":
                    fig = px.box(
                        data,
                        x=x_axis,
                        y=y_axis,
                        color=color_column,
                        title=f'{y_axis} Distribution by {x_axis}',
                        labels={x_axis: x_axis, y_axis: y_axis}
                    )

                elif chart_type == "Violin Plot":
                    fig = px.violin(
                        data,
                        x=x_axis,
                        y=y_axis,
                        color=color_column,
                        title=f'{y_axis} Distribution by {x_axis}',
                        labels={x_axis: x_axis, y_axis: y_axis},
                        box=True
                    )

                st.plotly_chart(fig, use_container_width=True)

        elif page == "About":
            st.header("About This Project")
            st.write("""
            ## Unemployment Analysis During COVID-19

            This project analyzes unemployment data to understand the impact of the COVID-19 pandemic on employment rates.

            ### Data Sources
            The data used in this analysis comes from unemployment statistics collected before and during the COVID-19 pandemic.

            ### Methodology
            - Data is cleaned and preprocessed to handle missing values and format issues
            - Exploratory data analysis is performed to understand patterns and trends
            - Visualizations are created to illustrate the impact of COVID-19 on unemployment
            - Regional and urban/rural comparisons are made to identify differential impacts

            ### Key Findings
            - COVID-19 caused a significant spike in unemployment rates
            - Different regions experienced varying levels of impact
            - Urban and rural areas showed different patterns of unemployment
            - The recovery has been uneven across different demographics

            ### Tools Used
            - Python for data analysis
            - Pandas for data manipulation
            - Plotly and Matplotlib for visualization
            - Streamlit for creating this interactive dashboard
            """)
    else:
        # If data is not loaded, provide option to upload
        st.header("Upload Unemployment Data")
        uploaded_file = st.file_uploader("Upload CSV file with unemployment data", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success("Data uploaded successfully!")
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
