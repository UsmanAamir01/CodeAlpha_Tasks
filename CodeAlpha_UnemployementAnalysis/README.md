# Unemployment Analysis During COVID-19

This project analyzes unemployment data to understand the impact of the COVID-19 pandemic on employment rates across different regions and demographics.

## Project Overview

The COVID-19 pandemic caused unprecedented disruption to labor markets worldwide. This analysis examines unemployment trends before and during the pandemic to quantify its impact and identify patterns across different regions and demographics.

## Features

- **Data Visualization**: Interactive charts and graphs showing unemployment trends
- **Regional Analysis**: Comparison of unemployment rates across different regions
- **Urban vs Rural Analysis**: Examination of differences between urban and rural unemployment
- **COVID-19 Impact Assessment**: Quantitative analysis of how the pandemic affected unemployment
- **Custom Analysis Tools**: User-driven analysis options for exploring the data

## Data

The dataset includes monthly unemployment data with the following information:
- Date: Month and year of the data point
- Region: Geographic region (North, South, East, West)
- Area: Urban or Rural classification
- Estimated Unemployment Rate (%): Percentage of unemployed individuals
- Estimated Employed: Number of employed individuals
- Estimated Labour Participation Rate (%): Percentage of working-age population in the labor force

## Key Findings

1. **Sharp Increase During COVID-19**: Unemployment rates spiked dramatically in March-April 2020 when the pandemic began
2. **Regional Variations**: Different regions experienced varying levels of unemployment impact
3. **Urban-Rural Divide**: Urban areas generally saw higher unemployment rates than rural areas
4. **Gradual Recovery**: Unemployment began to decrease after May 2020, but remained above pre-pandemic levels

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Basic data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Interactive web application

## How to Run

1. Install the required packages:
   ```
   pip install streamlit pandas numpy matplotlib seaborn plotly
   ```

2. Run the Streamlit app:
   ```
   streamlit run unemployment_analysis.py
   ```

3. The application will open in your default web browser.

## Project Structure

- `unemployment_analysis.py`: Main Streamlit application
- `Unemployment_Rate_upto_11_2020.csv`: Dataset containing unemployment data
- `README.md`: Project documentation

## Future Enhancements

- Add predictive modeling to forecast unemployment trends
- Include more demographic factors (age, gender, education level)
- Incorporate economic indicators to correlate with unemployment rates
- Expand the dataset to include post-2020 recovery data

## Acknowledgments

This project was created as part of the CodeAlpha Data Science Internship program.
