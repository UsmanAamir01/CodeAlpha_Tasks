import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data(filepath):
    """Load and prepare the unemployment data"""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("Preprocessing data...")
    
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Check for and handle missing values
    if data.isnull().sum().sum() > 0:
        print("Missing values found in the dataset:")
        print(data.isnull().sum())
        data = data.dropna()
        print("Missing values have been dropped.")
    
    # Convert date columns to datetime if they exist
    date_columns = [col for col in data.columns if 'date' in col.lower() or 'month' in col.lower()]
    for col in date_columns:
        try:
            data[col] = pd.to_datetime(data[col])
            print(f"Converted {col} to datetime format.")
        except:
            print(f"Could not convert {col} to datetime.")
    
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
    
    print("Preprocessing complete.")
    return data

def analyze_unemployment_trends(data):
    """Analyze and visualize unemployment trends over time"""
    print("\n--- Analyzing Unemployment Trends ---")
    
    if 'Date' in data.columns and 'Unemployment_Rate' in data.columns:
        # Group by date and calculate mean unemployment rate
        monthly_data = data.groupby('Date')['Unemployment_Rate'].mean().reset_index()
        
        # Print summary statistics
        print("\nMonthly Unemployment Rate Summary:")
        print(monthly_data['Unemployment_Rate'].describe())
        
        # Create a line plot
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_data['Date'], monthly_data['Unemployment_Rate'], marker='o', linestyle='-')
        plt.title('Monthly Unemployment Rate Trend')
        plt.xlabel('Month')
        plt.ylabel('Unemployment Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # Add a vertical line for COVID-19 start (March 2020)
        if monthly_data['Date'].min() <= pd.to_datetime('2020-03-01') <= monthly_data['Date'].max():
            plt.axvline(x=pd.to_datetime('2020-03-01'), color='r', linestyle='--', label='COVID-19 Start')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('unemployment_trend.png')
        print("Unemployment trend plot saved as 'unemployment_trend.png'")
        
        # Calculate pre-COVID and during-COVID statistics
        covid_start = pd.to_datetime('2020-03-01')
        data['Period'] = data['Date'].apply(lambda x: 'Pre-COVID' if x < covid_start else 'During-COVID')
        
        period_stats = data.groupby('Period')['Unemployment_Rate'].agg(['mean', 'min', 'max']).reset_index()
        print("\nUnemployment Rate by Period:")
        print(period_stats)
        
        # Calculate percentage increase
        pre_covid = period_stats[period_stats['Period'] == 'Pre-COVID']['mean'].values[0]
        during_covid = period_stats[period_stats['Period'] == 'During-COVID']['mean'].values[0]
        percent_increase = ((during_covid - pre_covid) / pre_covid) * 100
        
        print(f"\nPercentage increase in unemployment rate during COVID-19: {percent_increase:.2f}%")
    else:
        print("Required columns not found in the dataset.")

def analyze_regional_differences(data):
    """Analyze and visualize regional differences in unemployment"""
    print("\n--- Analyzing Regional Differences ---")
    
    if 'Region' in data.columns and 'Unemployment_Rate' in data.columns:
        # Group by region and calculate statistics
        region_data = data.groupby('Region')['Unemployment_Rate'].agg(['mean', 'min', 'max']).reset_index()
        
        print("\nUnemployment Rate by Region:")
        print(region_data)
        
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Region', y='mean', data=region_data)
        plt.title('Average Unemployment Rate by Region')
        plt.xlabel('Region')
        plt.ylabel('Average Unemployment Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('regional_unemployment.png')
        print("Regional unemployment plot saved as 'regional_unemployment.png'")
        
        # Analyze COVID-19 impact by region
        if 'Period' in data.columns:
            region_period_data = data.groupby(['Region', 'Period'])['Unemployment_Rate'].mean().reset_index()
            
            # Create a grouped bar plot
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Region', y='Unemployment_Rate', hue='Period', data=region_period_data)
            plt.title('COVID-19 Impact on Unemployment by Region')
            plt.xlabel('Region')
            plt.ylabel('Average Unemployment Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('covid_impact_by_region.png')
            print("COVID-19 impact by region plot saved as 'covid_impact_by_region.png'")
            
            # Calculate percentage increase by region
            print("\nPercentage Increase in Unemployment Rate by Region:")
            for region in data['Region'].unique():
                region_data = data[data['Region'] == region]
                pre_covid = region_data[region_data['Period'] == 'Pre-COVID']['Unemployment_Rate'].mean()
                during_covid = region_data[region_data['Period'] == 'During-COVID']['Unemployment_Rate'].mean()
                percent_increase = ((during_covid - pre_covid) / pre_covid) * 100
                print(f"{region}: {percent_increase:.2f}%")
    else:
        print("Required columns not found in the dataset.")

def analyze_urban_rural_differences(data):
    """Analyze and visualize urban vs rural unemployment differences"""
    print("\n--- Analyzing Urban vs Rural Differences ---")
    
    if 'Area' in data.columns and 'Unemployment_Rate' in data.columns:
        # Group by area and calculate statistics
        area_data = data.groupby('Area')['Unemployment_Rate'].agg(['mean', 'min', 'max']).reset_index()
        
        print("\nUnemployment Rate by Area:")
        print(area_data)
        
        # Create a bar plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Area', y='mean', data=area_data)
        plt.title('Average Unemployment Rate by Area (Urban vs Rural)')
        plt.xlabel('Area')
        plt.ylabel('Average Unemployment Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('urban_rural_unemployment.png')
        print("Urban vs rural unemployment plot saved as 'urban_rural_unemployment.png'")
        
        # Analyze COVID-19 impact by area
        if 'Period' in data.columns:
            area_period_data = data.groupby(['Area', 'Period'])['Unemployment_Rate'].mean().reset_index()
            
            # Create a grouped bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Area', y='Unemployment_Rate', hue='Period', data=area_period_data)
            plt.title('COVID-19 Impact on Urban vs Rural Unemployment')
            plt.xlabel('Area')
            plt.ylabel('Average Unemployment Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('covid_impact_by_area.png')
            print("COVID-19 impact by area plot saved as 'covid_impact_by_area.png'")
            
            # Calculate percentage increase by area
            print("\nPercentage Increase in Unemployment Rate by Area:")
            for area in data['Area'].unique():
                area_data = data[data['Area'] == area]
                pre_covid = area_data[area_data['Period'] == 'Pre-COVID']['Unemployment_Rate'].mean()
                during_covid = area_data[area_data['Period'] == 'During-COVID']['Unemployment_Rate'].mean()
                percent_increase = ((during_covid - pre_covid) / pre_covid) * 100
                print(f"{area}: {percent_increase:.2f}%")
    else:
        print("Required columns not found in the dataset.")

def main():
    """Main function to run the analysis"""
    print("=== Unemployment Analysis During COVID-19 ===\n")
    
    # Load and preprocess data
    filepath = "Unemployment_Rate_upto_11_2020.csv"
    df = load_data(filepath)
    
    if df is not None:
        data = preprocess_data(df)
        
        # Display basic information
        print("\nDataset Overview:")
        print(f"Number of Records: {data.shape[0]}")
        print(f"Number of Features: {data.shape[1]}")
        if 'Date' in data.columns:
            print(f"Time Period: {data['Date'].min().strftime('%b %Y')} - {data['Date'].max().strftime('%b %Y')}")
        
        # Display the first few rows
        print("\nSample Data:")
        print(data.head())
        
        # Run analyses
        analyze_unemployment_trends(data)
        analyze_regional_differences(data)
        analyze_urban_rural_differences(data)
        
        print("\nAnalysis complete. Check the generated plots for visualizations.")
    else:
        print("Analysis could not be performed due to data loading issues.")

if __name__ == "__main__":
    main()
