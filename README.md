# Property Price Prediction - Machine Learning Project

## Overview
This project focuses on predicting property prices using a web scrcaped dataset of house listings in Guargon, Haryana, India.The dataset includes various features such as property type, location, price, area, number of bedrooms, bathrooms, and additional amenities.The goal is to clean, preprocess, and analyze the data to build a machine learning model that can accurately predict property prices.

## Dataset
The dataset used in this project is named `house_cleaned.csv` and contains 964 rows with 20 columns. The dataset includes the following features:

- **property_name**: Name of the property.
- **property_type**: Type of property (e.g., house).
- **society**: Name of the society or community.
- **price**: Price of the property.
- **price_per_sqft**: Price per square foot.
- **area**: Area of the property.
- **areaWithType**: Detailed area description.
- **bedRoom**: Number of bedrooms.
- **bathroom**: Number of bathrooms.
- **balcony**: Number of balconies.
- **additionalRoom**: Additional rooms (e.g., servant room, study room).
- **address**: Address of the property.
- **floorNum**: Floor number.
- **facing**: Direction the property faces.
- **agePossession**: Age of the property.
- **nearbyLocations**: Nearby locations and amenities.
- **description**: Description of the property.
- **furnishDetails**: Details about the furnishings.
- **features**: Additional features of the property.
- **rating**: Ratings for the property.

## Data Preprocessing
The dataset underwent several preprocessing steps to prepare it for machine learning:

1. **Handling Missing Values**: Missing values in numerical columns were filled with the median, and categorical columns were filled with the mode.
2. **Feature Engineering**: New features were created based on existing ones, such as `furnishing_status` and `cumulative_weight`.
3. **Encoding Categorical Variables**: Categorical variables like `facing` were encoded using Label Encoding.
4. **Dropping Irrelevant Columns**: Columns like `description` and `rating` were dropped as they were not useful for prediction.
5. **Handling Outliers**: Outliers in latitude and longitude were handled by filling missing values with the median.

## Exploratory Data Analysis (EDA)
- **Latitude and Longitude Analysis**: The latitude and longitude of properties were analyzed to understand the geographical distribution.
- **Feature Importance**: Key features like the number of bedrooms, bathrooms, and area were analyzed for their impact on property prices.

## Machine Learning Model
The project aims to build a machine learning model to predict property prices. The following steps were taken:

1. **Feature Selection**: Relevant features were selected based on their importance and correlation with the target variable.
2. **Model Training**: Various machine learning models were trained on the preprocessed dataset.
3. **Model Evaluation**: The models were evaluated using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Usage
To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SuniL-0007/Property-Pricer.git
   cd Property-Pricer
