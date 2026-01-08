## 🏨 Hotel Reservations Classification

This project focuses on analyzing a hotel reservation dataset and building machine learning models to predict booking status (canceled or not canceled). The goal is to provide insights into factors influencing hotel cancellations and develop a robust predictive system.


## 🌟 Project Overview

This project undertakes a comprehensive machine learning pipeline for hotel reservation data, covering:

Data Loading and Initial Inspection: Understanding the structure and basic characteristics of the dataset.

Data Preprocessing: Cleaning, transforming, and encoding the data to prepare it for model training.

Exploratory Data Analysis (EDA): Gaining insights into the dataset through various statistical summaries and visualizations.

Model Building and Evaluation: Training and evaluating multiple classification models to predict booking outcomes.

## 📊 Dataset

The dataset used in this project is the "Hotel Reservation Database," which can be viewed and downloaded from Kaggle.

The dataset includes various features related to hotel bookings, such as:

Booking_ID

no_of_adults

no_of_children

no_of_weekend_nights

no_of_week_nights

type_of_meal_plan

required_car_parking_space

room_type_reserved

lead_time

arrival_year

arrival_month

arrival_date

market_segment_type

repeated_guest

no_of_previous_cancellations

no_of_previous_bookings_not_canceled

avg_price_per_room

no_of_special_requests

booking_status (Target variable: 'Not_Canceled' or 'Canceled')

## 🚀 Key Steps & Analysis

The Jupyter notebook (hotel-reservations.ipynb) details the following steps:

1. Data Loading and Initial Exploration
Reading the dataset into a Pandas DataFrame.

Displaying the first few rows (.head()).

Checking data types (.dtypes).

Determining the number of samples and features (.shape).

Identifying numeric and categorical columns.

Checking for missing values (.isnull().sum()).

2. Data Cleaning and Encoding
Inspecting unique values for categorical columns (type_of_meal_plan, room_type_reserved, market_segment_type, booking_status).

Encoding categorical features into numerical representations using replace() for various columns.

Dropping the Booking_ID column as it's not relevant for modeling.

3. Data Splitting and Scaling
Separating the dataset into input features (data_input) and the target variable (data_output).

Splitting the data into training, validation, and testing sets using train_test_split from sklearn.model_selection.

Checking for data imbalance in the training target variable (y_train.value_counts()) and visualizing it.

Applying StandardScaler from sklearn.preprocessing to scale the numerical features in the training, validation, and test sets.

4. Model Building and Evaluation
Helper Function: An evaluate_model function is defined to streamline the training and evaluation process for different classifiers, reporting training and validation accuracy.

Voting Classifier:

Combines DecisionTreeClassifier, LogisticRegression, and SVC.

Evaluates the ensemble's performance.

K-Nearest Neighbors (KNN) Model:

Trains a KNeighborsClassifier.

Evaluates its performance.

Extra Trees Classifier Model:

Utilizes ExtraTreesClassifier.

Performs GridSearchCV to find optimal hyperparameters (n_estimators, max_depth).

Reports accuracy on training and validation sets with the best model.

Random Forest Classifier Model:

Implements RandomForestClassifier.

Applies GridSearchCV for hyperparameter tuning (n_estimators, max_depth).

Reports accuracy on training and validation sets with the best model.

Gradient Boosting Classifier Model:

Employs GradientBoostingClassifier.

Conducts GridSearchCV for hyperparameter optimization (n_estimators, max_depth, learning_rate).

Reports accuracy on training and validation sets with the best model.

Bagging Classifier Model:

Uses BaggingClassifier with LogisticRegression as the base estimator.

Evaluates its performance.

Model Comparison: A DataFrame and bar plot are generated to visually compare the training and validation accuracies of all trained models.

## 🛠️ Technologies & Libraries

Python: The primary programming language.

Jupyter Notebook: For interactive development and documentation.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For machine learning models (e.g., train_test_split, StandardScaler, DecisionTreeClassifier, LogisticRegression, SVC, VotingClassifier, KNeighborsClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier), preprocessing, and evaluation metrics (accuracy_score, GridSearchCV).

Matplotlib: For creating static visualizations, particularly for model comparison.

## 💡 How to Run
To execute this analysis:

Clone the Repository:

git clone <repository_url> # Replace with your repository URL
cd <repository_name>

Install Dependencies:
It's recommended to use a virtual environment.

pip install pandas numpy scikit-learn matplotlib jupyter

Download Dataset:

Ensure the Hotel Reservations.csv dataset is downloaded from the Kaggle link provided and placed in the appropriate directory (or update the path in the notebook).

Run Jupyter Notebook:

jupyter notebook

Open hotel-reservations.ipynb and execute all cells sequentially to perform the entire analysis.
