#!/usr/bin/env python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the training data
train_data = pd.read_csv('ProductSalesTrainingData.csv')

# Load the testing data
test_data = pd.read_csv('ProductSalesTestingData.csv')

print("Columns in the training dataset:", train_data.columns)
print("Columns in the testing dataset:", test_data.columns)

# Identify the target column
target_column = train_data.columns[-1]
print(f"Target column identified as: {target_column}")

# Define the preprocessing steps
numeric_features = ['ProductPrice', 'FirstMonthSale', 'SecondMonthSale']
categorical_features = ['ProductStyle', 'Fabric', 'Brand', 'FabricType', 'Silhouette', 'Season', 'Pattern', 'WashCase', 'ColorGroup']

# Create a pipeline for preprocessing numeric features
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create a pipeline for preprocessing categorical features
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply the preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Separate features and target for training data
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

# Separate features and target for testing data
if target_column in test_data.columns:
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
else:
    X_test = test_data
    y_test = None
    print("Target column not found in testing data.")

# Define the machine learning model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
if y_test is not None:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse:.2f}")
else:
    print("No target column in testing data to evaluate the model.")
    y_pred = model.predict(X_test)
    print("Predictions:", y_pred)

# Save the model to a file using pickle
with open('best_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully.")



# In[22]:







# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the training data
train_data = pd.read_csv('ProductSalesTrainingData.csv')

# Load the testing data
test_data = pd.read_csv('ProductSalesTestingData.csv')

# Define the target column
target_column = 'ThirdMonthSale'

# Define the feature columns
numeric_features = ['ProductPrice', 'TotalSales', 'FirstMonthSale', 'SecondMonthSale']
categorical_features = ['ProductStyle', 'Fabric', 'Brand', 'FabricType', 'Silhouette', 'Season', 'Pattern', 'WashCase', 'ColorGroup']

# Create a preprocessing pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Create a preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a column transformer to apply the preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the models
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Random Forest Regressor': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))]),
    'Gradient Boosting Regressor': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))])
}

# Split the training data into features and target
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

# Split the testing data into features
X_test = test_data

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_train, model.predict(X_train))
    print(f"{name}: MSE = {mse:.2f}")

# Compare the models
best_model = min(models, key=lambda x: mean_squared_error(y_train, models[x].predict(X_train)))
print(f"Best model: {best_model}")



# In[ ]:




