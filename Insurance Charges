import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import optuna
import time

# Start the timer
start_time = time.time()

pd.set_option('future.no_silent_downcasting', True)

# Load the dataset
data = pd.read_csv(r'c:\Users\kenne\GIT DEMO 04.13.2025\PROJECTS\DONE\PREDICTING INSURANCE CHARGES\insurance.csv')

# Generate a summary of the dataset

def data_details(df):
    print("Data Shape:", df.shape)
    print("Data Types:\n", df.dtypes)
    print("Missing Values:\n", df.isnull().sum())
    print("First 5 Rows:\n", df.head())
    print("Information:\n", df.info())

# Replacing/dropping missing values, replacing unexpected values, and converting data types
def data_cleaning(df):
    df['sex'] = df['sex'].replace({'M': 'male', 
                                   'man': 'male', 
                                   'F': 'female', 
                                   'woman': 'female'})
    df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)
    df = df[df["age"] > 0]
    df.loc[df["children"] < 0, "children"] = 0
    df["region"] = df["region"].str.lower()
    return df.dropna()

data_cleaned = data_cleaning(data)

def linear_regression_model(df):
    # Split the data into features and target variable
    X = df.drop(columns=['charges'])
    y = df['charges']
    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use multiple linear regression models to predict insurance charges
    model = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(),
        'Ridge Regression': Ridge(),
        'ElasticNet Regression': ElasticNet(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Support Vector Regressor': SVR(),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'K-Neighbors Regressor': KNeighborsRegressor()
    }

    param_grid = {
    'Linear Regression': {
        'fit_intercept': [True, False]
    },
    'Lasso Regression': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [1000, 5000, 10000]
    },
    'Ridge Regression': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    },
    'ElasticNet Regression': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.5, 0.9],
        'max_iter': [1000, 5000, 10000]
    },
    'Random Forest Regressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Support Vector Regressor': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    },
    'Decision Tree Regressor': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'K-Neighbors Regressor': {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    }

    best_model_name = None
    best_model_obj = None
    best_score_value = float('-inf')  # Initialize for maximization (use float('inf') if minimizing)

    # Perform hyperparameter tuning using GridSearchCV
    best_params = {}
    best_scores = {}

    for model_name, model_obj in model.items():
        grid_search = GridSearchCV(model_obj, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')  # Using neg MSE
        grid_search.fit(X_train, y_train)
        
        best_params[model_name] = grid_search.best_params_
        best_scores[model_name] = grid_search.best_score_  # Storing the best score for each model

    # Print best parameters for each model
    print("Best Hyperparameters:", best_params)
    print("Best Scores (neg MSE):", best_scores)

    # Training the model with the best parameters
    for model_name, model_obj in model.items():
        model_obj.set_params(**best_params[model_name])
        model_obj.fit(X_train, y_train)

        # Make predictions
        y_pred = model_obj.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        neg_mse = -mse  # Negating MSE for scoring consistency

        # Print results
        print(f"{model_name} - MSE: {mse}, Neg MSE: {neg_mse}, RMSE: {rmse}, R2: {r2}, MAE: {mae}, MAPE: {mape}")

        # Identify the best model based on R² (or choose another metric like neg MSE or RMSE)
        if r2 > best_score_value:  # Change this condition if optimizing another metric
            best_score_value = r2
            best_model_name = model_name
            best_model_obj = model_obj

    # Print the best model and its score
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Score (R²): {best_score_value}")
    end_time = time.time()

    # Convert total elapsed time to minutes and seconds
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"The time it takes to run the ML model is: {int(minutes)} minutes and {seconds:.2f} seconds")

# Execute the ML Function
# linear_regression_model(data_cleaned)

def deep_learning_pytorch(df):
    # Split the data into features and target variable
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define the neural network model
    class InsuranceNN(nn.Module):
        def __init__(self):
            super(InsuranceNN, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # Initialize the model, loss function, and optimizer
    model = InsuranceNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track loss for plotting
    loss_values = []

    # Start timer
    start_time = time.time()

    # Training the model
    num_epochs = 100
    for epoch in range(num_epochs):  # Epoch loop
        model.train()
        total_loss = 0  # Track loss per epoch

        for inputs, targets in train_loader:  # Batch loop
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()  # Sum up batch losses

        # Compute average loss per epoch
        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)  # Store for plotting

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()
        mse = mean_squared_error(y_test_tensor.numpy(), y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_tensor.numpy(), y_pred)
        mae = mean_absolute_error(y_test_tensor.numpy(), y_pred)
        mape = mean_absolute_percentage_error(y_test_tensor.numpy(), y_pred)
        print(f'\nEvaluation Results:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"The time it takes to run the DL model is: {int(minutes)} minutes and {seconds:.2f} seconds")

    # Fix the plotting issue
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

# Execute the Deep Learning Function
deep_learning_pytorch(data_cleaned)

"""
--> Documentation:
This script performs data cleaning, feature engineering, and model training to predict insurance charges using both traditional machine learning models and a deep learning model implemented in PyTorch.
The script also includes timing functionality to measure the execution time of both the machine learning and deep learning models, providing insights into computational efficiency.
The script is structured to allow for easy modification and extension, enabling further experimentation with different models, hyperparameters, and data preprocessing techniques.

--> Conclusion:
It demonstrates the use of various regression models to predict insurance charges, including hyperparameter tuning for each model. It also implements a deep learning model using PyTorch, showcasing the training process and evaluation metrics. The results indicate the effectiveness of both approaches, with the deep learning model providing a robust alternative to traditional regression methods.
We can see that using machine learning models, we can achieve a good prediction of insurance charges. Using multiple models, we can see that Random Forest Regressor with 81% R² yield the best results in terms of R² and other metrics.
The deep learning model, while more complex, can capture non-linear relationships in the data and may outperform traditional models on larger datasets or more intricate patterns. The choice between these approaches depends on the specific dataset characteristics and the complexity of the relationships involved. We have achieved 74% accuracy with the Deep Learning Model.

"""
