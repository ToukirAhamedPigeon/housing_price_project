## Load the Dataset

# ---------------------------------------------------------------------------------------------------
# importing required libraries numpy, pandas, matplotlib.pyplot for data load and analysis
# importing train_test_split for data split and Algorithms (LinearRegression, Ridge, Lasso) for model
# importing mean_absolute_error, mean_squared_error, r2_score to measure mae, mse, rmse and r2_score
# ---------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------
# Loading Datasaet with provided data_url & column_names
# --------------------------------------------------------

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df = pd.read_csv(data_url, delim_whitespace=True, names=columns)


# --------------------------------------------------------
# Custom function to print data in a nice way in result
# --------------------------------------------------------
def custom_print(label,values):
  print(label)
  print("")
  print(values)
  print("")
  print("")
  print("")

custom_print("# Printing 1st 5 Rows from the dataframe", housing_df.head())
custom_print("# Assigned Column Names", housing_df.columns)
custom_print("# Dataframe Info", housing_df.info())
custom_print("# Dataframe Null value Status", housing_df.isnull().sum())


## Split the Dataset

# -----------------------------
# Prepare Features and Target
# -----------------------------
custom_print("# housing_df total columns", len(housing_df.columns))
x = housing_df.drop("MEDV", axis=1)  # dropping target column MEDV. Keeping all features except target in X
y = housing_df["MEDV"]               # Y= target column MEDV (house price in $1000s)


custom_print("# After Dropping MEDV column, X total columns", len(x.columns))
custom_print("# total rows of x before split", len(x))
custom_print("# X 1st 5 rows", x.head())
custom_print("# Y or MEDV", y)



# -------------------------------------------------
# Train/Test Split 80% Train Data, 20% Test Data
# -------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

custom_print("# Study after train/test split", "")
custom_print("# total rows of x_train after split", f"{len(x_train)}  {round((len(x_train)*100)/len(x),3)}%")
custom_print("# total rows of x_test after split", f"{len(x_test)}  {round((len(x_test)*100)/len(x),3)}%")
custom_print("# X_train 1st 5 rows", x_train)
custom_print("# Y_train or MEDV", y_train)

## Train Models

# -----------------------------
# Train Linear Regression
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
custom_print("# Training is completed of model named linear_model with linear regression", "")

# -----------------------------
# Train Ridge Regression
# -----------------------------
ridge_model = Ridge(alpha=1.0)   # alpha = regularization strength
ridge_model.fit(x_train, y_train)
custom_print("# Training is completed of model named ridge_model with ridge regression, alpha=1.0", "")

# -----------------------------
# Train Lasso Regression
# -----------------------------
lasso_model = Lasso(alpha=0.1)   # alpha = regularization strength
lasso_model.fit(x_train, y_train)
custom_print("# Training is completed of model named lasso_model with lasso regression alpha=0.1", "")

#Linear Regression Prediction
y_linear_pred = linear_model.predict(x_test)
custom_print("# Linear model prediction", y_linear_pred)

#Ridge Regression Prediction
y_ridge_pred = ridge_model.predict(x_test)
custom_print("# Ridge model prediction", y_ridge_pred)

#Lasso Regression Prediction
y_lasso_pred = lasso_model.predict(x_test)
custom_print("# Lasso model prediction", y_lasso_pred)

## Evaluate Models

# -----------------------------
# Linear Regression
# -----------------------------
linear_mse = mean_squared_error(y_test, y_linear_pred) #calculating mean squared error
linear_rmse = np.sqrt(linear_mse) #calculating root mean squared error
linear_r2 = r2_score(y_test, y_linear_pred) #calculating R square score
linear_mae = mean_absolute_error(y_test, y_linear_pred) #calculating mean absolute error

custom_print("Linear Regression Model Performance on Test Data:","")
print(f"MSE : {linear_mse:.3f}")
print(f"RMSE: {linear_rmse:.3f}")
print(f"R²  : {linear_r2:.3f}")
print(f"MAE : {linear_mse:.3f}")
custom_print("--------------------------------------------------------------------------------","")

# -----------------------------
# Ridge Regression
# -----------------------------
ridge_mse = mean_squared_error(y_test, y_ridge_pred) #calculating mean squared error
ridge_rmse = np.sqrt(ridge_mse) #calculating root mean squared error
ridge_r2 = r2_score(y_test, y_ridge_pred) #calculating R square score
ridge_mae = mean_absolute_error(y_test, y_ridge_pred) #calculating mean absolute error

custom_print("Ridge Regression Model Performance on Test Data:","")
print(f"MSE : {ridge_mse:.3f}")
print(f"RMSE: {ridge_rmse:.3f}")
print(f"R²  : {ridge_r2:.3f}")
print(f"MAE : {ridge_mse:.3f}")
custom_print("--------------------------------------------------------------------------------","")

# -----------------------------
# Lasso Regression
# -----------------------------
lasso_mse = mean_squared_error(y_test, y_lasso_pred) #calculating mean squared error
lasso_rmse = np.sqrt(lasso_mse) #calculating root mean squared error
lasso_r2 = r2_score(y_test, y_lasso_pred) #calculating R square score
lasso_mae = mean_absolute_error(y_test, y_lasso_pred) #calculating mean absolute error

custom_print("Lasso Regression Model Performance on Test Data:","")
print(f"MSE : {lasso_mse:.3f}")
print(f"RMSE: {lasso_rmse:.3f}")
print(f"R²  : {lasso_r2:.3f}")
print(f"MAE : {lasso_mse:.3f}")
custom_print("--------------------------------------------------------------------------------","")

## Compare Results

# -----------------------------------------------------------------------
# Assigning evaluation results in object arrays to show in tabular format
# -----------------------------------------------------------------------
results = {}
results[1] = {'Model':'Linear Regression','MSE':linear_mse,'RMSE':linear_rmse,'R2':linear_r2,'MAE':linear_mae}
results[2] = {'Model':'Ridge Regression','MSE':ridge_mse,'RMSE':ridge_rmse,'R2':ridge_r2,'MAE':ridge_mae}
results[3] = {'Model':'Lasso Regression','MSE':lasso_mse,'RMSE':lasso_rmse,'R2':lasso_r2,'MAE':lasso_mae}

# -----------------------------------------------------------------------
# Dataframe from Array
# -----------------------------------------------------------------------

results_df = pd.DataFrame(results).T
results_df = results_df[['Model','MSE','RMSE','R2','MAE']]

custom_print("# Evaluated Results in Tabular format", results_df)

plt.figure(figsize=(15,4))

# ----------------------------------
# Plot Linear Predicted vs Actual
# ----------------------------------
plt.subplot(1,3,1)
plt.scatter(y_test, y_linear_pred, color="orange", alpha=0.7)
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual (Linear Regression)")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")

# ----------------------------------
# Plot Ridge Predicted vs Actual
# ----------------------------------
plt.subplot(1,3,2)
plt.scatter(y_test, y_ridge_pred, color="orange", alpha=0.7)
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual (Ridge Regression)")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="green")

# ----------------------------------
# Plot Ridge Predicted vs Actual
# ----------------------------------
plt.subplot(1,3,3)
plt.scatter(y_test, y_ridge_pred, color="orange", alpha=0.7)
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual (Lasso Regression)")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="blue")

plt.tight_layout()
plt.show()

