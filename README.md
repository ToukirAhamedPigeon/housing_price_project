# Housing Prices Prediction with Linear, Ridge, and Lasso Regression

This project demonstrates how to build, train, and evaluate **Linear Regression**, **Ridge Regression**, and **Lasso Regression** models using the [Boston Housing dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data). The goal is to predict housing prices (`MEDV`) based on various features like crime rate, number of rooms, property tax rate, etc.

---

## 📌 Project Workflow

### 1. Import Required Libraries
- **NumPy, Pandas, Matplotlib** → data handling and visualization
- **scikit-learn** → model building, training, and evaluation
  - `LinearRegression`, `Ridge`, `Lasso`
  - Metrics: `mean_absolute_error`, `mean_squared_error`, `r2_score`

### 2. Load Dataset
- Data source: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data`
- Column names:
  ```
  ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
  ```
- Target variable: **`MEDV`** (Median value of owner-occupied homes in $1000s)

### 3. Data Preparation
- Define **features (X)** and **target (y)**
- Train-test split: **80% training / 20% testing**

### 4. Train Models
- **Linear Regression**
- **Ridge Regression** (with alpha = 1.0)
- **Lasso Regression** (with alpha = 0.1)

### 5. Predictions
- Generate predictions for test dataset with each model.

### 6. Model Evaluation
- Metrics calculated:
  - **MSE** (Mean Squared Error)
  - **RMSE** (Root Mean Squared Error)
  - **R² Score** (Coefficient of Determination)
  - **MAE** (Mean Absolute Error)

- Results are stored in a dataframe for comparison.

### 7. Visualization
- Scatter plots: **Predicted vs Actual values** for all three models
- Helps visualize accuracy and fit quality

---

## 📊 Sample Output

### Evaluation Metrics Table
| Model              | MSE   | RMSE  | R²   | MAE   |
|--------------------|-------|-------|------|-------|
| Linear Regression  | ...   | ...   | ...  | ...   |
| Ridge Regression   | ...   | ...   | ...  | ...   |
| Lasso Regression   | ...   | ...   | ...  | ...   |

### Plots
- **Predicted vs Actual** for:
  - Linear Regression (red line = perfect fit)
  - Ridge Regression (green line = perfect fit)
  - Lasso Regression (blue line = perfect fit)

---

## 🚀 How to Run
1. Clone this repository
   ```bash
   git clone https://github.com/your-repo/housing-regression.git
   cd housing-regression
   ```
2. Install dependencies
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
3. Run the script
   ```bash
   python housing_regression.py
   ```

---

## 📌 Key Learnings
- How to apply **Linear, Ridge, and Lasso regression** in scikit-learn
- Comparing regression models with metrics like **MSE, RMSE, R², and MAE**
- Visualizing regression performance with scatter plots
- Regularization (Ridge & Lasso) helps in reducing overfitting compared to plain Linear Regression

---

## 📜 References
- Dataset: [Boston Housing Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

Colab Link: https://colab.research.google.com/drive/1pBToZTEHuzaVfy_By20eDc6jvS8LSuwW?usp=sharing 



Github Link: https://github.com/ToukirAhamedPigeon/housing_price_project