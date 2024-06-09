# Car-price-prediction-using-linear-and-lasso-regression
Certainly! Here's a breakdown and brief explanation of the code provided for the car price prediction project using Linear Regression and Lasso Regression:

### Step-by-Step Explanation

#### 1. Import Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
```
- **pandas**: For data manipulation and analysis.
- **matplotlib.pyplot** and **seaborn**: For data visualization.
- **sklearn.model_selection.train_test_split**: For splitting the dataset into training and testing sets.
- **sklearn.linear_model.LinearRegression** and **Lasso**: For creating linear and lasso regression models.
- **sklearn.metrics**: For evaluating the performance of the models.

#### 2. Data Collection and Processing
```python
# Load the dataset
car_dataset = pd.read_csv('/content/cardekho.csv')

# Inspect the first few rows
car_dataset.head()

# Check the shape of the dataset
car_dataset.shape

# Get information about the dataset
car_dataset.info()

# Check for missing values
car_dataset.isnull().sum()

# Check the distribution of categorical data
print(car_dataset.fuel.value_counts())
print(car_dataset.seller_type.value_counts())
print(car_dataset.transmission.value_counts())
```
- **car_dataset.head()**: Displays the first five rows of the dataset.
- **car_dataset.shape**: Shows the number of rows and columns.
- **car_dataset.info()**: Provides information about the dataset, including data types and non-null counts.
- **car_dataset.isnull().sum()**: Checks for missing values in each column.
- **car_dataset.fuel.value_counts()**, etc.: Displays the count of unique values for categorical columns.

#### 3. Encoding Categorical Data
```python
# Encoding categorical columns
car_dataset.replace({'fuel': {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3}}, inplace=True)
car_dataset.replace({'seller_type': {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}}, inplace=True)
car_dataset.replace({'transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
car_dataset.replace({'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}}, inplace=True)
```
- **car_dataset.replace()**: Replaces categorical values with numerical values for easier processing.

#### 4. Splitting the Data
```python
# Splitting data into features and target
X = car_dataset.drop(['name', 'selling_price', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats'], axis=1)
Y = car_dataset['selling_price']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
```
- **X**: Features (independent variables) by dropping columns that are not needed.
- **Y**: Target variable (dependent variable).
- **train_test_split()**: Splits the data into training and testing sets with 10% of the data used for testing.

#### 5. Model Training and Evaluation: Linear Regression
```python
# Linear Regression Model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Training data prediction
training_data_prediction = lin_reg_model.predict(X_train)
print("R squared Error:", metrics.r2_score(Y_train, training_data_prediction))

# Visualize actual vs predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Testing data prediction
test_data_prediction = lin_reg_model.predict(X_test)
print("R squared Error:", metrics.r2_score(Y_test, test_data_prediction))

# Visualize actual vs predicted prices
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
```
- **LinearRegression()**: Initializes the linear regression model.
- **fit()**: Trains the model using the training data.
- **predict()**: Predicts the target variable for both training and testing data.
- **metrics.r2_score()**: Calculates the R-squared error to evaluate model performance.
- **plt.scatter()**: Visualizes the actual vs. predicted prices.

#### 6. Model Training and Evaluation: Lasso Regression
```python
# Lasso Regression Model
lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

# Training data prediction
training_data_prediction = lass_reg_model.predict(X_train)
print("R Squared Error: ", metrics.r2_score(Y_train, training_data_prediction))

# Visualize actual vs predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Testing data prediction
test_data_prediction = lass_reg_model.predict(X_test)
print("R squared Error:", metrics.r2_score(Y_test, test_data_prediction))

# Visualize actual vs predicted prices
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
```
- **Lasso()**: Initializes the lasso regression model.
- **fit()** and **predict()**: Similar to linear regression, these methods train the model and make predictions.
- **metrics.r2_score()**: Evaluates the model using the R-squared error.
- **plt.scatter()**: Visualizes the actual vs. predicted prices.

### Summary
This code implements a car price prediction system using linear and lasso regression models. The process involves:
1. Loading and inspecting the dataset.
2. Handling missing values and encoding categorical features.
3. Splitting the data into training and testing sets.
4. Training and evaluating both linear and lasso regression models.
5. Visualizing the performance of the models by comparing actual vs. predicted prices.

The evaluation is primarily done using the R-squared metric, which indicates how well the model explains the variability of the target variable. Visualization helps in understanding the model's performance visually.
