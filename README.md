# Regression

## Height Prediction using Linear Regression

### Overview

This project focuses on predicting a person's height based on their weight using linear regression. The dataset, 'height.csv,' contains information about individuals' weights and heights.

### Dependencies

Make sure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Data Exploration

The dataset is loaded using pandas:

```python
import pandas as pd

df = pd.read_csv('height.csv')
```

Here is a glimpse of the dataset:

```
   Weight  Height
0      70     175
1      55     165
2      75     180
3      50     160
4      65     172
```

Descriptive statistics of the dataset:

```python
df.describe()
```

### Data Visualization

Visualizing the relationship between weight and height using a scatter plot:

```python
import matplotlib.pyplot as plt

plt.scatter(df['Weight'], df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()
```

### Data Preprocessing

Checking for null values:

```python
df.isnull().sum()
```

Splitting the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X = df[['Weight']]
y = df['Height']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

Standardizing the features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Model Training

Training a linear regression model:

```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

### Model Evaluation

Extracting coefficients and intercept:

```python
regressor.coef_
regressor.intercept_
```

Visualizing the regression line:

```python
plt.scatter(X_train, y_train)
plt.scatter(X_train, regressor.predict(X_train))
plt.show()
```

Predicting heights on the test set:

```python
y_pred_test = regressor.predict(X_test)
```

### Model Evaluation Metrics

Calculating Mean Squared Error, Mean Absolute Error, and Root Mean Squared Error:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
```

### Model Performance

Calculating R-squared score:

```python
from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred_test)
print("R-squared score:", score)
```

Calculating Adjusted R-squared:

```python
adjusted_r2 = 1 - (1 - score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
print("Adjusted R-squared:", adjusted_r2)
```

### Conclusion

The linear regression model performs well in predicting heights based on weights, as indicated by the high R-squared score and low error metrics. Adjusted R-squared accounts for the number of predictors in the model.

Feel free to experiment with different models or feature engineering techniques to enhance predictions further.
