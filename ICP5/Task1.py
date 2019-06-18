import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


wine_quality = pd.read_csv('winequality-red.csv')

wine_quality.quality.describe()

#Working with Numeric Features
numeric_features = wine_quality.select_dtypes(include=[np.number])

corr = numeric_features.corr()

print (corr['quality'].sort_values(ascending=False)[:4], '\n')

##Null values

'''null_columns=wine_quality.columns[wine_quality.isnull().any()]
wine_quality[null_columns].isnull().sum()
print(wine_quality[wine_quality.isnull().any(axis=1)][null_columns].head())'''



nulls = pd.DataFrame(wine_quality.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


##handling missing value
data = wine_quality.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

##Build a linear model
y = np.log(wine_quality.quality)
X = data.drop(['quality'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)


##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))