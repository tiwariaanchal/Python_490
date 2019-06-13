import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
nvb = GaussianNB()

DataSet = pd.read_csv('glass.csv')
X = DataSet[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
Y = DataSet['Type']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
nvb.fit(X_train, y_train)
print('Accuracy is {:.2f}'.format(nvb.score(X_train, y_train)))
print('Accuracy is {:.2f}'.format(nvb.score(X_test, y_test)))
