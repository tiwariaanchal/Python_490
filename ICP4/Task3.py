import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
DataSet = pd.read_csv('glass.csv')
X = DataSet[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
Y = DataSet['Type']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

SVC = LinearSVC()
SVC.fit(X_train, y_train)
Y_pred = SVC.predict(X_test)
acc_svc = round(SVC.score(X_train, y_train) * 100, 2)
acc_svc1 = round(SVC.score(X_test, y_test) * 100, 2)
print("Training SVM accuracy is:", acc_svc)
print("Testing SVM accuracy is:", acc_svc1)