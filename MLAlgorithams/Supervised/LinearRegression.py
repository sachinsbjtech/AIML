from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


data = load_boston()
#dir(data)
print(data.data.shape)
print(data.target.shape)
print(data.DESCR)
X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.30)
#print(X_train)

scaler = StandardScaler()
scaler.fit(X_train)
#print(X_train)

#train the model using Linear Regression, calculate predicated
clf = LinearRegression()
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
expected = Y_test

X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)
#print(X_train)

print(clf.intercept_)
print(clf.coef_)

#print RMSE to validate the output
print('RMSE: %s' %np.sqrt(np.mean((predicted-expected) ** 2)))

#final plot the pediction line alon with true price
plt.figure(figsize=(4,3))
plt.scatter(expected,predicted)
plt.plot([0,60],[0,60],color='r')
plt.xlabel('True Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.show()


#sample test
Loc =113
print(X_test[Loc])
predict=clf.predict([X_test[Loc]])
print('Predicted:',predict, "Actual:",Y_test[Loc])


