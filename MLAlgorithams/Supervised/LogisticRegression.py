from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
cancer = load_breast_cancer();
print(cancer.target_names)

X_train,X_test,Y_train,Y_test = train_test_split(cancer.data,cancer.target,train_size=0.75)

logisticRegression = LogisticRegression(max_iter=5000);
logisticRegression.fit(X_train,Y_train);

predictions = logisticRegression.predict(X_test);

print("F1 Score",f1_score(Y_test,predictions,average="weighted"));

print(classification_report(Y_test,predictions));

#predict 0th data value
predict = logisticRegression.predict([X_test[0]]);
print(X_test[0])
print("Predicted value:",predict)