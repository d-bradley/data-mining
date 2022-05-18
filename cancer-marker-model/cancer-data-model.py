# numeric python and plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Scoring for classifiers
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score

# Classifiers from scikit-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


################    CLASSIFIERS     ################

svm = ["0.01",
       "0.1",
       "1",
       "10",
       "100"]

dt_gini = ["2",
           "5",
           "10",
           "20"]

dt_ig = ["2",
         "5",
         "10",
         "20"]

classifiers = ["SVM, C=0.01",
               "DT-gini, k=2",
               "DT-ig, k=2",
               "LDA",
               "Random Forest"]

svm_classifier = [SVC(kernel="linear", C=0.01),
                   SVC(kernel="linear", C=0.1),
                   SVC(kernel="linear", C=1),
                   SVC(kernel="linear", C=10),
                   SVC(kernel="linear", C=100)]

dt_gini_classifier = [DecisionTreeClassifier(max_leaf_nodes=2),
                      DecisionTreeClassifier(max_leaf_nodes=5),
                      DecisionTreeClassifier(max_leaf_nodes=10),
                      DecisionTreeClassifier(max_leaf_nodes=20)]

dt_ig_classifier = [DecisionTreeClassifier(max_leaf_nodes=2, criterion="entropy"),
                     DecisionTreeClassifier(max_leaf_nodes=5, criterion="entropy"),
                     DecisionTreeClassifier(max_leaf_nodes=10, criterion="entropy"),
                     DecisionTreeClassifier(max_leaf_nodes=20, criterion="entropy")]

compare_classifiers = [SVC(kernel="linear", C=0.01),
                       DecisionTreeClassifier(max_leaf_nodes=2),
                       DecisionTreeClassifier(max_leaf_nodes=2, criterion="entropy"),
                       LinearDiscriminantAnalysis(),
                       RandomForestClassifier()]


#################    DATASETS     #################

X_train = np.genfromtxt(r"/Users/daniel/PycharmProjects/cancer-data-train.csv", delimiter=',', usecols=range(0,30))
y_train = pd.read_csv(r"/Users/daniel/PycharmProjects/cancer-data-train.csv", usecols=[30], header=None).to_numpy()
for index, value in enumerate(y_train):
    if value == "M":
        y_train[index] = 1
    elif value == "B":
        y_train[index] = 0
y_train = y_train.reshape(len(y_train),)
y_train = y_train.astype('int')

X_test = np.genfromtxt(r"/Users/daniel/PycharmProjects/cancer-data-test.csv", delimiter=',', usecols=range(0,30))
y_test = pd.read_csv(r"/Users/daniel/PycharmProjects/cancer-data-test.csv", usecols=[30], header=None).to_numpy()
for index, value in enumerate(y_test):
    if value == "M":
        y_test[index] = 1
    elif value == "B":
        y_test[index] = 0
y_test = y_test.reshape(len(y_test),)
y_test = y_test.astype('int')


################    TRAIN AND PLOT     ################

f_measure_svm = []
f_measure_gini = []
f_measure_ig = []
f_measure_all = []
recall = []
precision = []
accuracy = []

for name, clf in zip(svm, svm_classifier):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    f_measure_svm.append(np.average(cross_val_score(clf, X_train, y_pred, cv=10)))
print("F-measure (CV_SVM): ", f_measure_svm)
plt.figure(1)
plt.xlabel('C Value')
plt.ylabel('Average F-measure')
plt.title('Average F-measure of 10-fold Cross Validation\n(SVM)')
plt.plot(svm, f_measure_svm, marker='o', markerfacecolor='blue', markersize=2, label="SVM")

for name, clf in zip(dt_gini, dt_gini_classifier):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    f_measure_gini.append(np.average(cross_val_score(clf, X_train, y_pred, cv=10)))
print("F-measure (CV_GINI): ", f_measure_gini)
plt.figure(2)
plt.xlabel('K Value')
plt.ylabel('Average F-measure')
plt.title('Average F-measure of 10-fold Cross Validation\n(DT-GINI)')
plt.plot(dt_gini, f_measure_gini, marker='o', markerfacecolor='orange', markersize=2, label="DT-gini")

for name, clf in zip(dt_ig, dt_ig_classifier):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    f_measure_ig.append(np.average(cross_val_score(clf, X_train, y_pred, cv=10)))
print("F-measure (CV_IG): ", f_measure_ig)
plt.figure(3)
plt.xlabel('K Value')
plt.ylabel('Average F-measure')
plt.title('Average F-measure of 10-fold Cross Validation\n(DT-IG)')
plt.plot(dt_ig, f_measure_ig, marker='o', markerfacecolor='green', markersize=2, label="DT-ig")


for name, clf in zip(classifiers, compare_classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f_measure_all.append(f1_score(y_test, y_pred, average='weighted'))
    precision.append(average_precision_score(y_test, y_pred, pos_label=1))
    recall.append(recall_score(y_test, y_pred, average='weighted', pos_label=1))
print("F-measure: ", f_measure_all)
print("Precision: ", precision)
print("Recall: ", recall)
plt.figure(4)
plt.xlabel('Classifier')
plt.ylabel('Average Class F-measure')
plt.title('Average F-measure Vs. Classifier')
plt.bar(classifiers, f_measure_all, color='green', width=0.4)
plt.figure(5)
plt.xlabel('Classifier')
plt.ylabel('Average Precision')
plt.title('Average Precision Vs. Classifier')
plt.bar(classifiers, precision, color='blue', width=0.4)
plt.figure(6)
plt.xlabel('Classifier')
plt.ylabel('Average Recall')
plt.title('Average Recall Vs. Classifier')
plt.bar(classifiers, recall, color='orange', width=0.4)

plt.tight_layout()
plt.show()
