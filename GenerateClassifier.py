from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


class GenerateClassifier:

    def get_classifer(self, classifier_name):

        if classifier_name == 'Naive Bayes':
            classifier = GaussianNB()

        elif classifier_name == 'SVM':
            classifier = SVC(probability=True)

        elif classifier_name == 'Logistic Regression':
            classifier = LogisticRegression()

        elif classifier_name == 'Decision Tree':
            classifier = DecisionTreeClassifier(criterion="entropy")

        elif classifier_name == 'Linear Discriminant Analysis':
            classifier = LinearDiscriminantAnalysis()

        return classifier

