from PreprocessData import PreprocessData
import os
import numpy as np
from GenerateClassifier import GenerateClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.utils import np_utils
import itertools


class Classifier:

    def __init__(self):
        self.preprocess = PreprocessData("News Datasets/train.csv", True, 0.8, False)
        self._load_data()
        self.classifier_list = []
        self._build_classifiers()


    def _load_data(self):

        self.model = self.preprocess.model
        self.data_X = self.preprocess.data_X
        self.data_Y = self.preprocess.data_Y
        self.tags = self.preprocess.tags

    def _get_random_data(self, seed=0):

        return self.preprocess.get_training_and_test_data(self.model,self.data_X,self.data_Y,self.tags,seed)

    def _build_classifiers(self):

        self.train_X, self.train_Y, self.test_X, self.test_Y = self._get_random_data(2)


        """
        # Gaussian NB
        name = "Naive Bayes"
        naive_bayes_clf,params = GenerateClassifier().get_classifer(name)
        accuracy ,precision, recall, f1 = self.get_results(naive_bayes_clf, self.train_X, self.train_Y, self.test_X, self.test_Y, name)
        self.display(name, accuracy ,precision, recall, f1)
        self.classifier_list.append((name,naive_bayes_clf))
        """

        # SVM
        name = "SVM"
        svm_clf,params = GenerateClassifier().get_classifer(name)
        #svm_clf = self.perform_grid_search(name,svm_clf,params, self.train_X, self.train_Y)
        accuracy ,precision, recall, f1 = self.get_results(svm_clf, self.train_X, self.train_Y, self.test_X, self.test_Y, name,False)
        self.display(name, accuracy ,precision, recall, f1)
        self.classifier_list.append((name,svm_clf))

        """

        # Logistic Regression
        name = "Logistic Regression"
        logistic_regression_clf,params = GenerateClassifier().get_classifer(name)
        #logistic_regression_clf = self.perform_grid_search(name, logistic_regression_clf, params, self.train_X, self.train_Y)
        accuracy ,precision, recall, f1 = self.get_results(logistic_regression_clf, self.train_X, self.train_Y, self.test_X, self.test_Y, name, False)
        self.display(name, accuracy ,precision, recall, f1)
        self.classifier_list.append((name,logistic_regression_clf))
        

        
        # Decision Trees
        name = "Decision Tree"
        decision_tree_clf,params = GenerateClassifier().get_classifer(name)
        #decision_tree_clf = self.perform_grid_search(name, decision_tree_clf, params, self.train_X, self.train_Y)

        accuracy ,precision, recall, f1 = self.get_results(decision_tree_clf, self.train_X, self.train_Y, self.test_X, self.test_Y, name,False)
        self.display(name, accuracy ,precision, recall, f1)
        self.classifier_list.append((name,decision_tree_clf))
        
        # Linear Discriminant Analysis
        name = "Linear Discriminant Analysis"
        linear_discriminant_clf,params = GenerateClassifier().get_classifer(name)
        accuracy ,precision, recall, f1 = self.get_results(linear_discriminant_clf, self.train_X, self.train_Y, self.test_X, self.test_Y, name)
        self.display(name, accuracy ,precision, recall, f1)
        self.classifier_list.append((name,linear_discriminant_clf))
        """

        """
        name = "Neural Network"
        neural_network, params = GenerateClassifier().get_classifer("Neural Network")
        self.train_network(neural_network,self.train_X,self.train_Y,self.test_X,self.test_Y)
        """

        #ensembling
        """
        self.ensemble_classifiers_voting()

    
        name = "Bagging Decision Trees"
        #self.ensemble_classifiers_bagging(decision_tree_clf,name)
        
        name = "Boosting Decision Trees"
        self.ensemble_classifiers_bagging(decision_tree_clf, name)
        """

    def display(self, name, accuracy ,precision, recall, f1):
        print("-"*5 + name + "-"*5)
        print("Accuracy", accuracy)
        print("Precision", precision)
        print("Recall", recall)
        print("F1 score", f1)

    def get_results(self, clf, train_x, train_y, test_x, test_y, name, grid_search=False):

        predictions = self.get_predictions(clf, train_x, train_y, test_x, grid_search)
        if len(test_y) > 0:
            accuracy, precision, recall, f1 = self._get_scores(test_y, predictions)

        # confusion matrix
        plt.figure(1)
        cm = confusion_matrix(test_y, predictions, labels=[0, 1])
        self.plot_confusion_matrix(cm, classes=["Reliable", "Unreliable"],title=name + " - Confusion Matrix")
        plt.savefig("Images/" + name + "- Confusion Matrix.png" )
        #plt.show()

        # ROC curve
        plt.figure(2)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        self.plot_ROC(false_positive_rate, true_positive_rate, thresholds, roc_auc, name)
        plt.savefig("Images/" + name + "- ROC.png")
        #plt.show()
        return accuracy, precision, recall, f1

    def get_predictions(self, clf, train_x, train_y, test_x, grid_search):
        if not grid_search:
            clf.fit(train_x, train_y)
        predictions = clf.predict(test_x)
        return predictions

    def _get_scores(self, test_y, predictions):
        accuracy = accuracy_score(test_y, predictions)
        precision = precision_score(test_y, predictions)
        recall = recall_score(test_y, predictions)
        f1 = f1_score(test_y, predictions)
        return accuracy, precision, recall, f1


    def perform_grid_search(self, name, clf, params,train_x, train_y):
        grid_search_clf = GridSearchCV(estimator=clf, param_grid=params, n_jobs=1)

        grid_search_clf.fit(train_x,train_y)

        print('Best estimator:', grid_search_clf.best_estimator_)
        print('Best score :', grid_search_clf.best_score_)

        return grid_search_clf

    def train_network(self, nn, train_x, train_y, test_x, test_y):
        print("Neural Network Structure",nn.summary())
        label_encoder = LabelEncoder()
        label_encoder.fit(train_y)
        encoded_train_y = np_utils.to_categorical((label_encoder.transform(train_y)))
        label_encoder.fit(test_y)
        encoded_test_y = np_utils.to_categorical((label_encoder.transform(test_y)))
        estimator = nn.fit(train_x, encoded_train_y, epochs=20, batch_size=64)
        print("Model Trained!")
        score = nn.evaluate(test_x, encoded_test_y)
        print("")
        print("Accuracy = " + format(score[1] * 100, '.2f') + "%")  # 92.69%

        probabs = nn.predict_proba(test_x)
        y_pred = np.argmax(probabs, axis=1)


    def ensemble_classifiers_voting(self):

        estimators = self.classifier_list

        # hard voting
        name = "Ensemble - Hard Voting"
        ensemble_clf_voting_hard = VotingClassifier(estimators,voting="hard")
        accuracy, precision, recall, f1 = self.get_results(ensemble_clf_voting_hard, self.train_X, self.train_Y, self.test_X,
                                                           self.test_Y, name)
        self.display(name, accuracy, precision, recall, f1)

        # soft voting
        name = "Ensemble - Soft Voting"
        ensemble_clf_voting_soft = VotingClassifier(estimators, voting="soft")
        accuracy, precision, recall, f1 = self.get_results(ensemble_clf_voting_soft, self.train_X, self.train_Y,
                                                           self.test_X,
                                                           self.test_Y, name)
        self.display(name, accuracy, precision, recall, f1)

    def ensemble_classifiers_bagging(self, clf, name):
        # bagging
        bagging_clf = BaggingClassifier(clf, max_samples=0.8, oob_score=True)
        accuracy, precision, recall, f1 = self.get_results(bagging_clf, self.train_X, self.train_Y, self.test_X,
                                                           self.test_Y, name, False)
        self.display(name, accuracy, precision, recall, f1)

    def ensemble_classifiers_boosting(self, clf, name):
        # bagging on decision trees
        boosting_clf = AdaBoostClassifier(n_estimators=10, base_estimator=clf,learning_rate=1)

        accuracy, precision, recall, f1 = self.get_results(boosting_clf, self.train_X, self.train_Y, self.test_X,
                                                           self.test_Y, name, False)
        self.display(name, accuracy, precision, recall, f1)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def plot_ROC(self, false_positive_rate, true_positive_rate, thresholds, roc_auc, name):
        plt.clf()
        plt.title(name + ' - Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.show()


classifier = Classifier()