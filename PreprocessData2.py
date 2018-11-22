from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
import string
from sklearn.svm import SVC
import numpy as np
from GenerateClassifier import GenerateClassifier
import matplotlib.pyplot as plt
import itertools


class PreprocessData2:

    def __init__(self, file_path, is_training=True, train_value=0.8, clear_old_files=False):
        self.file_path = file_path
        self.is_training = is_training
        self.train_value = train_value

        if self.is_training :
            print("in process")
            self._process()

    def _check_files(self):

        if len(os.listdir("CountVectors")) == 4:
            return True

        return False

    def clean_text(self,text):

        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = text.lower().split()
        stopwords_list = set(stopwords.words("english"))
        text = [w for w in text if not w in stopwords_list]
        text = " ".join(text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def _process(self):
        news_dataframe = pd.read_csv(self.file_path)
        news_dataframe = news_dataframe[news_dataframe['text'].str.len() > 0]
        # print(news_dataframe.head(3))
        news_dataframe['text'] = news_dataframe['text'].apply(lambda txt: self.clean_text(txt))

        y = news_dataframe['label']



        X_train, X_test, y_train, y_test = train_test_split(news_dataframe["text"], y, test_size=0.2, random_state=2)

        porter = PorterStemmer()

        def stem_word(word):
            return porter.stem(word)

        reg_exp_digits = re.compile('\d')

        def tokenize(text):
            tokens = text.split()
            filtered_tokens = [stem_word(word) for word in tokens if len(word) > 2 and not reg_exp_digits.search(word)]
            return filtered_tokens

        # Initialize the `tfidf_vectorizer`
        tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.8, min_df=0.01,
                                           use_idf=True)

        # Fit and transform the training data
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)
        #print(tfidf_vectorizer.get_feature_names()[:10])
        # Transform the test set
        tfidf_test = tfidf_vectorizer.transform(X_test)

        name = "Naive Bayes"
        naive_bayes_clf = MultinomialNB()
        accuracy, precision, recall, f1 = self.get_results(naive_bayes_clf, tfidf_train, y_train, tfidf_test,
                                                           y_test, name)
        self.display(name, accuracy, precision, recall, f1)

        """
        # SVM
        name = "SVM"
        svm_clf = GenerateClassifier().get_classifer(name)
        accuracy, precision, recall, f1 = self.get_results(svm_clf, tfidf_train, y_train, tfidf_test,
                                                           y_test, name)
        self.display(name, accuracy, precision, recall, f1)
        #self.classifier_list.append((name, svm_clf))
        """

        # Logistic Regression
        name = "Logistic Regression"
        logistic_regression_clf = GenerateClassifier().get_classifer(name)
        accuracy, precision, recall, f1 = self.get_results(logistic_regression_clf, tfidf_train, y_train,
                                                           tfidf_test, y_test, name)
        self.display(name, accuracy, precision, recall, f1)
        #self.classifier_list.append((name, logistic_regression_clf))

        # Decision Trees
        name = "Decision Tree"
        decision_tree_clf = GenerateClassifier().get_classifer(name)
        accuracy, precision, recall, f1 = self.get_results(decision_tree_clf, tfidf_train, y_train, tfidf_test,
                                                           y_test, name)
        self.display(name, accuracy, precision, recall, f1)
        #self.classifier_list.append((name, decision_tree_clf))

        # Linear Discriminant Analysis
        name = "Linear Discriminant Analysis"
        linear_discriminant_clf = GenerateClassifier().get_classifer(name)
        accuracy, precision, recall, f1 = self.get_results(linear_discriminant_clf, tfidf_train, y_train,
                                                           tfidf_test, y_test, name)
        self.display(name, accuracy, precision, recall, f1)
        #self.classifier_list.append((name, linear_discriminant_clf))

    def _shuffle_data(self, total_docs, data_X, data_Y, seed_val=0):
        np.random.seed(seed_val)

        # Get length
        n_sample = total_docs

        order = np.random.permutation(n_sample)

        data_X = data_X[order]
        data_Y = data_Y[order]

        return data_X, data_Y

    def display(self, name, accuracy ,precision, recall, f1):
        print("-"*5 + name + "-"*5)
        print("Accuracy", accuracy)
        print("Precision", precision)
        print("Recall", recall)
        print("F1 score", f1)

    def get_results(self, clf, train_x, train_y, test_x, test_y, name):

        predictions = self.get_predictions(clf, train_x, train_y, test_x)
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

    def get_predictions(self, clf, train_x, train_y, test_x):
        clf.fit(train_x, train_y)
        predictions = clf.predict(test_x)
        return predictions

    def _get_scores(self, test_y, predictions):
        accuracy = accuracy_score(test_y, predictions)
        precision = precision_score(test_y, predictions)
        recall = recall_score(test_y, predictions)
        f1 = f1_score(test_y, predictions)
        return accuracy, precision, recall, f1

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

        # print(cm)
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
        # plt.show()


PreprocessData2("News Datasets/train.csv")