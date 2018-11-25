from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD

class GenerateClassifier:

    def get_classifer(self, classifier_name):

        hyper_parameters = []

        if classifier_name == 'Naive Bayes':
            classifier = GaussianNB()

        elif classifier_name == 'SVM':
            classifier = SVC(probability=True)
            hyper_parameters =  {'C': [1],'gamma': [0.001], 'kernel': ['rbf']}

        elif classifier_name == 'Logistic Regression':
            classifier = LogisticRegression()
            hyper_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

        elif classifier_name == 'Decision Tree':
            classifier = DecisionTreeClassifier(criterion="entropy")
            hyper_parameters = {'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}

        elif classifier_name == 'Linear Discriminant Analysis':
            classifier = LinearDiscriminantAnalysis()

        elif classifier_name == "Neural Network":
            classifier = self.create_neural_network()

        return classifier,hyper_parameters


    def create_neural_network(self):
        nn = Sequential()
        nn.add(Dense(256, input_dim=300, activation='relu', kernel_initializer='normal'))
        nn.add(Dropout(0.3))
        nn.add(Dense(256, activation='relu', kernel_initializer='normal'))
        nn.add(Dropout(0.5))
        nn.add(Dense(80, activation='relu', kernel_initializer='normal'))
        nn.add(Dense(2, activation="softmax", kernel_initializer='normal'))

        # gradient descent
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        # configure the learning process of the model
        nn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return nn