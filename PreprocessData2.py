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


class PreprocessData2:

    def __init__(self, file_path, is_training=True, train_value=0.8, clear_old_files=False):
        self.file_path = file_path
        self.is_training = is_training
        self.train_value = train_value

        if self.is_training and (clear_old_files or not self._check_files()):
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
        X_train, X_test, y_train, y_test = train_test_split(news_dataframe["text"], y, test_size=0.2, random_state=53)

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

        clf = MultinomialNB()
        clf.fit(tfidf_train, y_train)
        pred = clf.predict(tfidf_test)
        score = accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)

        clf = SVC()
        clf.fit(tfidf_train, y_train)
        pred = clf.predict(tfidf_test)
        score = accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)


PreprocessData2("News Datasets/train.csv")