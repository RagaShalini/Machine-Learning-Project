import pandas as pd
import re
from nltk.corpus import stopwords
import string
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Doc2Vec
import numpy as np
import os


class PreprocessData:

    def __init__(self, file_path, is_training=True, train_value=0.8, clear_old_files = False):
        self.file_path = file_path
        self.is_training = is_training
        self.train_value = train_value

        if self.is_training and (clear_old_files or not self._check_files()):
            print("in process")
            self._process()



    def _check_files(self):

        if len(os.listdir("VectorRepresentations")) == 4:
            return True

        return False

    def _process(self):
        news_dataframe = pd.read_csv(self.file_path)
        #remove rows with missing text
        news_dataframe = news_dataframe[news_dataframe['text'].str.len() > 0]
        #print(news_dataframe.head(3))
        news_dataframe['text'] = news_dataframe['text'].apply(lambda txt:self.clean_text(txt))
        #print(news_dataframe.head(3))

        data_X, tags = self.construct_tags(news_dataframe['text'])
        data_Y = news_dataframe['label'].values
        #print(data_Y)

        dimension = 300

        self._build_and_train(data_X, data_Y, dimension, tags)



    def clean_text(self,text):

        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = text.lower().split()
        stopwords_list = set(stopwords.words("english"))
        text = [w for w in text if not w in stopwords_list]
        text = " ".join(text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text


    #constructs tags for Doc2Vec Training
    def construct_tags(self,data):
        sentences = []
        tags = []
        for index, row in data.iteritems():
            sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Tag_' +  str(index)]))
            tags.append(str(index))

        print(len(sentences),len(tags))
        return sentences,tags

    def _build_and_train(self, data_X, data_Y, dimension, tags):

        text_model = Doc2Vec(min_count=1, window=5, vector_size=dimension, sample=1e-4, negative=5, workers=7,
                             epochs=10,
                             seed=1,
                             dm=1)

        text_model.build_vocab(data_X)
        text_model.train(data_X, total_examples=text_model.corpus_count, epochs=text_model.iter)

        total_docs = len(data_X)
        train_size = int(total_docs * 0.8)
        test_size = total_docs - train_size
        print(train_size,test_size)

        train_X = np.zeros((train_size,dimension))
        train_Y = np.zeros(train_size)

        for idx in range(train_size):
            train_X[idx] = text_model.docvecs['Tag_' + tags[idx]]
            train_Y[idx] = data_Y[idx]
            #print(tags[idx])

        test_X = np.zeros((test_size,dimension))
        test_Y = np.zeros(test_size)

        cnt = train_size
        for idx in range(test_size):
            test_X[idx] = text_model.docvecs['Tag_' + tags[cnt]]
            test_Y[idx] = data_Y[cnt]
            #print(tags[cnt])
            cnt += 1

        self._save_vectors(train_X, train_Y, test_X, test_Y)


    def _save_vectors(self, train_X, train_Y, test_X, test_Y):
        file_template = "VectorRepresentations/{filename}.npy"
        np.save(file_template.format(filename="X_train"),train_X)
        np.save(file_template.format(filename="Y_train"), train_Y)
        np.save(file_template.format(filename="X_test"), test_X)
        np.save(file_template.format(filename="Y_test"), test_Y)

#preprocess = PreprocessData("News Datasets/train.csv", True, 0.8, False)