import pandas as pd
import re
from nltk.corpus import stopwords
import string
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Doc2Vec
import numpy as np
import os
from gensim.test.utils import get_tmpfile

class PreprocessData:

    def __init__(self, file_path, is_training=True, train_value=0.8, clear_old_files = False):
        self.file_path = file_path
        self.is_training = is_training
        self.train_value = train_value
        self.fname = get_tmpfile("Fake_News_Doc2Vec")
        self.dimension = 300

        if self.is_training and (clear_old_files or not self._check_files()):
            print("in process")
            self.model, self.data_X, self.data_Y, self.tags = self._process()

        else:
            print("loading saved model")
            self.model, self.data_X, self.data_Y, self.tags = self._load_data()



    def _check_files(self):

        if len(os.listdir("VectorRepresentations")) == 3:
            return True

        return False

    def _process(self):
        news_dataframe = pd.read_csv(self.file_path)
        #remove rows with missing text
        news_dataframe = news_dataframe[news_dataframe['text'].str.len() > 0]
        #print(news_dataframe.head(3))
        news_dataframe['text'] = news_dataframe['text'].apply(lambda txt:self._clean_text(txt))
        #print(news_dataframe.head(3))

        data_X, tags = self._construct_tags(news_dataframe['text'])
        data_Y = news_dataframe['label'].values
        #print(data_Y)



        return self._build_vectors(data_X, data_Y, tags)

    def _clean_text(self,text):

        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = text.lower().split()
        stopwords_list = set(stopwords.words("english"))
        text = [w for w in text if not w in stopwords_list]
        text = " ".join(text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text


    #constructs tags for Doc2Vec Training
    def _construct_tags(self,data):
        sentences = []
        tags = []
        for index, row in data.iteritems():
            sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Tag_' +  str(index)]))
            tags.append(str(index))

        print(len(sentences),len(tags))
        return sentences,tags

    def _build_vectors(self, data_X, data_Y, tags):

        text_model = Doc2Vec(min_count=1, window=5, vector_size=self.dimension, sample=1e-4, negative=5, workers=7,
                             epochs=10,
                             seed=1,
                             dm=1)

        text_model.build_vocab(data_X)
        text_model.train(data_X, total_examples=text_model.corpus_count, epochs=text_model.iter)

        #save the model for future use
        text_model.save(self.fname)

        self._save_vectors(data_X, data_Y, tags)

        return text_model, data_X, data_Y, tags

    # splits data to training and test - call from Classifier
    def get_training_and_test_data(self, text_model, data_X, data_Y, tags, seed_value = 0):

        total_docs = len(data_X)

        data_X, data_Y, tags = self._shuffle_data(total_docs, data_X, data_Y, tags, seed_value)

        train_size = int(total_docs * self.train_value)
        test_size = total_docs - train_size
        print(train_size, test_size)

        train_X = np.zeros((train_size, self.dimension))
        train_Y = np.zeros(train_size)

        for idx in range(train_size):
            train_X[idx] = text_model.docvecs['Tag_' + tags[idx]]
            train_Y[idx] = data_Y[idx]
            # print(tags[idx])

        test_X = np.zeros((test_size, self.dimension))
        test_Y = np.zeros(test_size)

        cnt = train_size
        for idx in range(test_size):
            test_X[idx] = text_model.docvecs['Tag_' + tags[cnt]]
            test_Y[idx] = data_Y[cnt]
            # print(tags[cnt])
            cnt += 1

        return train_X, train_Y, test_X, test_Y

    def _save_vectors(self, data_X, data_Y, tags):
        file_template = "VectorRepresentations/{filename}.npy"
        np.save(file_template.format(filename="data_X"),data_X)
        np.save(file_template.format(filename="data_Y"), data_Y)
        np.save(file_template.format(filename="tags"), tags)

    def _load_data(self):
        # load the model
        text_model = Doc2Vec.load(self.fname)

        file_template = "VectorRepresentations/{filename}.npy"
        data_X = np.load(file_template.format(filename="data_X"))
        data_Y = np.load(file_template.format(filename="data_Y"))
        tags = np.load(file_template.format(filename="tags"))

        return text_model, data_X, data_Y, tags

    def _shuffle_data(self,total_docs, data_X, data_Y, tags, seed_val=0):
        np.random.seed(seed_val)

        # Get length
        n_sample = total_docs

        order = np.random.permutation(n_sample)

        data_X = data_X[order]
        data_Y = data_Y[order]
        tags = np.array(tags)
        tags = tags[order]

        return data_X, data_Y, tags

#preprocess = PreprocessData("News Datasets/train.csv", True, 0.8, False)