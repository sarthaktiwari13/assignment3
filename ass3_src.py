"""
Python Module to perform preprocessing for
a sentiment analysis task with a CNN + Embedding model.
"""
import re
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from urllib.request import urlopen

class TweetProcessor():
    """Class to Pre Process Tweets"""
    def __init__(self, max_length_tweet=50, max_lenght_dictionary=100):
        self.max_length_tweet = max_length_tweet
        self.max_lenght_dictionary = max_lenght_dictionary

    def clean_text(self, text):
        """Removes URL and other unwanted sequences from the text"""
        text = text.lower()
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
        text = re.sub(r"[0-9]+",'',text)
        text = re.sub(r'[^\w]', ' ', text)
        text = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", '', text)
        text = re.sub(r"@[a-zA-Z0-9]+",'',text)
        text = re.sub(r'https?:\/\/.*[\r\n]*', '' ,text)
        return text

    def tokenize_text(self, text):
        """ Tokenize the Text """
        tk = TweetTokenizer()
        token =  tk.tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in text if not w in stop_words]
        filtered_sentence = []
        for w in token:
            if w not in stop_words:
                filtered_sentence.append(w)
        s = " "
        s = s.join(filtered_sentence)
        return tk.tokenize(s)
    def token_to_index(self, text):
        """Token to Index """
        glove = urlopen('https://curren-tipnis-ass3.s3.amazonaws.com/glove.twitter.27B.25d.txt')
        #glove_file = "glove.twitter.27B.25d.txt"
        emb_dict = {}
        #glove = open(glove_file)
        for line in glove:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            emb_dict[word] = vector
        embd = []
        for word in text:
            index = emb_dict.get(word)
            if type(index) == np.ndarray:
                embd.append(index)
        return embd
    def pad_sequence(self, t_index):
        """padding a list of indices with 0 until a maximum length (max_length_tweet)"""
        l = len(t_index)
        if l < self.max_length_tweet:
            req_d = self.max_length_tweet - l
            t_index.extend([np.zeros_like(t_index[0])] * req_d)
        elif l > self.max_length_tweet:
            t_index = t_index[:self.max_length_tweet].copy()

        return t_index
