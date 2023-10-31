import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import contractions
'''
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

'''

def preprocessing(text):
  # extract
  stops = set(stopwords.words('english'))
  ps = PorterStemmer()
  text = contractions.fix(text)
  sentences = sent_tokenize(text)
  filtered_text = []
  for sentence in sentences:
    filtered_sentence = []
    for word in sentence.split():
      word = word.lower().strip('",?.@#$%!*&()%')
      if ((word not in stops) and ((word.isalpha() == True) or word.isalnum() == True)):
        word = ps.stem(word)
        filtered_sentence.append(word)
    filtered_text.append(" ".join(filtered_sentence))
  return filtered_text

def pos_tagging(text):
  pos_tags = pos_tag(text.split())
  pos_tagged_noun_verb = []
  for word,tag in pos_tags:
    if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
      pos_tagged_noun_verb.append(word)
  return pos_tagged_noun_verb


def pos_tagging_glove(text):
    final_text = []
    for sent in text:
        words = word_tokenize(sent)
        pos_words = pos_tag(words)
        pos_tagged_words = []
        for word,tag in pos_words:
            if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
                pos_tagged_words.append(word)
        final_text.append(" ".join(pos_tagged_words))
    return final_text
        


