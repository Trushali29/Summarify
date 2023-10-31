import numpy as np
import pandas as pd
from operator import itemgetter
import pre_processing as pp
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt


class TF_IDF :
    def __init__(self,all_documents):
        self.all_documents = all_documents
        self.TermFreq = {}
        self.InvertedDocFreq = {}
      
    def TermFrequency(self):
        for document in self.all_documents:
            words = document.split()
            for word in words:
                tf = round(words.count(word)/len(words),4)
                self.TermFreq[word] = tf
        return self.TermFreq

    def InvertedDocumentFrequency(self):
      for document in self.all_documents:
        for term in document.split():
          num_docs_with_term = sum(1 for doc in self.all_documents if term in doc)
          idf = np.log(len(self.all_documents) / num_docs_with_term)
          self.InvertedDocFreq[term] = idf
      return self.InvertedDocFreq
      
    def result_tf_idf(self): # working
      
      get_tf_idf = {}
      terms_all_docs = list(set(w for sent in self.all_documents for w in sent.split()))

      number_of_documents = len(self.all_documents)
      
    
      # get tf_idf values of all terms
      for term in terms_all_docs:
        get_tf_idf[term] = round(self.TermFreq[term] * self.InvertedDocFreq[term],4)

      # create a dataframe for tf-idf matrix 
      tf_idf_table = pd.DataFrame(columns = terms_all_docs,index = np.arange(number_of_documents))
      # add values in the dataframe
      for document in self.all_documents:
        for term in tf_idf_table:
          if (term in document):
            tf_idf_table.loc[self.all_documents.index(document)][term] = get_tf_idf[term] 
          else:
            tf_idf_table.loc[self.all_documents.index(document)][term] = 0

      return tf_idf_table
    

def sentence_scoring(article,tf_idf_table): #Max_len

  # preprocess the text
  sentences = sent_tokenize(article)
  filtered_sentences = []
  for sentence in sentences:
    res = " ".join(pp.preprocessing(sentence))
    #pos_tags_sent = " ".join(pp.pos_tagging(res))
    filtered_sentences.append(res)
  
  # calculate the scores for each sentences of the text
  sentences_scores = {}  # Dictionary to store sentence scores
  for sentence_idx in range(len(filtered_sentences)):
    words = word_tokenize(filtered_sentences[sentence_idx]) # sentence word tokens
    # Calculate the sentence score using TF-IDF values
    sentence_score = 0
    for word in words:
      if word in tf_idf_table.columns:  # Check if the word exists in TF-IDF columns
        sentence_score += tf_idf_table.loc[0][word]
    sentences_scores[sentence_idx] = round(sentence_score,4)
  # selecting the top sentences as a summary
  res = dict(sorted(sentences_scores.items(), key=itemgetter(1), reverse=True)[:5]) # get top 5 keys having high values #Max_len
  res = dict(sorted(res.items()))# sorting the keys in sequence
  
  summary = ''
  for keys in res:
    summary += ' '+ sentences[keys]
  return summary


def get_input_text(article): #Max_len
    clean_text = pp.preprocessing(article)
    # Calculate TF-IDF
    t1 = TF_IDF(clean_text)
    term_freq = t1.TermFrequency()
    inverted_doc_freq = t1.InvertedDocumentFrequency()
    tf_idf = t1.result_tf_idf()
    summary = sentence_scoring(article,tf_idf) #Max_len
    return summary


article = '''
British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid.
The 25-year-old has already smashed the British record over 60m hurdles twice this season, setting a new mark of 7.96 seconds to win the AAAs title.
"I am quite confident," said Claxton. "But I take each race as it comes. "As long as I keep up my training but not do too much I think there is a chance of a medal."
Claxton has won the national 60m hurdles title for the past three years but has struggled to translate her domestic success to the international stage.
Now, the Scotland-born athlete owns the equal fifth-fastest time in the world this year.
And at last week's Birmingham Grand Prix, Claxton left European medal favourite Russian Irina Shevchenko trailing in sixth spot.
For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form.
In previous seasons, the 25-year-old also contested the long jump but since moving from Colchester to London she has re-focused her attentions.
Claxton will see if her new training regime pays dividends at the European Indoors which take place on 5-6 March.
'''

print(get_input_text(article))

