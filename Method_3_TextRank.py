import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings
import contractions
import pre_processing as pp

# Preprocess the text
stop_words = set(stopwords.words('english'))

# Create a graph using the TextRank algorithm
def create_graph(sentences):
    words_list = [pp.pos_tagging(" ".join(pp.preprocessing(sentence))) for sentence in sentences]
    # set of unique words of whole article
    words = set(word for words in words_list for word in words)
    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for word, i in word2id.items()}

    # Create a matrix where each row represents a sentence and each column represents a word
    matrix = np.zeros((len(sentences), len(words)))
    for i, words in enumerate(words_list):
        for word in words:
            matrix[i][word2id[word]] += 1
            
    # Calculate sentence similarity using cosine similarity - similartiy between sentences
    similarity_matrix = np.dot(matrix, matrix.T)
    norm_matrix = np.outer(np.linalg.norm(matrix, axis=1), np.linalg.norm(matrix, axis=1))
    similarity_matrix = similarity_matrix / norm_matrix
    
    # Apply the PageRank algorithm to rank sentences
    damping_factor = 0.85
    scores = np.ones(len(sentences))
    for _ in range(10):  # Number of iterations
        scores = (1 - damping_factor) + damping_factor * np.dot(similarity_matrix, scores)
    return scores

# Get the top N sentences as the summary
def get_summary(sentences, scores,top_n = 5):
    ranked_sentences = [(sentence, score) for sentence, score in zip(sentences, scores)]
    ranked_sentences.sort(key=lambda x: x[1], reverse=True)
    get_sentences = sorted(ranked_sentences[:top_n])
    # get real index values of the sentences
    #and sorted them according to their position in the article
    final_sent = {}
    for sentence, _ in get_sentences:
        final_sent[sentences.index(sentence)] = sentence
    summary = [final_sent[key] for key in final_sent]
    return " ".join(summary)


def get_input_text3(article):
    sentences = nltk.sent_tokenize(article)
    # Generate the summary
    scores = create_graph(sentences)
    predict_summary = get_summary(sentences, scores)
    return predict_summary



"""
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
print(get_input_text3(article))
"""

