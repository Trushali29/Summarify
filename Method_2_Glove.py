from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pre_processing as pp
from nltk import pos_tag

# Load the pre-trained GloVe embeddings
def load_glove_embeddings(file_path):
    word_embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = embedding
    return word_embeddings

# Load pre-trained GloVe embeddings (adjust path accordingly)
glove_embeddings = load_glove_embeddings('glove6/glove.6B.300d.txt')

# Calculate sentence embedding using average of word embeddings
def calculate_sentence_embedding(sentence, word_embeddings):
    words = sentence.split()        
    sentence_embedding = np.mean([word_embeddings.get(word, np.zeros(word_embeddings['a'].shape)) for word in words], axis=0)
    return sentence_embedding


def get_input_text2(article): #Max_len
    clean_text = pp.preprocessing(article)
    
    sentence_embeddings = []
    for sentence in clean_text:
        sent_embedding = calculate_sentence_embedding(sentence,glove_embeddings)
        sentence_embeddings.append(sent_embedding)
    similarity_matrix = cosine_similarity(sentence_embeddings)
    num_summary_sentences = 5 # top n sentences to take as a summary
    summary_indices = np.argsort(np.sum(similarity_matrix,axis = 1))[-num_summary_sentences:] #Max_len
    summary_indices = sorted(summary_indices)
    original_text = sent_tokenize(article)
    summary_sentences = [ original_text[idx] for idx in summary_indices]
    summary_res = "\n".join(summary_sentences)
    return summary_res
 
"""article = '''
The History Boys by Alan Bennett has been named best new play in the Critics' Circle Theatre Awards.
Set in a grammar school, the play also earned a best actor prize for star Richard Griffiths as teacher Hector.
The Producers was named best musical, Victoria Hamilton was best actress for Suddenly Last Summer and Festen's Rufus Norris was named best director.
The History Boys also won the best new comedy title at the Theatregoers' Choice Awards.
Partly based upon Alan Bennett's experience as a teacher, The History Boys has been at London's National Theatre since last May.
The Critics' Circle named Rebecca Lenkiewicz its most promising playwright for The Night Season, and Eddie Redmayne most promising newcomer for The Goat or, Who is Sylvia?
Paul Rhys was its best Shakespearean performer for Measure for Measure at the National Theatre and Christopher Oram won the design award for Suddenly Last Summer.
Both the Critics' Circle and Whatsonstage.com Theatregoers' Choice award winners were announced on Tuesday.
Chosen by more than 11,000 theatre fans, the Theatregoers' Choice Awards named US actor Christian Slater best actor for One Flew Over the Cuckoo's Nest.
Diana Rigg was best actress for Suddenly Last Summer, Dame Judi Dench was best supporting actress
for the RSC's All's Well That Ends Well and The History Boys' Samuel Barnett was best supporting actor.
'''
print(get_input_text2(article))
"""

