from nltk import word_tokenize

def calculate_f1_score(predict_summary, reference_summary):
    # Tokenize the predicted and reference summaries (you can use word_tokenize or custom tokenizer)
    predict_tokens = word_tokenize(predict_summary)
    reference_tokens = word_tokenize(reference_summary)
    
    # Calculate precision
    common_tokens = set(predict_tokens) & set(reference_tokens)
    precision = len(common_tokens) / len(predict_tokens) if len(predict_tokens) > 0 else 0.0
    
    # Calculate recall
    recall = len(common_tokens) / len(reference_tokens) if len(reference_tokens) > 0 else 0.0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print("precision {} recall {}  ,f1_score {}".format(precision,recall,f1_score))
    return int(round(f1_score,1)*100)



def max_length_summary(num_of_sentences):
    rate  = 30 # %
    return int(round((rate/100)*num_of_sentences,0))
