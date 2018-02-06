import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import gensim

from flaskexample.preprocess import standardize_question_text, nlp_preprocess_text
from sklearn.metrics.pairwise import cosine_similarity



def generate_responses(input_question, w2v_model, train_w2v_df, train_df, count_cutoff):
    """
    Generate Top5 responses (or less)
    
    :type string: input_question
    :type gensim.models.word2vec.Word2Vec: w2v_model
    :type pandas DataFrame: train_w2v_df
    :type pandas DataFrame: train_df
    :rtype : filtered_msg
    :rtype list: filtered_reply_text_list
    :rtype list: filtered_reply_cosine_list
    :rtype int: count
    """ 
    # preprocess input question to wordlist  
    input_question_proc = standardize_question_text(input_question)
    input_question_nlp_proc = nlp_preprocess_text(input_question_proc)
    input_question_wordlist = word_tokenize(input_question_nlp_proc)

    # generate input w2v, averaged
    input_w2v = compute_avg_w2v_vector(w2v_model.wv, input_question_wordlist)
    input_w2v = np.array([input_w2v])

    # compare cosine similarity
    cosine_sim = cosine_similarity(input_w2v, train_w2v_df)

    # generate top20 responses's index and cosine value
    top20_indices, top20_cosine = generate_top20_candidates(cosine_sim)
    top20_response_df = pd.DataFrame({'reply_stander_proc': [train_df.iloc[i]['reply_stander_proc'] for i in top20_indices],
                                      'cosine_sim': top20_cosine}
                                    )
    filtered_reply_text_list, filtered_reply_cosine_list, count = filter_reply_msg(top20_response_df, 0.5, count_cutoff)

    return filtered_reply_text_list, filtered_reply_cosine_list, count


def compute_avg_w2v_vector(w2v_dict, text_nlp_proc):
    """
    Compute average word2vec vector
    
    :type gensim.models.keyedvectors.KeyedVectors: w2v_dict (word2vec.wv file)
    :type string: text_nlp_proc (nlp processed text)
    :rtype numpy.ndarray: result
    """
    SIZE = 50 # size of the w2v dimension
    list_of_word_vectors = [w2v_dict[w] for w in text_nlp_proc if w in w2v_dict.vocab.keys()]
    if len(list_of_word_vectors) == 0:
        result = [0.0]*SIZE
    else:
        result = np.sum(list_of_word_vectors, axis=0) / len(list_of_word_vectors)
    return result


def generate_top20_candidates(cosine_sim):
    """
    Get top 20 indices and it's cosine value
    
    :type numpy.ndarray: cosine_sim
    :rtype numpy.ndarray: top20_indices
    :rtype list: top20_cosine
    """
    top20_indices = cosine_sim[0].argsort()[:-21:-1]
    top20_cosine = [cosine_sim[0][i] for i in top20_indices]
    return top20_indices, top20_cosine


def filter_reply_msg(response_df, cosine_cut_off, count_cutoff):
    """
    Filter out responses
    1. cosine similarity < cosine_cut_off
    2. too short replies (word count <= count_cutoff)
    3. replies contain other special characters
    
    :type pandas DataFrame: response_df
    :type float: cosine_cut_off
    :type int: count_cutoff
    :rtype list: filtered_msg
    :rtype list: filtered_msg_cosine
    :rtype int: count
    """   
    filtered_msg = []
    filtered_msg_cosine = []
    count = 0
    for i in range(response_df.shape[0]):
        if count == count_cutoff:
            break
        cosine = response_df.iloc[i]['cosine_sim']
        if cosine < cosine_cut_off:
            break
            
        reply = response_df.iloc[i]['reply_stander_proc']
        # too short
        if len(reply.split()) <= 5:
            continue
        # have special characters
        if len(re.findall(r"[~@#\$%\^&\*\(\)_\+{}\":;\[\]\/]|(\.\.\.)", reply)) != 0:
            continue
        else:
            filtered_msg.append(reply)
            filtered_msg_cosine.append(cosine)
            count += 1
    return filtered_msg, filtered_msg_cosine, count
