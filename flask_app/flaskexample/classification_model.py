import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from flaskexample.preprocess import standardize_question_text, nlp_preprocess_text



def generate_tags(input_question, tfidf_vectorizer, clf_tfidf):
    """
    Generate Top3 tags
    
    :type string: input_question
    :type sklearn.feature_extraction.text.TfidfVectorizer: tfidf_vectorizer
    :type sklearn.linear_model.logistic.LogisticRegression: clf_tfidf
    :rtype list: tag_list
    :rtype list: prob_list
    """ 
    # preprocess input question 
    input_question_proc = standardize_question_text(input_question)
    input_question_nlp_proc = nlp_preprocess_text(input_question_proc)

    # generate input tf-idf matrix
    input_tfidf = tfidf_vectorizer.transform([input_question_nlp_proc])

    # generate result
    tag_probs = clf_tfidf.predict_log_proba(input_tfidf)
    order_list = tag_probs[0].argsort()[:-4:-1] # top 3
    tag_list = [clf_tfidf.classes_[i] for i in order_list]
    prob_list = [tag_probs[0][i] for i in order_list]

    return tag_list, prob_list