import re
import numpy as np
import pandas as pd

# NLP
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def standardize_question_text(text):
    """
    Standardize input question.
    
    :type string: text
    :rtype string: text_proc
    """
    # remove @ tags
    text_proc = re.sub("@\\w+ *", '', text.strip())
    # remove url links
    text_proc = re.sub(r'http.?://[^\s]+[\s]?', '', text_proc)
    # remove the hash tag sign (#) but not the actual tag as this may contain information
    text_proc = re.sub(r'#', '', text_proc)
    # remove the emoji sign (^) but keep the actual converted emojis as one word.
    text_proc = re.sub(r'\^', '', text_proc)
    # strip() to remove whitespace and convert to lower case
    text_proc = text_proc.strip()

    return text_proc


def nlp_preprocess_text(text_proc):
    """
    NLP preprocessing pipeline which uses RegExp, sets basic token requirements, and removes stop words.
    
    :type string: input_proc
    :rtype: string
    """

    # tokenizer, stops, and stemmer
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))  # can add more stop words to this set
    stemmer = SnowballStemmer('english')

    cleaned_tokens = []
    tokens = tokenizer.tokenize(text_proc.lower())
    for token in tokens:
        if token not in stop_words:
            if len(token) > 0 and len(token) < 20: # removes non words
                if not token[0].isdigit() and not token[-1].isdigit(): # removes numbers
                    stemmed_tokens = stemmer.stem(token)
                    cleaned_tokens.append(stemmed_tokens)

    text_nlp_proc = ' '.join(wd for wd in cleaned_tokens)

    return text_nlp_proc