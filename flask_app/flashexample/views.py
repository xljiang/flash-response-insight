from flaskexample.w2v_model import generate_responses
from flaskexample.classification_model import generate_tags
from flask import request
from flask import render_template
from flask import json
from flask import jsonify
from flaskexample import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import json
import pickle

'''
# Connect to Postgres
user = 'xiaolu'     
host = 'localhost'
dbname = 'airline_train_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)
'''

# Loading the saved w2v model from disk
w2v_model_pkl = open('./flaskexample/file/w2v_model.pkl', 'rb')
w2v_model = pickle.load(w2v_model_pkl)

# Loading the saved train average w2v data frame
train_w2v_df_pkl = open('./flaskexample/file/train_avg_w2v_df.pkl', 'rb')
train_w2v_df = pickle.load(train_w2v_df_pkl)

# Loading the saved train data frame
train_df_pkl = open('./flaskexample/file/train_reply_df.pkl', 'rb')
train_df = pickle.load(train_df_pkl)

# Loading the saved tf-idf logistic regression model
clf_tfidf_pkl = open('./flaskexample/file/clf_tfidf.pkl', 'rb')
clf_tfidf = pickle.load(clf_tfidf_pkl)

# Loading the saved tf-idf vectorizer
tfidf_vectorizer_pkl = open('./flaskexample/file/tfidf_vectorizer.pkl', 'rb')
tfidf_vectorizer = pickle.load(tfidf_vectorizer_pkl)


@app.route('/')


@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/tag_link')
def tag_link():
    return render_template("tag_link.html")

@app.route('/generate_response',  methods=['POST'])
def generate_response():
    customer_question = request.form['customer_question']
    # generate response by predefined QA
    count_qa = 0
    
    # generate response based on corpus search (ir: information retrival)
    reply_text_list, reply_cosine_list, count_ir = generate_responses(customer_question, w2v_model, train_w2v_df, train_df, 5-count_qa)
    
    count = count_qa + count_ir

    # generate top 3 topic list
    tag_list, prob_list = generate_tags(customer_question, tfidf_vectorizer, clf_tfidf)


    '''
    # generate result from database
    agent_replies = []
    for i in range(len(result_indices)):
        idx = result_indices[i]
        query = "SELECT reply_stander_proc FROM airline_data_table WHERE iloc=%d" % idx
        query_results_df=pd.read_sql_query(query,con)
        reply = query_results_df['reply_stander_proc'].values[0]
        agent_replies.append(reply)
    '''   

    # generate result
    agent_replies = []
    for i in range(count_ir):
        reply = {'reply_text' : reply_text_list[i], 'reply_cosine': reply_cosine_list[i]} 
        agent_replies.append(reply)   

    question_tags = []
    for i in range(len(tag_list)):
        tag = {'tag': tag_list[i], 'probability': prob_list[i]}
        question_tags.append(tag)  

    ret = {
        'customer_question': customer_question,
        'agent_replies': agent_replies,
        'count': count,
        'question_tags': question_tags
    }

    return jsonify(ret)

    