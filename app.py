from distutils.log import debug
from flask import Flask
from flask_restful import Api,Resource,abort,reqparse,marshal_with,fields
from flask_sqlalchemy import SQLAlchemy,Model
import pickle
import re
import csv
import re
import string
import emoji
import numpy as np
import pandas as pd
import pickle
from datetime import date, datetime

from pythainlp import word_tokenize
from tqdm import tqdm_notebook
from pythainlp.ulmfit import process_thai
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords
from nltk.stem import PorterStemmer

#viz
import matplotlib.pyplot as plt

from pythainlp.ulmfit import *


app=Flask(__name__)
db=SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
api=Api(app)




class SentimentModel(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    text=db.Column(db.String(1000),nullable=False)
    pre=db.Column(db.String(1000),nullable=False)
    prob=db.Column(db.String(100),nullable=False)
    like=db.Column(db.Float,nullable=False)
    love=db.Column(db.Float,nullable=False)
    haha=db.Column(db.Float,nullable=False)
    wow=db.Column(db.Float,nullable=False)
    sad=db.Column(db.Float,nullable=False)
    angry=db.Column(db.Float,nullable=False)
    p_date=db.Column(db.String(100),nullable=False)
    def __repr__(self):
        return f"create"

db.create_all()

predict_add_args=reqparse.RequestParser()
predict_add_args.add_argument("text",type=str)
# predict_add_args.add_argument("pre",type=str)
# predict_add_args.add_argument("prob",type=str)
# predict_add_args.add_argument("like",type=str)
# predict_add_args.add_argument("love",type=str)
# predict_add_args.add_argument("haha",type=str)
# predict_add_args.add_argument("wow",type=str)
# predict_add_args.add_argument("sad",type=str)
# predict_add_args.add_argument("angry",type=str)
# predict_add_args.add_argument("date",type=datetime)

resource_field={
    "id":fields.Integer,
    "text":fields.String,
    "pre":fields.String,
    "prob":fields.String,
    "like":fields.String,
    "love":fields.String,
    "haha":fields.String,
    "wow":fields.String,
    "sad":fields.String,
    "angry":fields.String,
    "p_date":fields.String,
}

class Sentiment(Resource):
    @marshal_with(resource_field)
    def get(self):
        result = SentimentModel.query.all()
        return result

    @marshal_with(resource_field)
    def post(self):
        args=predict_add_args.parse_args()
        loaded_tfidf = pickle.load(open('tfidf_fit.pkl', 'rb'))
        loaded_model = pickle.load(open('model-0.763.pkl', 'rb'))
        texts = [args["text"]]
        text = loaded_tfidf.transform(texts)
        X_test = text.toarray()
        y_pred = loaded_model.predict(X_test)
        probs = loaded_model.predict_proba(X_test)
        probs_df = pd.DataFrame(probs)
        probs_df.columns = loaded_model.classes_
        like = probs_df['like'][0]
        love = probs_df['love'][0]
        haha = probs_df['haha'][0]
        wow = probs_df['wow'][0]
        sad = probs_df['sad'][0]
        angry = probs_df['angry'][0]
        df1 = probs_df.apply(np.argsort, axis=1)
        cdf = df1[['sad']]
        cdf = cdf.replace(to_replace = 0 ,
                 value ="angry")
        cdf = cdf.replace(to_replace = 1 ,
                 value ="haha")
        cdf = cdf.replace(to_replace = 2 ,
                 value ="like")
        cdf = cdf.replace(to_replace = 3 ,
                 value ="love")
        cdf = cdf.replace(to_replace = 4 ,
                 value ="sad")
        cdf = cdf.replace(to_replace = 5 ,
                 value ="wow")
        prob = cdf['sad'][0]
        print(datetime.now())
        p_date = datetime.now()




        sen = SentimentModel(text=texts[0],pre=y_pred[0], prob=prob,like=like,love=love,haha=haha,wow=wow,sad=sad,angry=angry,p_date=p_date)
        db.session.add(sen)
        db.session.commit()
        return sen,201


api.add_resource(Sentiment,'/sentiment')




if __name__ == "__main__":
    def clean_msg(msg):
            # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
        msg = re.sub(r'<.*?>','', msg)

            # ลบ hashtag
        msg = re.sub(r'#','',msg)
    
        msg = re.sub(r'>>','',msg)
    

            # ลบ เครื่องหมายคำพูด (punctuation)
        for c in string.punctuation:
            msg = re.sub(r'\{}'.format(c),'',msg)

            # ลบ separator เช่น \n \t
        msg = ' '.join(msg.split())

            # Delete emoji
        regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        
                           "]+", flags = re.UNICODE)
        msg = regex_pattern.sub(r'', msg)
        msg = ''.join([c for c in msg if c not in emoji.UNICODE_EMOJI])

            # r = re.compile('[a-zA-Z]')
            # msg = re.sub(r'[a-zA-Z]','', msg)

            # stopwords = list(thai_stopwords())
            # msg = [i for i in msg if i not in stopwords]
        ps = PorterStemmer()
        msg = ps.stem(normalize(msg))
        return msg
    def stop_words(t): 
        stopwords = list(thai_stopwords())
        list_word = process_thai(t)
        list_word_not_stopwords = [i for i in list_word if i not in stopwords]
        t = list_word_not_stopwords
        return t
    app.run(debug=True) 