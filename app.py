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
from flask_cors import CORS


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
CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
db=SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
api=Api(app)




class SentimentModel(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    text=db.Column(db.String(1000),nullable=False)
    pre=db.Column(db.String(1000),nullable=False)
    prob=db.Column(db.String(100),nullable=False)
    purport=db.Column(db.String(1000),nullable=False)
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
    "purport":fields.String,
    "like":fields.Float,
    "love":fields.Float,
    "haha":fields.Float,
    "wow":fields.Float,
    "sad":fields.Float,
    "angry":fields.Float,
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
        p_date = datetime.now()
        if y_pred[0] == 'like' and prob == 'love':
            purport = 'Generally wholesome post'
        if y_pred[0] == 'like' and prob == 'haha':
            purport = 'Generally funny me-me'
        if y_pred[0] == 'like' and prob == 'wow':
            purport = 'Fun fact'
        if y_pred[0] == 'like' and prob == 'sad':
            purport = 'Sad, somewhat true story from generic relatable page'
        if y_pred[0] == 'like' and prob == 'angry':
            purport = 'Some controversial statement by a personality'
        if y_pred[0] == 'love' and prob == 'like':
            purport = 'Very acceptable'
        if y_pred[0] == 'love' and prob == 'haha':
            purport = 'Funny post'
        if y_pred[0] == 'love' and prob == 'wow':
            purport = 'Absurdist post'
        if y_pred[0] == 'love' and prob == 'sad':
            purport = 'Aww that is cute and quite ironic with our society post'
        if y_pred[0] == 'love' and prob == 'angry':
            purport = 'Really, really, really bad post'
        if y_pred[0] == 'haha' and prob == 'like':
            purport = 'Funny post haha from a generic post'
        if y_pred[0] == 'haha' and prob == 'love':
            purport = 'Really funny post and everyone agrees that it is funny'
        if y_pred[0] == 'haha' and prob == 'wow':
            purport = 'Very obvious post/fact'
        if y_pred[0] == 'haha' and prob == 'sad':
            purport = 'post that some people feel bad to laugh at'
        if y_pred[0] == 'haha' and prob == 'angry':
            purport = 'Dark post/bad post'
        if y_pred[0] == 'wow' and prob == 'like':
            purport = 'Surprising or Unknown fact'
        if y_pred[0] == 'wow' and prob == 'love':
            purport = 'wholesome post'
        if y_pred[0] == 'wow' and prob == 'haha':
            purport = 'probably memeing'
        if y_pred[0] == 'wow' and prob == 'sad':
            purport = 'Bad news'
        if y_pred[0] == 'wow' and prob == 'angry':
            purport = 'Doing fake thing'
        if y_pred[0] == 'sad' and prob == 'like':
            purport = 'Sad story'
        if y_pred[0] == 'sad' and prob == 'love':
            purport = 'Aww that is sad'
        if y_pred[0] == 'sad' and prob == 'haha':
            purport = 'When a person got a bad thing because of his own stupidity'
        if y_pred[0] == 'sad' and prob == 'wow':
            purport = 'Sad, shocking'
        if y_pred[0] == 'sad' and prob == 'angry':
            purport = 'A real drama'
        if y_pred[0] == 'angry' and prob == 'like':
            purport = 'angry but not much'
        if y_pred[0] == 'angry' and prob == 'love':
            purport = 'bad post but some one love'
        if y_pred[0] == 'angry' and prob == 'haha':
            purport = 'uncomfortable post'
        if y_pred[0] == 'angry' and prob == 'wow':
            purport = 'angry post'
        if y_pred[0] == 'angry' and prob == 'sad':
            purport = 'disgusting post'

        if like == love and like == wow and like == haha and like == sad and like == angry:
            y_pred[0] = 'unpredict'
            prob = 'unpredict'
            purport = "Can't predict this text"


        sen = SentimentModel(text=texts[0],pre=y_pred[0], prob=prob,purport=purport,like=like,love=love,haha=haha,wow=wow,sad=sad,angry=angry,p_date=p_date)
        db.session.add(sen)
        db.session.commit()
        return sen,201

class LogResource(Resource):
    @marshal_with(resource_field)
    def get(self,id):
        result = SentimentModel.query.filter_by(id = id).first()
        return result



api.add_resource(Sentiment,'/sentiment')
api.add_resource(LogResource,'/sentiment/history/<int:id>')




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