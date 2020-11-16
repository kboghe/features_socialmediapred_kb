########
#import#
########

#general packages#
import re
import time
import random
from datetime import datetime, timedelta
#data manipulation#
import numpy as np
import pandas as pd
#translation,sentiment,morality
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from moralizer import *
#named entity recognition and cleaning text#
import spacy
import preprocessor as p
#for perspective API#
import json
import requests
#Establishing connection with database
import psycopg2
#Options for preprocessing text, loading Dutch spacy model for named entity recognition
#and loading English spacy model for perspective API (which uses spacy in the background)
p.set_options(p.OPT.URL,p.OPT.MENTION,p.OPT.SMILEY,p.OPT.RESERVED)
nlp = spacy.load('nl_core_news_sm')
spacy.load('en_core_web_sm')

#################
#write functions#
#################

#1.Hot topic detection
def connect_todb(info):
    #this function expects a csv file with 5 columns (database,user,password,host,port)...
    #...with a single row (besides the header) containing the login info
    info = pd.read_csv(info, sep=",", encoding="utf-8")
    print("\nConnecting to database...\n"
          "========================\n")
    con = psycopg2.connect(database=info.database[0], user=info.user[0], password=info.password[0], host=info.host[0], port=info.port[0])
    print("Done!\n")
    return con
def get_hottopic(text,connection,date,country):
    if date is None:
        date = str(datetime.today().strftime('%Y-%m-%d'))
    date2 = str((datetime.strptime(date,'%Y-%m-%d') - timedelta(random.randrange(7,14))).date())
    cur = connection.cursor()
    headlines_samples = dict()
    for period,key in zip([date,date2],['target_day','random_day']):
        cur.execute('''SELECT *
            FROM(SELECT *
            FROM headlines
            WHERE
            date(date) BETWEEN '{0}' AND '{1}') as headlines_today
            INNER
            JOIN
            outlets
            ON(headlines_today.account = outlets.account) 
            WHERE country = '{2}';;'''.format(period,str((datetime.strptime(period,'%Y-%m-%d') + timedelta(1)).date()),country.lower()))
        headlines_samples[key] = cur.fetchall()
        connection.commit()

    compare_ner = pd.DataFrame()
    for key,value in headlines_samples.items():
        ner_generallist = []
        engagement_generallist = []
        for headline in value:
            headline_list = set([re.sub("{|}|#|vtm|vrt|nieuws","", headline[x]) for x in list(range(11,18))]) #remove the tuple markers, hashtags and other meaningless tokens, only keep unique NERs present in a single text
            ners_headline = list(filter(None,[re.sub('\"|\'','',x) for x in list(headline_list)])) #remove quotations around NERs
            ners_headline = [x.split(',') for x in ners_headline] #some NERs contain lists. Split these.
            ners_headline = list(filter(None,(sum(ners_headline,[]))))
            ners_headline_engagement = [sum(headline[7:10])] * len(ners_headline)
            ner_generallist.append(ners_headline)
            engagement_generallist.append(ners_headline_engagement)
        ner_generallist = list(filter(None,sum(ner_generallist, []))) #sum all the lists to one single list
        engagement_generallist = list(sum(engagement_generallist, []))
        ner_totalengagement = pd.DataFrame(engagement_generallist,ner_generallist).groupby(level=0).sum()
        compare_ner[key+'_n'] = pd.Series(ner_generallist).value_counts()
        compare_ner[key+'_n'] = compare_ner[key+'_n'].fillna(0)
        compare_ner[key+'_engagement'] = pd.Series(ner_totalengagement.iloc[:,0])
        compare_ner[key + '_engagement'] = compare_ner[key + '_engagement'].fillna(0)
        ner_generallist_prop = pd.Series(ner_generallist).value_counts()/len(headlines_samples[key])
        compare_ner[key+'_prop'] = ner_generallist_prop
    compare_ner['random_day_prop'] = compare_ner.random_day_prop.fillna(0.001)
    compare_ner['comparison_metric'] = compare_ner.target_day_prop / compare_ner.random_day_prop #comparison metric
    random_day_engagement_fillna =  compare_ner.random_day_engagement.fillna(1)
    random_day_engagement_fillna = np.where(random_day_engagement_fillna < 1,1,random_day_engagement_fillna)
    weights = compare_ner.target_day_engagement / random_day_engagement_fillna
    compare_ner['comparison_metric_weighted'] =  compare_ner.comparison_metric * np.sqrt(weights) #comparison metric wieghted for engagement
    compare_ner = compare_ner.sort_values('comparison_metric_weighted',ascending=False)
    checklen = [len(x) > 1 for x in compare_ner.index]
    compare_ner = compare_ner[checklen]  # remove single letter entities (tend to be meaningless)
    compare_ner['hot_topic'] = sum([[1] * 10] + [[0]*int(len(compare_ner)-10)],[])

    ner_cleaned = []
    text = p.clean(text)
    entities = [nlp(text)] #detect entities in html escaped text (so & becomes &amp etc)
    for doc in entities:
        ner_cleaned.append([ent.lemma_ for ent in doc.ents])
    compare_ner_match = [item in sum(ner_cleaned,[]) for item in compare_ner.index]
    compare_ner_filtered = compare_ner[compare_ner_match]
    compare_ner_metrics = dict(compare_ner_filtered.agg({'target_day_n':'max','comparison_metric':'max',
                                                    'comparison_metric_weighted':'max','hot_topic':'max'}))
    return compare_ner_metrics

#2.Textual proporties: alternative sentiment measure,morality and toxicity#
def translate(text):
    translator = Translator()
    while True:
        try:
            transl_text = translator.translate(text,dest='en')
            break
        except Exception: #sometimes returns an error at random (see Github issues), but a simple retry tends to solve it!
            translator = Translator()
    return transl_text.text
def sentiment(text):
    sent_vader = list(SentimentIntensityAnalyzer().polarity_scores(text).values())
    return sent_vader[3]
def morality(text):
    morality_scores = dict()
    moral_scores = moralize(text.lower())
    for key,value in moral_scores.items():
        count_moral = 0
        for key_spec,value_spec in value.items():
            for key_count,value_count in value_spec.items():
                count_moral += value_count
        morality_scores[key] = count_moral
    return morality_scores
def toxic(text):
    toxicity_score = dict()
    api_key = open("perspective_api_key.txt", "r").read()
    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
           '?key=' + api_key)
    time.sleep(1)
    data_dict = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'SEVERE_TOXICITY': {},'INSULT':{},'PROFANITY':{}}
    }
    toxic_resp = requests.post(url=url, data=json.dumps(data_dict))
    toxic_dict = json.loads(toxic_resp.content)
    for key,value in toxic_dict['attributeScores'].items():
        toxicity_score[key] = value['summaryScore']['value']
    return toxicity_score

######
#DEMO#
######

#a couple of examples
demotexts = pd.read_csv('demotexts.csv',sep=";",quotechar="'",encoding="utf-8")

#establish connection with headlines database (necessary for hot topic detection)
con = connect_todb('connection_db.csv')

for index,row in demotexts.iterrows():
    print("\nText #"+str(index+1)+":\n############\n")
    print("Text to analyze: " + row['text'] + '\nDate: ' + row['date'])
    print('\nHot topic metrics:\n-----------------')
    print(get_hottopic(text=row['text'],connection=con,date=row['date'],country=row['country']))
    text_translated = translate(row['text'])
    print('\nVader sentiment metric:\n-----------------')
    print(sentiment(text_translated))
    print('\nMorality metrics:\n-----------------')
    print(morality(text_translated))
    print('\nToxicity metrics:\n-----------------')
    print(toxic(text_translated))