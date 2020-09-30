# -*- coding: utf-8 -*-
"""
[First name, Last name]
[Student number]
[Program]
[Cohort]
[Course Number]
[Date]


Submission to Question [X], Part [X]
"""

import pandas as pd
# TODO: import other libraries as necessary
import nltk
import re
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.corpus import stopwords 

df_train = pd.read_csv("sentiment_train.csv")

print(df_train.info())
print(df_train.head())

df_test = pd.read_csv("sentiment_test.csv")

print(df_test.info())
print(df_test.head())

# TODO: insert code here to perform the given task. 
# Don't forget to document your code!


#data cleaning TRAIN


#Document Term Matrix
from sklearn.feature_extraction.text import CountVectorizer

# Stopword removal  
stop_words = set(stopwords.words('english')) 
for i, line in enumerate(data_clean.Sentence): 
    data_clean.Sentence[i] = ' '.join([x for 
        x in nltk.word_tokenize(line) if 
        ( x not in stop_words ) and ( x not in your_list )]) 



# Applying TFIDF 
vectorizer = TfidfVectorizer(ngram_range = (3,3)) 
X2 = vectorizer.fit_transform(data_clean.Sentence) 
scores = (X2.toarray()) 
print("\n\nScores : \n", scores) 

sums = X2.sum(axis = 0) 
data1 = [] 
for col, term in enumerate(features): 
    data1.append( (term, sums[0,col] )) 
ranking = pd.DataFrame(data1, columns = ['term','rank']) 
words = (ranking.sort_values('rank', ascending = False)) 
print ("\n\nWords head : \n", words.head(7)) 

##Document-Term Matrix
#cv = CountVectorizer(stop_words='english')
#data_cv = cv.fit_transform(data_clean.transcript)
#data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
#data_dtm.index = data_clean.index
#data_dtm



#######################################
# TRAIN CLEANING AND SENTIMATE ANALYSIS
############
a = 0 
for i in range(a,a+4):
    print(df_train['Sentence'][i])
    print()

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1_train = lambda x: clean_text_round1(x)
df_clean_train = pd.DataFrame(df_train.Sentence.apply(round1_train))
df_clean_train


# Stopword removal  
stop_words = set(stopwords.words('english')) 
for i, line in enumerate(df_clean_train.Sentence): 
    df_clean_train.Sentence[i] = ' '.join([x for 
        x in nltk.word_tokenize(line) if 
        ( x not in stop_words ) and ( x not in your_list )]) 
df_clean_train

# Applying TFIDF 
vectorizer = TfidfVectorizer(ngram_range = (3,3)) 
X2 = vectorizer.fit_transform(df_clean_train.Sentence) 
scores = (X2.toarray()) 
print("\n\nScores : \n", scores) 

sums = X2.sum(axis = 0) 
data1 = [] 
for col, term in enumerate(features): 
    data1.append( (term, sums[0,col] )) 
ranking = pd.DataFrame(data1, columns = ['term','rank']) 
words = (ranking.sort_values('rank', ascending = False)) 
print ("\n\nWords head : \n", words.head(7)) 


#Sentiment Analysis 

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_clean_train['polarity'] = df_clean_train['Sentence'].apply(pol)
df_clean_train['subjectivity'] = df_clean_train['Sentence'].apply(sub)
df_clean_train['review_len'] = df_clean_train['Sentence'].astype(str).apply(len)
df_clean_train['word_count'] = df_clean_train['Sentence'].apply(lambda x: len(str(x).split()))
df_clean_train

print('5 random reviews with the highest positive sentiment polarity: \n')
cl = df_clean_train.loc[df_clean_train.polarity == 1, ['Sentence']].sample(5).values
for c in cl:
    print(c[0])

#positive 
df_clean_train.loc[df_clean_train.polarity == 1]
#negative
df_clean_train.loc[df_clean_train.polarity == -1]
#neutral
df_clean_train.loc[df_clean_train.polarity == 0]

#######################################
# TEST CLEANING AND SENTIMATE ANALYSIS
############
a = 0 
for i in range(a,a+4):
    print(df_test['Sentence'][i])
    print()

def clean_text_round2(text1):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text1 = text1.lower()
    #text1 = re.sub('\[.*?\]', '', text1)
   # text1 = re.sub('[%s]' % re.escape(string.punctuation), '', text1)
   # text1 = re.sub('\w*\d\w*', '', text1)
    return text1

round2_test = lambda x: clean_text_round2(x)
df_clean_test = pd.DataFrame(df_test.Sentence.apply(round2_test))
df_clean_test

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_clean_test['polarity'] = df_clean_test['Sentence'].apply(pol)
df_clean_test['subjectivity'] = df_clean_test['Sentence'].apply(sub)
df_clean_test['review_len'] = df_clean_test['Sentence'].astype(str).apply(len)
df_clean_test['word_count'] = df_clean_test'Sentence'].apply(lambda x: len(str(x).split()))
df_clean_test

print('5 random reviews with the highest positive sentiment polarity: \n')
cl = df_clean_train.loc[df_clean_train.polarity == 1, ['Sentence']].sample(5).values
for c in cl:
    print(c[0])

#positive 
df_clean_test.loc[df_clean_test.polarity == 1]
#negative
df_clean_test.loc[df_clean_test.polarity == -1]
#neutral
df_clean_test.loc[df_clean_test.polarity == 0]

##EDA on TRAIN

    #TOP WORDS
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df_clean_train['Sentence'], 30)
for word, freq in common_words:
    print(word, freq)
df_topwords = pd.DataFrame(common_words, columns = ['Sentence' , 'count'])

#TOP BIGRAMS
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df_clean_train['Sentence'], 20)
for word, freq in common_words:
    print(word, freq)
df_bigram = pd.DataFrame(common_words, columns = ['Sentence' , 'count'])

#TOP TRIGRAMS
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df_clean_train['Sentence'], 20)
for word, freq in common_words:
    print(word, freq)
df_trigram = pd.DataFrame(common_words, columns = ['Sentence' , 'count'])


#SVM MODEL
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.metrics import f1_score, classification_report, accuracy_score

#df_clean_train['polarity'] = df_clean_train['polarity'].astype(int).astype('category')
train_posts = df_clean_train['Sentence']
train_tags = df_clean_train['polarity']
test_posts = df_clean_test['Sentence']
test_tags = df_clean_test['polarity']


svm_C = make_pipeline(CountVectorizer(ngram_range=(1,2)),SGDClassifier(loss='hinge', penalty='l2', alpha=0.001,  random_state=42), ).fit(train_posts, train_tags)
svm_prediction = svm_C.predict(test_posts)
SVM_score_train = f1_score(train_tags, svm_C.predict(train_posts)
SVM_score_test = f1_score(test_tags, svm_C.predict(test_posts)
print('SVM_score_f1(test):{}, SVM_score_f1(train):{}'.format(SVM_score_test, SVM_score_train))

#LR MODEL
from sklearn.linear_model import LogisticRegression
lr1 = make_pipeline(CountVectorizer(ngram_range=(1,2)), LogisticRegression(), ).fit(train_posts, train_tags)
lr1_prediction = lr1.predict(test_posts)
lr1_score_train = f1_score(train_tags, lr1.predict(train_posts))
lr1_score_test = f1_score(test_tags, lr1.predict(test_posts))
print ('lr1_score_f1(test):{} --- lr1_score_f1(train):{}'.format(lr1_score_test, lr1_score_train))

