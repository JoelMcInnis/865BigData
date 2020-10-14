Joel McInnis

20191841
GMMA 865
October 14, 2020

Submission to Question 2, Part 1

#import other libraries as necessary
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn import tree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import unidecode
from textblob import TextBlob
import textstat
import string  
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#load data
df_train = pd.read_csv("sentiment_train.csv")
print(df_train.info())
print(df_train.head())

df_test = pd.read_csv("sentiment_test.csv")
print(df_test.info())
print(df_test.head())

#Train data preprocessing
stop_words = set(stopwords.words('english'))

lemmer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\b\d+\b', 'NUM', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    text = [lemmer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(text) 

cleaned_train = lambda x: clean_text(x)
df_clean_train = pd.DataFrame(df_train.Sentence.apply(cleaned_train))
df_clean_train

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_clean_train['subjectivity'] = df_clean_train['Sentence'].apply(sub)
df_clean_train['polar'] = df_clean_train['Sentence'].apply(pol)
df_clean_train['Polarity'] = df_train['Polarity']
df_clean_train['review_len'] = df_clean_train['Sentence'].astype(str).apply(len)
df_clean_train['word_count'] = df_clean_train['Sentence'].apply(lambda x: len(str(x).split()))
df_clean_train['syllable_count'] =  df_clean_train['Sentence'].apply(lambda x: textstat.syllable_count(x))
df_clean_train['lexicon_count'] =  df_clean_train['Sentence'].apply(lambda x: textstat.lexicon_count(x))
df_clean_train['sentence_count'] =  df_clean_train['Sentence'].apply(lambda x: textstat.sentence_count(x))
df_clean_train['flesch_reading_ease'] =  df_clean_train['Sentence'].apply(lambda x: textstat.flesch_reading_ease(x))
df_clean_train['flesch_kincaid_grade'] =  df_clean_train['Sentence'].apply(lambda x: textstat.flesch_kincaid_grade(x))
df_clean_train['gunning_fog'] =  df_clean_train['Sentence'].apply(lambda x: textstat.gunning_fog(x))

df_clean_train.head()

tfidf_vectorizer = TfidfVectorizer(min_df=.01, max_df=.25, ngram_range=[1,3], max_features=1000, stop_words='english')
dtm_tfidf = tfidf_vectorizer.fit_transform(df_clean_train['Sentence'])

bow_df_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=df_clean_train.index)
bow_df_tfidf.shape

df_bow_tfidf = pd.concat([df_clean_train, bow_df_tfidf], axis=1)
df_bow_tfidf.drop(columns=['Sentence'], inplace=True)
df_bow_tfidf.shape
df_bow_tfidf.head()

#Split the training data into test/train
y = df_clean_train['Polarity']
X = df_clean_train.drop(['Polarity','Sentence'], axis=1)

feature_names = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)

##Decision Tree
clf = DecisionTreeClassifier(random_state=42, 
                             min_samples_split=15, 
                             min_samples_leaf=10, 
                             max_depth=4)


clf.fit(X_train, y_train)
y_pred_dt = clf.predict(X_val)
class_names = [str(x) for x in clf.classes_]
imp = clf.tree_.compute_feature_importances(normalize=False)
ind = sorted(range(len(imp)), key=lambda i: imp[i])[-15:]
imp[ind]
feature_names[ind]
confusion_matrix(y_val, y_pred_dt)
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred_dt))
print(classification_report(y_val, y_pred_dt, target_names=class_names))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_dt, average='micro')))


##Random Forest
print("\n\nRF")
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42)
rf_clf.fit(X_train, y_train)
imp_rf = pd.DataFrame(rf_clf.feature_importances_, index = feature_names, columns=['importance']).sort_values('importance', ascending=False).iloc[0:15,:]
print(imp_rf)
y_pred_rf = rf_clf.predict(X_val)
print(accuracy_score(y_val, y_pred_rf))
print(confusion_matrix(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_rf, average='micro')))


#KNN
print("\n\nKNN")
knn_clf = KNeighborsClassifier(n_neighbors=15)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_val)
print(accuracy_score(y_val, y_pred_knn))
print(confusion_matrix(y_val, y_pred_knn))
print(classification_report(y_val, y_pred_knn))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_knn, average='micro')))


##NB
print("NB")
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
y_pred_nb = gnb.predict(X_val)
print(accuracy_score(y_val, y_pred_nb))
print(confusion_matrix(y_val, y_pred_nb))
print(classification_report(y_val, y_pred_nb))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_nb, average='micro')))


#GradientBoost
print("\n\nGTB")
grb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=42)
grb_clf.fit(X_train, y_train)
y_pred_grb = grb_clf.predict(X_val)
print(accuracy_score(y_val, y_pred_grb))
print(confusion_matrix(y_val, y_pred_grb))
print(classification_report(y_val, y_pred_grb))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_grb, average='micro')))

#####################################################
#Clean Test Data
cleaned_test = lambda x: clean_text(x)
df_clean_test = pd.DataFrame(df_test.Sentence.apply(cleaned_test))
df_clean_test

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity


df_clean_test['subjectivity'] = df_clean_test['Sentence'].apply(sub)
df_clean_test['polar'] = df_clean_test['Sentence'].apply(pol)
df_clean_test['Polarity'] = df_test['Polarity']
df_clean_test['review_len'] = df_clean_test['Sentence'].astype(str).apply(len)
df_clean_test['word_count'] = df_clean_test['Sentence'].apply(lambda x: len(str(x).split()))
df_clean_test['syllable_count'] =  df_clean_test['Sentence'].apply(lambda x: textstat.syllable_count(x))
df_clean_test['lexicon_count'] =  df_clean_test['Sentence'].apply(lambda x: textstat.lexicon_count(x))
df_clean_test['sentence_count'] =  df_clean_test['Sentence'].apply(lambda x: textstat.sentence_count(x))
df_clean_test['flesch_reading_ease'] =  df_clean_test['Sentence'].apply(lambda x: textstat.flesch_reading_ease(x))
df_clean_test['flesch_kincaid_grade'] =  df_clean_test['Sentence'].apply(lambda x: textstat.flesch_kincaid_grade(x))
df_clean_test['gunning_fog'] =  df_clean_test['Sentence'].apply(lambda x: textstat.gunning_fog(x))

df_clean_test.head()

test_y = df_clean_test['Polarity']
test_X = df_clean_test.drop(['Polarity','Sentence'], axis=1)

test_X = scaler.fit_transform(test_X)


#Predict using Test Data

#Decision Tree
test_pred_dt = clf.predict(test_X)
print(accuracy_score(test_y, test_pred_dt ))
print(confusion_matrix(test_y, test_pred_dt ))
print(classification_report(test_y, test_pred_dt ))
print("\nF1 Score = {:.5f}".format(f1_score(test_y, test_pred_dt, average="micro" )))

#Random Forest
test_pred_rf = rf_clf.predict(test_X)
print(accuracy_score(test_y, test_pred_rf ))
print(confusion_matrix(test_y, test_pred_rf ))
print(classification_report(test_y, test_pred_rf ))
print("\nF1 Score = {:.5f}".format(f1_score(test_y, test_pred_rf, average="micro" )))



## Review the incorrect predictions
df_test['test_pred_dt']  = test_pred_dt
df_clean_test['test_pred_dt']  = test_pred_dt

correct_df_clean = df_clean_test
correct_df = df_test
correct_df_clean['Correct_Predict'] = np.where(correct_df_clean['test_pred_dt']==correct_df_clean['Polarity'], 'yes', 'no')
show_df_clean = correct_df_clean[correct_df_clean['Correct_Predict']== 'no']
correct_df['Correct_Predict'] = np.where(correct_df['test_pred_dt']==correct_df['Polarity'], 'yes', 'no')
show_df = correct_df[correct_df['Correct_Predict']== 'no']


show_df
show_df_clean

#show_df.to_csv(r'C:\Users\mcinn\Desktop\Sentence_wordcount.csv', index = False)
#show_df_clean.to_csv(r'C:\Users\mcinn\Desktop\Sentence_wrong1.csv', index = False)
#correct_df_clean.to_csv(r'C:\Users\mcinn\Desktop\Sentence_all.csv', index = False)
