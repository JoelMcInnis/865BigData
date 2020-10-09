[First name, Last name]

[Student number]
[Program]
[Cohort]
[Course Number]
[Date]


Submission to Question [X], Part [X]


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import stop_words



# TODO: import other libraries as necessary
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF
from sklearn.neural_network import MLPClassifier


df_train = pd.read_csv("sentiment_train.csv")

print(df_train.info())
print(df_train.head())

df_test = pd.read_csv("sentiment_test.csv")

print(df_test.info())
print(df_test.head())


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


    

vectorizer = TfidfVectorizer(preprocessor=clean_text, 
                             max_features = 1000, 
                             ngram_range=[1,4],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.25, min_df=0.001, use_idf=True)




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

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=.02, max_df=.5, ngram_range=[1,3], max_features=500, stop_words='english')
dtm_tfidf = tfidf_vectorizer.fit_transform(df_clean_train['Sentence'])

bow_df_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=df_clean_train.index)
bow_df_tfidf.shape

df_bow_tfidf = pd.concat([df_clean_train, bow_df_tfidf], axis=1)
df_bow_tfidf.drop(columns=['Sentence'], inplace=True)
df_bow_tfidf.shape
df_bow_tfidf.head()

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

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

from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, y_pred_dt)
from sklearn.metrics import classification_report

print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred_dt))

print(classification_report(y_val, y_pred_dt, target_names=class_names))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_dt, average='micro')))


##Random Forest
print("\n\nRF")
clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=55)
clf.fit(X_train, y_train)

imp = pd.DataFrame(clf.feature_importances_, index = feature_names, columns=['importance']).sort_values('importance', ascending=False).iloc[0:15,:]
print(imp)

y_pred = clf.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_dt, average='micro')))


#KNN
print("\n\nKNN")
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=15)
knn_clf.fit(X_train, y_train)

y_pred_knn = knn_clf.predict(X_val)
print(accuracy_score(y_val, y_pred_knn))
print(confusion_matrix(y_val, y_pred_knn))
print(classification_report(y_val, y_pred_knn))
print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_dt, average='micro')))

from sklearn.naive_bayes import GaussianNB

print("NB")

gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

y_pred_nb = gnb.predict(X_val)
print(accuracy_score(y_val, y_pred_nb))
print(confusion_matrix(y_val, y_pred_nb))
print(classification_report(y_val, y_pred_nb))

gnb.theta_