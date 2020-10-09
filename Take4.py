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

X, y = df_train.Sentence, df_train.Polarity
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)


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
    

vectorizer = TfidfVectorizer(preprocessor=clean_text, 
                             max_features = 1000, 
                             ngram_range=[1,4],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.25, min_df=0.001, use_idf=True)




pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity


df_train['subjectivity'] = df_train['Sentence'].apply(sub)
df_train['review_len'] = df_train['Sentence'].astype(str).apply(len)
df_train['word_count'] = df_train['Sentence'].apply(lambda x: len(str(x).split()))
df_train['syllable_count'] =  df_train['Sentence'].apply(lambda x: textstat.syllable_count(x))
df_train['lexicon_count'] =  df_train['Sentence'].apply(lambda x: textstat.lexicon_count(x))
df_train['sentence_count'] =  df_train['Sentence'].apply(lambda x: textstat.sentence_count(x))
df_train['flesch_reading_ease'] =  df_train['Sentence'].apply(lambda x: textstat.flesch_reading_ease(x))
df_train['flesch_kincaid_grade'] =  df_train['Sentence'].apply(lambda x: textstat.flesch_kincaid_grade(x))
df_train['gunning_fog'] =  df_train['Sentence'].apply(lambda x: textstat.gunning_fog(x))

df_train.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=.02, max_df=.5, ngram_range=[1,3], max_features=500, stop_words='english')
dtm_tfidf = tfidf_vectorizer.fit_transform(df_train['Sentence'])

bow_df_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=df_train.index)
bow_df_tfidf.shape

kiva_df_bow_tfidf = pd.concat([df_train, bow_df_tfidf], axis=1)
kiva_df_bow_tfidf.drop(columns=['Sentence'], inplace=True)
kiva_df_bow_tfidf.shape
kiva_df_bow_tfidf.head()






clf = DecisionTreeClassifier(random_state=42, 
                             min_samples_split=10, 
                             min_samples_leaf=10, 
                             max_depth=6)

dt=clf.fit(X_train, y_train)

y_pred_dt = clf.predict(X_val)










#
#nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
#rf = RandomForestClassifier(criterion='entropy', random_state=223)
#mlp = MLPClassifier(random_state=42, verbose=2, max_iter=200)
#
#
#
#
#rf = RandomForestClassifier(criterion='entropy', random_state=223)
#
#
#def lexicon_count(corpus):
#    return np.array([textstat.lexicon_count(doc) for doc in corpus]).reshape(-1, 1)
#
#def punc_count(corpus):
#    return np.array([_get_punc(doc) for doc in corpus]).reshape(-1, 1)
#
#feature_processing =  FeatureUnion([ 
#    ('bow', Pipeline([('cv', vectorizer), ])),
#    ('words', FunctionTransformer(lexicon_count, validate=False)),
#    ('punc_count', FunctionTransformer(punc_count, validate=False)),
#])
#
#steps = [('features', feature_processing)]
#
#pipe = Pipeline([('features', feature_processing), ('clf', rf)])
#
#param_grid = {}
#
#which_clf = "RF"
#
#if which_clf == "RF":
#
#    steps.append(('clf', rf))
#
#    param_grid = {
#        'features__bow__cv__preprocessor': [None, clean_text],
#        'features__bow__cv__max_features': [200, 500, 1000],
#        'features__bow__cv__use_idf': [False],
#        'features__topics__cv__stop_words': [None],
#        'features__topics__nmf__n_components': [25, 75],
#        'clf__n_estimators': [100, 500],
#        'clf__class_weight': [None],
#    }
#
#pipe = Pipeline(steps)
#
#search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=3, scoring='f1_micro', return_train_score=True, verbose=2)
#search = search.fit(X_train, y_train)
#print("Best parameter (CV scy_train%0.3f):" % search.best_score_)
#print(search.best_params_)
#
#def cv_results_to_df(cv_results):
#    results = pd.DataFrame(list(cv_results['params']))
#    results['mean_fit_time'] = cv_results['mean_fit_time']
#    results['mean_score_time'] = cv_results['mean_score_time']
#    results['mean_train_score'] = cv_results['mean_train_score']
#    results['std_train_score'] = cv_results['std_train_score']
#    results['mean_test_score'] = cv_results['mean_test_score']
#    results['std_test_score'] = cv_results['std_test_score']
#    results['rank_test_score'] = cv_results['rank_test_score']
#
#    results = results.sort_values(['mean_test_score'], ascending=False)
#    return results
#
#results = cv_results_to_df(search.cv_results_)
#results