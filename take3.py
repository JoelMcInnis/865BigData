[First name, Last name]

[Student number]
[Program]
[Cohort]
[Course Number]
[Date]


Submission to Question [X], Part [X]


import pandas as pd
# TODO: import other libraries as necessary
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)




nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
rf = RandomForestClassifier(criterion='entropy', random_state=223)
mlp = MLPClassifier(random_state=42, verbose=2, max_iter=200)




rf = RandomForestClassifier(criterion='entropy', random_state=223)


def lexicon_count(corpus):
    return np.array([textstat.lexicon_count(doc) for doc in corpus]).reshape(-1, 1)

def punc_count(corpus):
    return np.array([_get_punc(doc) for doc in corpus]).reshape(-1, 1)

feature_processing =  FeatureUnion([ 
    ('bow', Pipeline([('cv', vectorizer), ])),
    ('words', FunctionTransformer(lexicon_count, validate=False)),
    ('punc_count', FunctionTransformer(punc_count, validate=False)),
])

steps = [('features', feature_processing)]

pipe = Pipeline([('features', feature_processing), ('clf', rf)])

param_grid = {}

which_clf = "RF"

if which_clf == "RF":

    steps.append(('clf', rf))

    param_grid = {
        'features__bow__cv__preprocessor': [None, clean_text],
        'features__bow__cv__max_features': [200, 500, 1000],
        'features__bow__cv__use_idf': [False],
        'features__topics__cv__stop_words': [None],
        'features__topics__nmf__n_components': [25, 75],
        'clf__n_estimators': [100, 500],
        'clf__class_weight': [None],
    }