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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df_train = pd.read_csv("sentiment_train.csv")

print(df_train.info())
print(df_train.head())

df_test = pd.read_csv("sentiment_test.csv")

print(df_test.info())
print(df_test.head())







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
print ("\n\nWords head : \n", words.head(10)) 


#Sentiment Analysis 

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_clean_train['polarity'] = df_clean_train['Sentence'].apply(pol)
df_clean_train['subjectivity'] = df_clean_train['Sentence'].apply(sub)
df_clean_train['review_len'] = df_clean_train['Sentence'].astype(str).apply(len)
df_clean_train['word_count'] = df_clean_train['Sentence'].apply(lambda x: len(str(x).split()))
df_clean_train['Polarity'] = df_train['Polarity']
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








#Split training dataset into test/train
X_train, X_test, y_train, y_test = train_test_split(df_clean_train['Sentence'], df_clean_train['Polarity'], test_size=0.1, random_state=1337)
print( X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


#SVM
vectorizer = CountVectorizer()
svm = LinearSVC()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
_ = svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

#Decicion Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

