# guide to text-preprocessing
#  (helping functions)

# remove stop-words
import nltk
import re
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')

def remstopwords(text):
    text = re.sub('\[\*\*[^\]]*\*\*\]', '', text)
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower()) 
    text = re.sub(" \d+", " ", text)
    return " ".join([i for i in text.split() if i not in stopword_list])

# removes newlines,carriage return replaces them with space
# input dataframe
def preprocess_text(df):
    df.text = df.text.fillna(' ')
    df.text = df.text.str.replace('\n',' ')
    df.text = df.text.str.replace('\r',' ')
    return df

# stemming 
import nltk
from nltk.stem.snowball import SnowballStemmer
snowBallStemmer = SnowballStemmer("english")
def stemming(sentence):
    wordList = nltk.word_tokenize(sentence)
    return " ".join([snowBallStemmer.stem(i) for i in wordList])

# tokenisation, after train_test_split
import keras
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(x_train)+list(x_test)+list(x_val))
train_X = tokenizer.texts_to_sequences(x_train)
test_X = tokenizer.texts_to_sequences(x_test)
test_val = tokenizer.texts_to_sequences(x_val)

# padding
def avg_len(train_X,test_X,test_val):
    l=0
    for i in train_X:
        l+=len(i)
    for i in test_X:
        l+=len(i)
    for i in test_val:
        l+=len(i)
    return l/(len(train_X)+len(test_X)+len(test_val))

from keras.preprocessing.sequence import pad_sequences
maxlen = int(avg_len(train_X,test_X,test_val))
train_X = pad_sequences(train_X, maxlen=maxlen,padding='post')
test_X = pad_sequences(test_X, maxlen=maxlen,padding='post')
test_val = pad_sequences(test_val,maxlen=maxlen,padding='post')