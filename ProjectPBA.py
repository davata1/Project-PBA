import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re, string
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from itertools import chain
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Download resources
nltk.download('popular')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('https://github.com/davata1/Project-PBA/blob/main/covid.csv')
df.drop_duplicates(inplace=True)
df = df.drop(['Unnamed:','Datetime', 'Tweet Id'], axis=1)

# Text Cleaning
def cleaning(text):
    # HTML Tag Removal
    text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))

    # Case folding
    text = text.lower()

    # Trim text
    text = text.strip()

    # Remove punctuations, karakter spesial, and spasi ganda
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)

    # Number removal
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Mengubah text 'nan' dengan whitespace agar nantinya dapat dihapus
    text = re.sub('nan', '', text)

    return 

def preprocess_data(df):
    # Preprocess text
    df['Username'] = df['Username'].apply(lambda x: cleaning(x)) 
    df['Text'] = df['Text'].apply(lambda x: cleaning(x))
    
    # Tokenizing text
    df['Text_token'] = df['Text'].apply(lambda x: word_tokenize(x))
    
    # Removing stopwords
    stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
    df['Text_token'] = df['Text_token'].apply(lambda x: [w for w in x if not w in stop_words])

    # Stemming text
    tqdm.pandas()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df['Text_token'] = df['Text_token'].progress_apply(lambda x: stemmer.stem(' '.join(x)).split(' '))

    return df


# Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def label_text():
    score = sia.polarity_scores()
    if score['compound'] >= 0.05:
        return 1  # tweet bernilai positif
    else:
        return 0  # tweet bernilai negatif
    
    # Preprocess data
@st.cache_data()
def preprocess_data_cached(df):
    return preprocess_data(df)

# Streamlit App
st.title("Analisis Sentimen dan Pemodelan")
st.header("Analisis Dataset")
st.dataframe(df)

st.header("Preprocessing dan Tokenisasi")
if st.button("Preprocessing Data"):
    processed_data = preprocess_data_cached(df)
    st.success("Preprocessing data selesai.")
    st.dataframe(processed_data)

st.header("Analisis Sentimen")
sentiment_text = st.text_input("Masukkan teks untuk analisis sentimen:")
if sentiment_text:
    sentiment_prediction = label_text(sentiment_text)
    sentiment_label = "Positif" if sentiment_prediction == 1 else "Negatif" if sentiment_prediction == 2 else "Netral"
    st.success(f"Sentimen: {sentiment_label}")

st.header("Performa Model")
accuracy = 0.0
precision = 0.0
recall = 0.0




