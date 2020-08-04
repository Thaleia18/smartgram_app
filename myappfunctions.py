import streamlit as st
import nltk
from nltk.corpus import stopwords
import en_core_web_sm
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
import random


recommendations=['You can get more likes if your picture is a selfie.',
                'You can get more likes if your picture is an outdoor picture.',
                'You can get more likes if your picture is a full body picture.',
                'You can get more comments if your picture is an outdoor picture.',
                'You can get more comments if your picture includes a fashion product.',
                'Using hashtags like #ootd #welovefasion #instafashion and brandhashtags increase your metrics',
                'Editorialized images like runways, awards ceremonies have less engagement.']

nltk.download('words')
nlp = en_core_web_sm.load()
words = set(nltk.corpus.words.words())
nltk.download('stopwords')

stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def cleaning(frame,col):
    """ 
    Function to clean text from a column in a data frame 
  
    This funtion removes non alphabethic characters,stop words and numerical characters and return text in lowers. 
  
    Parameters: 
    Data frame, text column 
  
    Returns: 
    values clean text from column
  
    """
    newframe=frame.copy()   
    punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','#',"%"]
    stop_words = text.ENGLISH_STOP_WORDS.union(punc)
    stop_words = list(stop_words)  
    newframe[col]=newframe[col].str.replace('\d+', '').str.replace('\W', ' ').str.lower().str.replace(r'\b(\w{1,3})\b', '')
    newframe[col] = [' '.join([w for w in x.lower().split() if w not in stop_words]) for x in newframe[col].tolist()] 
    newframe['Cleantext'] =[' '.join(word for word in x.split() if not word.startswith('uf')if not word.startswith('ue')if not word.startswith('u0')) for x in newframe[col].tolist()] 
 
    content = newframe['Cleantext']#.values
    return content


def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text)]
    
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)