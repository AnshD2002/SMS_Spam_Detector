#importing libraries
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

#Loading models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
Mnbmodel = pickle.load(open('mnbmodel.pkl','rb'))

#function to convert meassages from upper case to lower and stemming them
def lowercase(x):
    L = []
    # converting words in lowercase
    x = x.lower()
    # Tokenization
    x = nltk.word_tokenize(x)
    # removing special characters
    for i in x:
        if i.isalnum():
            L.append(i)

    x = L[:]
    L.clear()
    # removing stop words and punctuation
    for i in x:
        if i not in stopwords.words('english') and i not in string.punctuation:
            L.append(i)
    x = L[:]
    L.clear()

    # Stemming
    for i in x:
        L.append(ps.stem(i))

    return " ".join(L)

# streamlit web code to display
st.title("SMS Spam Detector")
input_msg = st.text_input("Enter the message")

if st.button('predict'):

    transform_sms = lowercase(input_msg)
    vector_input = tfidf.transform([transform_sms])
    result = Mnbmodel.predict(vector_input)[0]
    if result == 1:
        st.header("spam")
    else:
        st.header("Not Spam")