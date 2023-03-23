import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import string
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email Spam Classifier")
input_email=st.text_area("Enter the email")
if st.button('Predict'):
# preprocess
  def transform_text(text):
      text=text.lower()
      text=nltk.word_tokenize(text)
      y=[]
      for i in text:
          if i.isalnum():
              y.append(i)
      text=y[:]
      y.clear()
      for i in text:
          if i not in stopwords.words('english') and i not in string.punctuation:
              y.append(i)
      text=y[:]
      y.clear()
      for i in text:
          y.append(ps.stem(i))
      return " ".join(y)
  transform_email=transform_text(input_email)
  # vectorize
  vector_input=tfidf.transform([transform_email])
  #predict
  result=model.predict(vector_input)[0]
  #display
  if result==1:
      st.header("Spam")
  else:
      st.header("Not Spam")