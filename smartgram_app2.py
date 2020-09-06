import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np 
import random
from myappfunctions import cleaning, tokenize, local_css, recommendations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

        
local_css("style2.css")

# load Vectorizer
vectorizer = open("vectorizer.pkl","rb")
caption_vectorizer = joblib.load(vectorizer)
# load Models
model = open("likesmodel.pkl","rb")
likes_model = joblib.load(model)
model2 = open("commentmodel.pkl","rb")
comment_model = joblib.load(model2)
#load text df
textdf = open("textdf.pkl","rb")
vect_text_df = joblib.load(textdf)

##emptyvalues
numerical = pd.DataFrame(columns=['followings', 'followers', 'mediacount', 'selfie', 'bodysnap',
       'marketing', 'productonly', 'nonfashion', 'face', 'logo', 'brandlogo',
       'smile', 'outdoor', 'numberofpeople', 'numberoffashionproduct'])
  
######### APP BODY
title = "<h1 class='highlight instagram'> <center> Smartgram  </center><center><small style='text-transform:capitalize'> A tool for fashion influencers </small></center></h1>"
st.markdown(title, unsafe_allow_html=True)

if st.sidebar.button("App info"):
    inf0 = "<h4 style='text-transform:none'> This app predicts likes and comments in fashion related instagram posts.</span></h4>"
    inf1 = "<h4 style='text-transform:none'> It uses basic instagram information (number of followers, posts, accounts following,...), photo information characteristics (it's a selfie? product only? editorial image?) and the caption within a regresion algorithm.</span></h4>"
    inf2 = "<h4 style='text-transform:none'>  Enter your information and caption including hashtags. </span></h4>"
    st.markdown(inf0, unsafe_allow_html=True)
    st.markdown(inf1, unsafe_allow_html=True)     
    st.markdown(inf2, unsafe_allow_html=True)  
    inf3 = "<h4 style='text-transform:none'> More details about the algorithm and data used to create the app <a href='https://thaleia18.github.io/projects/smartgram_app.html'> here</a> </span></h4>"
    st.markdown(inf3, unsafe_allow_html=True)  

t0 = "<div><span class='highlight purple bold2'> Fill your data below and hit run </span></div><br>"
st.sidebar.markdown(t0, unsafe_allow_html=True)    
##### text data manipulation
tf = "<div><span class='highlight red bold'> Enter your caption (english only) </span></div>"
st.sidebar.markdown(tf, unsafe_allow_html=True)
init_text_df = pd.DataFrame(columns=['caption'])
caption = st.sidebar.text_input('')
init_text_df.at['1', 'caption'] = caption
clean_captions = cleaning(init_text_df,'caption')
clean_captions_vector = caption_vectorizer.transform(clean_captions.values).toarray()
vect_text_new = pd.DataFrame(clean_captions_vector, columns=vect_text_df.columns) 
 
####numerical data manipulation
t1 = "<div><span class='highlight red bold'> How many followers you </span></span></div>"
st.sidebar.markdown(t1, unsafe_allow_html=True)       
inp_followers = st.sidebar.number_input('',min_value=100,step=100)
numerical.at['1', 'followers'] = inp_followers
kinds = ['selfie', 'bodysnap','marketing', 'productonly', 'nonfashion', 'face', 'logo', 'brandlogo',
   'smile', 'outdoor']
t2 = "<div><span class='highlight red bold'> Kind of picture, SELECT ALL that apply </span></div>"
st.sidebar.markdown(t2, unsafe_allow_html=True)
kind_sel = st.sidebar.multiselect('',kinds)
for i in range(0,len(kinds)):
    if kinds[i] in kind_sel:
        numerical.at['1', kinds[i]] = 1  
    else:
        numerical.at['1', kinds[i]] = 0
t3 = "<div><span class='highlight red bold'> Number of people that appear in your picture </span></div>"
st.sidebar.markdown(t3, unsafe_allow_html=True)
inp_people = st.sidebar.number_input('',value=0)
numerical.at['1', 'numberofpeople'] = inp_people
t4 = "<div><span class='highlight red bold'> Number of fashion products in your picture </span></div>"
st.sidebar.markdown(t4, unsafe_allow_html=True)
inp_products = st.sidebar.number_input('',value=0,min_value=0, max_value=100)
numerical.at['1', 'numberoffashionproduct'] = inp_products
t5 = "<div><span class='highlight red bold'> How many accounts you follow </span></div>"
st.sidebar.markdown(t5, unsafe_allow_html=True)
inp_following = st.sidebar.number_input('', min_value=1,step=100)
numerical.at['1', 'followings'] = inp_following
t6 = "<div><span class='highlight red bold'> Posts in your feed </span></div>"
st.sidebar.markdown(t6, unsafe_allow_html=True)
inp_mediacount = st.sidebar.number_input('',min_value=1,step=10)
numerical.at['1', 'mediacount'] = inp_mediacount


###running the app

if st.sidebar.button("Run me!"):
    X =  pd.concat([numerical,vect_text_new.set_index(numerical.index)], axis=1, sort=False)
    pred_test_rr = likes_model.predict(X)
    pred_test_rr = pred_test_rr[0].clip(min=0)
    r = "<div><h3 class='highlight blue bold'>The predicted likes:</h3></div>"
    st.markdown(r, unsafe_allow_html=True)
    st.write(int(pred_test_rr))
    pred_test_rr2 = comment_model.predict(X)
    pred_test_rr2 = pred_test_rr2[0].clip(min=0)
    r1 = "<div><h3 class='highlight blue bold'>The predicted comments:</h3></div>"
    st.markdown(r1, unsafe_allow_html=True)
    st.write(int(pred_test_rr2))
    r2 = "<div><h3 class='highlight blue bold'>Recomendation:</h3></div>"
    st.markdown(r2, unsafe_allow_html=True)
    st.write(random.choice(recommendations))

image = Image.open('insta.jpeg')
st.image(image, use_column_width = True)

