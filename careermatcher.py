import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle as pkl
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
# nlp = spacy.load('en_core_web_lg')
import re
import docx2txt
from spacy.matcher import PhraseMatcher

from transformers import BertForSequenceClassification
from transformers import BertTokenizer


import torch

st.set_page_config(
    page_title="Resume Scanner",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


output_dir = "model_save"
enc_dir = "target_encodings.pkl"
matcher_dir = "linkedin_skill.txt"

@st.cache
def nlp_load():
    load_llm = spacy.load('en_core_web_lg')
    return load_llm

nlp = nlp_load()

matcher = PhraseMatcher(nlp.vocab)

@st.cache
def bert(dir_):
    model_loaded_temp = BertForSequenceClassification.from_pretrained(dir_)
    return model_loaded_temp

@st.cache
def bert_token(dir_tok):
    tokenizer_loaded_temp = BertTokenizer.from_pretrained(dir_tok)
    return tokenizer_loaded_temp

@st.cache
def label_enc(dir_enc):
    enc = pkl.load(open(dir_enc, 'rb'))
    return enc

@st.cache
def ph_match(fold):
    with open(fold, 'r', encoding='utf-8') as file:
        text = file.read()

    return text



# encode =  pkl.load(open("target_encodings.pkl", 'rb'))
# output_dir = "D:\\Models\\model_save"

label_encoder = label_enc(enc_dir)
# model_loaded = BertForSequenceClassification.from_pretrained(output_dir)
# tokenizer_loaded = BertTokenizer.from_pretrained(output_dir)
model_loaded = bert(output_dir)
tokenizer_loaded = bert_token(output_dir)

txt = ph_match(matcher_dir)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 150px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: centre; color: cyan;'>RESUME/CV SCANNER</h1>",
                unsafe_allow_html=True)
st.markdown("<h6 style='text-align: centre; color: white;'>Know which domain fit's your resume :)</h1>",
                unsafe_allow_html=True)

stops = list(STOP_WORDS) 

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None

def cleanResume(resumeText):
    resumeText = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",resumeText).split())
    resumeText = re.sub(r'[^\x00-\x7F]+',r' ', resumeText)
    resumeText = ''.join(resumeText.splitlines())
    return resumeText

def complete_pack(x):
  demo = nlp(x)
  lst = [i.text.lower() for i in demo if i.text.lower() not in stops]
  return lst


with st.sidebar:
    global resume_text, upload
    global resume_text_spacy, re_temp
    upload = st.file_uploader("DRAG AND DROP YOUR RESUME NOW")
    st.markdown("<h5 style='text-align: centre; color: red;'>Only .docx type files accepted</h1>",
                unsafe_allow_html=True)
    if upload:
        try:
            resume_text = extract_text_from_docx(upload)
            resume_text = resume_text.replace('\n\n', ' ')
            re_temp = cleanResume(resume_text)
            resume_text_spacy = nlp(re_temp)
        except Exception as e:
            st.error('WRONG FILE FORMAT : Only .docx(WORD DOC) type of files are accepted')


scan = st.button('SCAN 📝')
if scan:
    try:
        emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", resume_text)
        phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', resume_text)
        links = re.findall(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", resume_text)


        txt = txt.split('\n')
        ev = [nlp.make_doc(i) for i in txt]
        matcher.add("SKILLS", None, *ev)
        get_skills = matcher(resume_text_spacy)

        demo = []
        for match_id, start, end in get_skills:
            span = resume_text_spacy[start : end]
            demo.append(span.text)

        re_text = ' '.join(demo)  
        my_skills_re_text = re_text
        my_skills_clean_re_text = cleanResume(my_skills_re_text)

        skills = complete_pack(my_skills_clean_re_text)
        skills = ' '.join(skills)
        lst = []
        lst.append(skills)


        model_loaded.eval()

# Tokenize the input text
        input_ids = tokenizer_loaded.encode(lst[0], add_special_tokens=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

        # Move the input tensor to the same device as the model
        # input_ids = input_ids.to(device)
        # model_loaded = model_loaded.to(device)

        # Perform the forward pass to get the model's predictions
        with torch.no_grad():
            result = model_loaded(input_ids, token_type_ids=None, attention_mask=None, return_dict=True)
            logits = result.logits

        # Move the logits to the CPU and convert to numpy array
        logits = logits.detach().cpu().numpy()

        # Get the predicted label
        predicted_label = np.argmax(logits)

        # Print the predicted label
        # st.write("Predicted Label:", predicted_label)

        probs = logits[0]
        # print("text:", lst[0])
        # print("predictions:", probs)
        pred_idx = np.argmax(probs) 
        # kp = list(pred_idx)
        d = {}
        ind = 0

        for i in probs:
            d[label_encoder.inverse_transform([ind])[0]] = i
            ind+=1
        # st.write("Your skills are matching to : ", label_encoder.inverse_transform([pred_idx])[0])
        domain = label_encoder.inverse_transform([pred_idx])[0]
        data = pd.DataFrame({'Domains' : list(d.keys()), 'Probs' : list(d.values())})
        # st.markdown(f"**Your skills are matching to:** <span style='color: cyan;'>{domain}</span>", unsafe_allow_html=True) #BF3EFF
        st.markdown(f"<span style='color: #BF3EFF;'>**Your skills are matching to :**</span> <span style='color: #54FF9F;'>{domain}</span>", unsafe_allow_html=True)
        datacpy = data.copy()
        datacpy['Probs'] = datacpy['Probs']*10
        datacpy.rename(columns={'Probs': 'Percentage Prediction of your Domain'}, inplace=True)

        st.markdown("<h3 style='text-align: centre; color: blue;'>PERCENT OF YOUR DOMAIN MATCH</h3>",
                unsafe_allow_html=True)

        st.dataframe(datacpy.sort_values('Percentage Prediction of your Domain', ascending=False))
        domains = px.bar(data, x = 'Domains', y = 'Probs',width=800, height=400) 
        st.plotly_chart(domains)

        
        if len(list(set(emails))) > 0:
            st.markdown("<h4 style='text-align: centre; color: blue;'>EMAIL ✔️ </h1>",
                unsafe_allow_html=True)
            st.success(list(set(emails)))
        else:
            st.markdown("<h4 style='text-align: centre; color: blue;'>EMAIL ❌ </h1>",
                unsafe_allow_html=True)
            st.error('Email-Id is not present try including it in your Resume')



        if len(list(set(phone))) > 0:
            st.markdown("<h4 style='text-align: centre; color: blue;'>MOBILE NO ✔️ </h1>",
                unsafe_allow_html=True)
            st.success(list(set(phone)))
        else:
            st.markdown("<h4 style='text-align: centre; color: blue;'>MOBILE NO ❌ </h1>",
                unsafe_allow_html=True)
            st.error('Mobile number is not present try including it in your Resume')


        
        if len(list(set(links))) > 0:
            st.markdown("<h4 style='text-align: centre; color: blue;'>LINKS ✔️ </h1>",
                unsafe_allow_html=True)
            st.success(list(set(links)))
        else:
            st.markdown("<h4 style='text-align: centre; color: blue;'>LINKS ❌</h1>",
                unsafe_allow_html=True)
            st.error("Link's are not present try including your Github or LinkedIn Profile in your Resume")


    except Exception as e:
        st.write(e)
        st.error("😲 Try uploading your file again")



