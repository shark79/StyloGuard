# streamlit_app_updated.py

import sys
import asyncio
import platform

# Fix asyncio event loop issue on Windows with Streamlit
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

try:
    import torch
except Exception as e:
    print(f"Warning: torch import failed with error: {e}", file=sys.stderr)

import streamlit as st
import spacy
import string
import numpy as np
from collections import Counter
import psycopg2
import json
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

import PyPDF2
import docx

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# =======================
# ✨ Apply Custom CSS for ASU Theme
# =======================
# Inject custom CSS to fix label and text color globally
# --- Force Light Theme and Custom Colors ---
st.set_page_config(page_title="Stylometric App", page_icon="📝", layout="wide")

st.markdown("""
    <style>
    /* Set full-page background white */
    .main {
        background-color: #FFFFFF; /* ASU White */
    }

    /* Sidebar background color and boundary */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF; /* ASU Maroon */
    border-right: 4px solid #000000; /* ASU Rich Black Border */
    padding-right: 10px;
}
    /* Make all input labels maroon */
    label, .stTextInput > label, .stNumberInput > label, .stTextArea > label, .stSelectbox > label, .stFileUploader > label, .stRadio > label {
        color: #8C1D40 !important;
        font-weight: 600;
    }

    /* Radio/checkbox label text color */
    .stRadio > div > label, .stCheckbox > div > label {
        color: #8C1D40;
    }

    /* Header text (h1, h2, h3) color */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #8C1D40; /* Black for headings */
    }

    /* Sidebar text color white */
    .css-1v0mbdj, .css-16huue1, .css-1vzeuhh {
        color: #FFFFFF;
    }

    /* Buttons: Gold color with maroon text */
    div.stButton > button {
        background-color: #FFC627; /* ASU Gold */
        color: #8C1D40; /* Maroon text */
        font-weight: bold;
        border-radius: 8px;
        border: 2px solid #8C1D40;
    }
    div.stButton > button:hover {
        background-color: #8C1D40; /* Maroon on hover */
        color: #FFFFFF; /* White text on hover */
        transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# Shared Resources
# =======================
nlp = spacy.load("en_core_web_sm")
EMOTION_WORDS = {
    "happy","joy","delight","pleasure","elated","excited","cheerful","content",
    "sad","sorrow","grief","mourn","depressed","gloomy","melancholy",
    "angry","anger","furious","irate","annoyed",
    "fear","fright","dread","scared","terrified",
    "disgust","repulsion","revulsion","dislike",
    "surprise","astonishment","amazement",
    "trust","confidence","admiration"
}

# (Database helpers, file extraction, feature extraction, SBERT embedding functions)
def get_db_connection():
    return psycopg2.connect(
        dbname="approj", user="postgres", password="shark",
        host="localhost", port="5432"
    )

# ——————————————
# File Parsing
# ——————————————
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ):
        docx_doc = docx.Document(uploaded_file)
        for para in docx_doc.paragraphs:
            text += para.text + "\n"
    else:
        st.error("Unsupported file type.")
    return text.strip()

# ——————————————
# Stylometric Feature Extraction
# ——————————————
def count_syllables(word):
    w = word.lower()
    vowels = "aeiouy"
    n, prev = 0, False
    for c in w:
        if c in vowels and not prev:
            n += 1
            prev = True
        elif c not in vowels:
            prev = False
    if w.endswith("e") and n > 1:
        n -= 1
    return n or 1

def compute_readability(sentences, words, syllables):
    if sentences == 0 or words == 0:
        return 0
    wps = words/sentences
    spw = syllables/words
    score = 206.835 - 1.015*wps - 84.6*spw
    return round(max(score,0),2)

def compute_gunning_fog(words, sentences, complex_words):
    if words==0 or sentences==0:
        return 0
    return round(0.4*((words/sentences)+100*(complex_words/words)),2)

def compute_idio(doc):
    stops = [t.text.lower() for t in doc if t.is_alpha and t.is_stop]
    bi = list(zip(stops, stops[1:]))
    cnt = Counter(bi)
    rep = [(bg,c) for bg,c in cnt.items() if c>=2]
    return len(rep), [f"{a} {b}: {c}" for (a,b),c in rep]

def analyze_text(text):
    doc = nlp(text)
    sents = list(doc.sents)
    ns = len(sents)
    words = [t.text for t in doc if t.is_alpha]
    nw = len(words)
    uw = len(set(w.lower() for w in words))
    ttr = uw/nw if nw else 0
    avg_wlen = round(sum(len(w) for w in words)/nw,2) if nw else 0
    freqs = Counter(w.lower() for w in words)
    hapax = sum(1 for _,c in freqs.items() if c==1)
    hapax_rate = hapax/nw if nw else 0
    punct = sum(1 for t in doc if t.text in string.punctuation)
    noun_chunks = len(list(doc.noun_chunks))
    pos_counts = {p:0 for p in ["VERB","NOUN","ADJ","CCONJ","ADV","PRON"]}
    for t in doc:
        if t.pos_ in pos_counts:
            pos_counts[t.pos_] += 1
    idio_cnt, idio_list = compute_idio(doc)
    syll = sum(count_syllables(t.text) for t in doc if t.is_alpha)
    gre = compute_readability(ns,nw,syll)
    comp_words = sum(1 for t in doc if t.is_alpha and count_syllables(t.text)>2)
    gfn = compute_gunning_fog(nw,ns,comp_words)
    emo = sum(1 for t in doc if t.lemma_.lower() in EMOTION_WORDS)
    blob = TextBlob(text)
    pol = round(blob.sentiment.polarity,2)
    sia = SentimentIntensityAnalyzer()
    vad = round(sia.polarity_scores(text)["compound"],2)
    fp = {"i","me","my","mine","we","us","our","ours"}
    fp_cnt = sum(1 for t in doc if t.lower_ in fp)
    pers_ent = sum(1 for ent in doc.ents if ent.label_=="PERSON")

    return {
        "Total Word Count":         {"Count": nw,   "Note": "Total word count for the essay."},
        "Unique Word Count":        {"Count": uw,    "Note":"Total distinct words. Range: 0 to total words; higher implies broader vocabulary."},
        "Average Word Length":      {"Value": avg_wlen, "Note":"Mean number of characters per word; larger values suggest more complex vocabulary."},
        "Type-Token Ratio":         {"Value": round(ttr,2), "Note":"The ratio of number of unique words to the number of total words (0-1). Higher value suggests greater lexical diversity"},
        "Hapax Legomenon Rate":     {"Value": round(hapax_rate,2),"Note":"Proportion of words appearing once (0–1); closer to 1 indicates more unique words."},
        "Stopword Count":           {"Count": sum(1 for t in doc if t.is_stop),"Note":"Number of common function words; higher value suggests that there are more words than necessary."},
        "Contraction Count":        {"Count": sum(1 for t in doc if "'" in t.text),"Note":"Number of contractions (e.g., don't, I'm); may signal informal style."},
        "Emotion Word Count":       {"Count": emo,"Note":"Frequency of emotion-related words from an expanded lexicon."},
        "Polarity (TextBlob)":      {"Value": pol,"Note":"Sentiment polarity between -1 (very negative) and +1 (very positive), with 0 as neutral."},
        "Vader Compound":           {"Value": vad,"Note":"Sentiment polarity from Vader Compound between -1 (very negative) and +1 (very positive), with 0 as neutral."},
        "GunningFog Score":         {"Score": gfn,"Note":"Readability complexity; typically from ~5 (easy) to 20+ (difficult)."},
        "Flesch Reading Ease":      {"Value": gre,"Note":"Readability on a scale from 0 to 100; higher scores indicate easier text."},
        "First Person Count":       {"Count": fp_cnt,"Note":"Count of first-person pronouns (e.g., I, we); higher may indicate personal style."},
        "Person Entities":          {"Count": pers_ent,"Note":"Number of entities tagged as PERSON."},
        "Words per Sentence":       {"Average": round(nw/ns,2) if ns else 0,"Note":"Average count of words per sentence."},
        "Sentence Structure":       {"Sentence Length Variance": round(np.var([len([t for t in s if t.is_alpha]) for s in sents]),2),"Note":"Variance in sentence lengths; higher values indicate greater variability."},
        "Punctuation Usage":        {"Count": punct,"Note":"Total number of punctuation marks."},
        "Topics and Phrases":       {"Noun Chunks": noun_chunks,"Note":"Count of noun phrases, reflecting descriptive detail."},
        "POS Distribution":         {"Counts": pos_counts,"Note":"Frequencies of various parts of speech (e.g., VERB, NOUN, ADJ, etc.)."},
        "Idiosyncratic Expressions":{"Repeated Bigrams Count": idio_cnt,"Repeated Bigrams List": idio_list,"Note":"Count and list of repeated function-word bigrams; higher counts indicate recurring stylistic patterns."},
    }

def extract_feature_vector(feat):
    v = []
    order = [
        ("Unique Word Count","Count"),
        ("Average Word Length","Value"),
        ("Type-Token Ratio","Value"),
        ("Hapax Legomenon Rate","Value"),
        ("Stopword Count","Count"),
        ("Contraction Count","Count"),
        ("Emotion Word Count","Count"),
        ("Polarity (TextBlob)","Value"),
        ("Vader Compound","Value"),
        ("GunningFog Score","Score"),
        ("Flesch Reading Ease","Value"),
        ("First Person Count","Count"),
        ("Person Entities","Count"),
        ("Words per Sentence","Average"),
        ("Sentence Structure","Sentence Length Variance"),
        ("Punctuation Usage","Count"),
        ("Topics and Phrases","Noun Chunks"),
        ("Idiosyncratic Expressions","Repeated Bigrams Count"),
    ]
    for feat_name, subkey in order:
        v.append(feat[feat_name].get(subkey,0))
    # POS
    for p in ["VERB","NOUN","ADJ","CCONJ","ADV","PRON"]:
        v.append(feat["POS Distribution"]["Counts"].get(p,0))
    arr = np.array(v,dtype=float)
    norm = np.linalg.norm(arr)
    return arr/norm if norm>0 else arr

def student_exists(conn, sid):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM Students WHERE student_id=%s",(sid,))
    ok = cur.fetchone() is not None
    cur.close()
    return ok

def insert_student(conn, sid,name,email):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO Students(student_id,name,email) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
        (sid,name,email)
    )
    conn.commit(); cur.close()

def essay_exists(conn,sid,text):
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM Essays WHERE student_id=%s AND essay_text=%s",(sid,text))
    ok=cur.fetchone() is not None
    cur.close()
    return ok

def insert_essay(conn,sid,text,style):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO Essays(student_id,essay_text,fingerprint) VALUES(%s,%s,%s)",
        (sid,text,json.dumps({"style_index":style}))
    )
    conn.commit(); cur.close()

def fetch_student_essays(conn,sid):
    cur = conn.cursor()
    cur.execute("SELECT essay_text FROM Essays WHERE student_id=%s",(sid,))
    rows = cur.fetchall(); cur.close()
    return [r[0] for r in rows]
# =======================
# Navigation
# =======================
if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.sidebar.radio("Navigate to:", [
    "Home",
    "Direct Analysis", 
    "Feature Comparison", 
    "Stylometric Similarity Checker"
], index=["Home", "Direct Analysis", "Feature Comparison", "Stylometric Similarity Checker"].index(st.session_state.page))


# =======================
# 📜 Home Page
# =======================
if page == "Home":
    st.session_state.page = "Home"  # Ensure reset on load
    st.title("Home")

    st.markdown("""
    # StyloGuard : Detailed Authorship Verification and Analysis
    """)

    st.header("Why use this tool ?")
    st.markdown("""
    - The current authentication system can be bypassed using tools that can humanize and paraphrase AI generated and unoriginal text to make it undetectable.  
    - Using this tool as an addition to TurnitIn, evaluators can perform an in-depth analysis on writing styles of students using textual submissions from their tests and assignments.  
    - The tool can be used to capture any stylistic deviations that may help in determining integrity violations.  
    """)

    st.header("Explaining the system:")

    st.subheader("There are 3 parts of the application:")

    ## 1. Direct Analysis
    st.markdown("### 1. Direct Analysis")
    st.markdown("""
    - The first part of the tool is to perform a detailed Writing style analysis on a Single essay's features.  
    - This page extracts 20 unique features from the essay submitted for analysis, and each feature is displayed with a name, value and a note that explains the feature.  
    - The user can save the analyzed essay along with the student's details and the writing style score using the "Save to Database" option.  
    """)
    st.markdown("**Steps to use the first page:**")
    st.markdown("""
    1. Enter student details  
    2. Select an input method (paste text or upload a file)  
    3. Enter the data using selected input method  
    4. Click "Analyze" to get an Analysis Report for the essay's writing style  
    5. Save the essay with results to the database if required.  
    """)
    if st.button("🔎 Go to Direct Analysis"):
        st.session_state.page = "Direct Analysis"

    st.markdown("---")

    ## 2. Feature Comparison
    st.markdown("### 2. Feature Comparison")
    st.markdown("""
    - The second part of the tool is to perform a detailed Writing style comparison between the features of 2 essays: A reference essay and a test essay.  
    - This page compares the 20 unique features from the essays, and each feature is displayed with a name, a weighted similarity percentage and the raw similarity percentage, along with a radar chart that visualizes these similarities.  
    - The user can observe and review the comparisons to see if there are any deviations from their usual stylistic consistencies.  
    """)

    st.markdown("**Weights explained:**")
    st.markdown("""
    After personally testing and observing the patterns of all the features multiple times, I have observed that few features are more consistent in representing writing styles, in contrast to some features which have lesser or no impact at all.  

    The following weights have been added based on observations:  
    - Flesch Reading Ease: 3x  
    - Average Word Length: 3x  
    - GunningFog Score: 3x  
    - Type-Token Ratio: 2x  
    - Hapax Legomenon Rate: 2x  
    - Every other feature: 1x weight  
    """)

    st.markdown("**Steps to use the second page:**")
    st.markdown("""
    1. Enter student details  
    2. Select an input method for the reference essay (paste text, upload a file or retrieve from database)  
    3. Enter the data using selected input method  
    4. Enter the test essay using an input method of choice (Paste text or upload file)  
    5. Click "Compute feature similarity" to get a detailed weighted per-feature similarity % score, along with the radar chart.  
    6. The user can interact with the chart and also download it as an image for future use.  
    """)
    if st.button("📊 Go to Feature Comparison"):
        st.session_state.page = "Feature Comparison"

    st.markdown("---")

    ## 3. Stylometric Similarity Checker
    st.markdown("### 3. Stylometric Similarity Checker")
    st.markdown("""
    - The last part of the tool is to perform a Stylometric comparison between two essays, only this time, a fine-tuned BERT model will do the job for you!  
    - This model is trained on more than 2000 academic essays, and is fine-tuned to replace topic-related words with a masked word. So this means, the model can identify whether two essays have been written by the same author or a different author irrespective of the topic they have been written on.  
    - Since the model has been trained on academic essays, the model can only detect patterns in academic genre of essays.  
    """)

    st.markdown("**Explanation of the results:**")
    st.markdown("""
    - If the same author has written both the essays, a similarity score > 0.9 was observed.  
    - If two different authors have written the essays or the genre of the essay changes, then the similarity score dropped down to the range of 0.8.  
    """)

    st.markdown("**Steps to use the last page:**")
    st.markdown("""
    1. Select an input method for the reference and the test essays (paste text or upload a file)  
    2. Enter the data using selected input method  
    3. Click "Compute Similarity" to get a Similarity score for the two essays.  
    """)
    if st.button("🧠 Go to Stylometric Similarity Checker"):
        st.session_state.page = "Stylometric Similarity Checker"

    st.markdown("""
    ---
    <div style='text-align: center; font-size: 14px; color: #888888; margin-top: 40px;'>
    © 2025 Shashank Jamkhandi. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# ——————————————
# Page 1: Direct Analysis
# ——————————————
elif page=="Direct Analysis":
    st.title("Direct Analysis & Save to Database")
    st.subheader("Enter Student Details")
    sid = st.number_input("Student ID",min_value=0,format="%d")
    name = st.text_input("Student Name")
    email = st.text_input("Student Email")
    st.subheader("Enter Essay")
    method = st.radio("Input method",["Paste text","Upload file"])
    txt = ""
    if method=="Paste text":
        txt = st.text_area("Essay Text",height=200)
    else:
        up = st.file_uploader("PDF/DOCX file",type=["pdf","doc","docx"])
        if up: 
            txt = extract_text_from_file(up); st.write("Extracted:"); st.write(txt)

    if st.button("Analyze"):
        if sid and name and email and txt.strip():
            feats = analyze_text(txt)
            st.write("### Analysis Report for Writing Style")
            for k,v in feats.items():
                st.write(f"**{k}:**")
                for sub,val in v.items():
                    st.write(f"- **{sub}:** {val}")
        else:
            st.error("Please fill all fields.")
    if st.button("Save to Database"):
        if sid and name and email and txt.strip():
            conn=get_db_connection()
            if not student_exists(conn,sid):
                insert_student(conn,sid,name,email); st.write("Student added.")
            else:
                st.write("Student exists.")
            if not essay_exists(conn,sid,txt):
                insert_essay(conn,sid,txt,analyze_text(txt)); st.write("Essay saved.")
            else:
                st.write("Essay already in DB.")
            conn.close()
        else:
            st.error("Complete all fields.")
    st.markdown("""
    ---
    <div style='text-align: center; font-size: 14px; color: #888888; margin-top: 40px;'>
    © 2025 Shashank Jamkhandi. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# ——————————————
# Page 2: Feature Comparison
# ——————————————
elif page=="Feature Comparison":
    st.title("Feature Comparison")
    st.subheader("Enter Student Details")
    sid = st.number_input("Student ID", min_value=0, format="%d", key="cmp_id")

    # Reference Essay Input (with DB option)
    st.subheader("Reference Essay")
    ref_method = st.radio(
        "Choose reference essay input method",
        ["Paste text", "Upload file", "Use saved essay from Database"],
        key="cmp_ref_method"
    )
    if ref_method == "Paste text":
        ref = st.text_area("Reference Essay", height=150, key="cmp_ref")
    elif ref_method == "Upload file":
        up = st.file_uploader("Upload PDF/DOCX", type=["pdf","doc","docx"], key="cmp_ref_file")
        ref = ""
        if up:
            ref = extract_text_from_file(up)
            st.write("Extracted Reference Text:")
            st.write(ref)
    else:  # Use saved essay(s) from Database
        ref = ""
        if sid:
            conn = get_db_connection()
            saved = fetch_student_essays(conn, sid)
            conn.close()
            if saved:
                st.write("Using saved essays from the database as reference. They will be concatenated.")
                ref = " ".join(saved)
                st.write("Combined reference text:")
                st.write(ref)
            else:
                st.error("No saved essays found for this student. Please paste or upload instead.")
    st.subheader("Test Essay")
    m2 = st.radio("Method",["Paste text","Upload file"], key="cmp2")
    test=""
    if m2=="Paste text":
        test = st.text_area("Test Essay",height=150,key="cmp_test")
    else:
        up2 = st.file_uploader("Test PDF/DOCX",type=["pdf","doc","docx"],key="cmp_test_file")
        if up2: test=extract_text_from_file(up2); st.write(test)

    if st.button("Compute Feature Similarity"):
        if ref.strip() and test.strip():
            f1 = analyze_text(ref)
            f2 = analyze_text(test)
            # raw per-feature sim
            raw = {}
            for k in f1:
                # pick a scalar
                for sub in ("Count","Value","Score","Average"):
                    if sub in f1[k]:
                        v1=f1[k][sub]; v2=f2[k][sub]
                        break
                else:
                    continue
                if v1==v2==0: p=100
                elif v1==0 or v2==0: p=0
                else: p = min(v1,v2)/max(v1,v2)*100
                raw[k]=round(p,1)
            # define weights
            rw = {
                "Flesch Reading Ease":3,
                "Average Word Length":3,
                "GunningFog Score":3,
                "Type-Token Ratio":2,
                "Hapax Legomenon Rate":2
            }
            maxw = max(rw.values())
            normw = {k:v/maxw for k,v in rw.items()}
            # weighted sim
            weighted = {k: raw[k]*normw.get(k,1/maxw) for k in raw}
            # display
            st.write("### Weighted Per-Feature Similarity (%)")
            for k in weighted:
                st.write(f"**{k}:** {weighted[k]:.1f}% (raw {raw[k]}%)")
            # radar
            cats = list(weighted)
            vals = [weighted[c] for c in cats]
            cats += cats[:1]; vals += vals[:1]
            fig = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill="toself", name="Weighted %"
            ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])),showlegend=False)
            st.plotly_chart(fig)
        else:
            st.error("Provide both essays.")
    st.markdown("""
    ---
    <div style='text-align: center; font-size: 14px; color: #888888; margin-top: 40px;'>
    © 2025 Shashank Jamkhandi. All rights reserved.
    </div>
    """, unsafe_allow_html=True)
# ——————————————
# Page 3: SBERT Masked Similarity
# ——————————————
else:
    st.title("Stylometric Similarity Checker")
    st.markdown("""
    Content words are masked; similarity is computed via a fine-tuned SBERT model.
    """)

    @st.cache_resource
    def load_models():
        nlp_m = spacy.load("en_core_web_sm")
        sbert = SentenceTransformer("fine_tuned_triplet_model")
        tok = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", use_fast=True
        )
        return nlp_m, sbert, tok

    nlp3, model3, tokenizer3 = load_models()
    CONTENT_POS = {"NOUN","VERB","PROPN","ADJ","ADV"}

    def mask_content(text):
        doc = nlp3(text)
        return " ".join(f"<{t.pos_}>" if t.pos_ in CONTENT_POS else t.text for t in doc)

    def embed_text(text, stride=50):
        max_len = tokenizer3.model_max_length
        ids = tokenizer3(text, return_tensors="pt", truncation=False)["input_ids"][0]
        total = len(ids)
        embs = []
        i = 0

        with st.spinner("Embedding text..."):
            progress = st.progress(0)
            while i < total:
                j = min(i + max_len, total)
                chunk = tokenizer3.decode(ids[i:j], skip_special_tokens=True)
                emb = model3.encode(
                    chunk, convert_to_tensor=True,
                    truncation=True, max_length=max_len
                )
                embs.append(emb.cpu().numpy())
                i += max_len - stride
                progress.progress(min(i / total, 1.0))  # update progress bar
            progress.empty()  # remove the progress bar when done

        if not embs:
            return model3.encode(
                text, convert_to_tensor=True,
                truncation=True, max_length=max_len
            ).cpu().numpy()
        return np.mean(np.vstack(embs), axis=0)


    def compute_similarity(a, b):
        v1 = embed_text(mask_content(a))
        v2 = embed_text(mask_content(b))
        return util.cos_sim(v1, v2).item()

    st.header("Reference Essay")
    m = st.selectbox("Method", ["Paste text", "Upload file"], key="rf")
    ref = ""
    if m == "Paste text":
        ref = st.text_area("Enter reference essay", height=200, key="ref_text_area")
    else:
        f = st.file_uploader("Upload PDF/DOCX", type=["pdf", "doc", "docx"], key="rf_file")
        if f:
            ref = extract_text_from_file(f)
            st.write("Extracted Reference Text:")
            st.write(ref)

    st.header("Test Essay")
    n = st.selectbox("Method", ["Paste text", "Upload file"], key="tf")
    test = ""
    if n == "Paste text":
        test = st.text_area("Enter test essay", height=200, key="test_text_area")
    else:
        g = st.file_uploader("Upload PDF/DOCX", type=["pdf", "doc", "docx"], key="tf_file")
        if g:
            test = extract_text_from_file(g)
            st.write("Extracted Test Text:")
            st.write(test)

    if st.button("Compute Similarity"):
        if ref.strip() and test.strip():
            score = compute_similarity(ref, test)
            if score >= 0.9:
                st.success(f"**Stylometric Similarity:** {score:.3f}")
            else:
                st.error(f"**Stylometric Similarity:** {score:.3f}")
        else:
            st.error("Both essays required.")

    st.markdown("""
    ---
    <div style='text-align: center; font-size: 14px; color: #888888; margin-top: 40px;'>
    © 2025 Shashank Jamkhandi. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

