import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_data():
    df = pd.read_csv('StudySpot.csv')

    # function to add keywords to each spot
    def create_sentence_list(spot):
        sentences = []
        if spot['Aesthetics'] > 0.6: sentences.append("aesthetic pretty vibey architecture")
        if spot['Noise'] < 0.3: sentences.append("silent peaceful quiet focus not-loud")
        elif spot['Noise'] > 0.6: sentences.append("loud social group-work friends talkative")
        if spot['Traffic'] < 0.3: sentences.append("empty quiet uncrowded not-crowded seats")
        if spot['Outlets'] == 1: sentences.append("outlets charging electricity cables laptop ipad charger")
        if spot['Food'] == 1: sentences.append("cafe food coffee matcha snacks pastries drinks water")
        # add a space between and join into a string
        return " ".join(sentences)

    # add keywords to each spot
    df['Spot_Description'] = df.apply(create_sentence_list, axis=1)
    
    # initialize vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # transform each spot descriptions into its own matrix using vectorizer
    tfidf_matrix = vectorizer.fit_transform(df['Spot_Description'])
    
    return df, vectorizer, tfidf_matrix
df, vectorizer, tfidf_matrix = load_data()

st.title("UCLA Study Spot Recommendation System")

# UPDATE SLIDERS ================================================
# create base state for sliders
if 'Noise' not in st.session_state:
    st.session_state['Noise'] = 0.5
if 'Aesthetics' not in st.session_state:
    st.session_state['Aesthetics'] = 0.5
if 'Traffic' not in st.session_state:
    st.session_state['Traffic'] = 0.5

# set sliders/toggles using text input
def update_sliders_from_text():
    # retrieve session state text_query if there is text input
    query = st.session_state['text_query']

    # create scores from text input & find top match's index
    if query:
        # use vectorizer to transform query into vector
        query_vec = vectorizer.transform([query])
        # create scores from cosine similarity of query (vector form)
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        # find index of top match
        top_idx = scores.argmax()
        
        # update sliders using scores
        if scores[top_idx] > 0:
            # add best match's level to each slider's session state
            spot = df.iloc[top_idx]
            st.session_state['Noise'] = float(spot['Noise'])
            st.session_state['Aesthetics'] = float(spot['Aesthetics'])
            st.session_state['Traffic'] = float(spot['Traffic'])
            st.session_state['Outlets'] = bool(spot['Outlets']==1)
            st.session_state['Food'] = bool(spot['Food']==1)
    else:
        st.session_state['Noise'] = 0.5
        st.session_state['Aesthetics'] = 0.5
        st.session_state['Traffic'] = 0.5
        st.session_state['Outlets'] = False
        st.session_state['Food'] = False

# RESET BUTTON ================================================
# reset sliders/toggles to text input state
def reset_to_text():
    # works only if text query exists; basically same process as above
    if st.session_state.text_query:
        query_vec = vectorizer.transform([st.session_state.text_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_idx = scores.argmax()
        if scores[top_idx] > 0:
            spot = df.iloc[top_idx]
            st.session_state['Noise'] = float(spot['Noise'])
            st.session_state['Aesthetics'] = float(spot['Aesthetics'])
            st.session_state['Traffic'] = float(spot['Traffic'])
            st.session_state['Outlets'] = bool(spot['Outlets']==1)
            st.session_state['Food'] = bool(spot['Food']==1)

# TEXT INPUT BOX ================================================
st.header("Step 1: Describe!")
# text input using the callback
st.text_input("Search keywords to auto-set sliders", 
    placeholder="e.g. quiet, matcha",
    key="text_query",
    on_change=update_sliders_from_text
)

# PRODUCE SLIDERS AND TOGGLES ================================================
st.header("Step 2: Adjust!")
# noise slider
st.subheader("Noise Level",divider="#FFD100")
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.write("Silent")
with col2:
    noise_val = st.slider("noise_slider", 0.0, 1.0, key="Noise", label_visibility="collapsed")
with col3:
    st.write("Social")
# aesthetic slider
st.subheader("Aesthetic Level",divider="#FFD100")
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.write("Plain")
with col2:
    aes_val = st.slider("aesthetics_slider", 0.0, 1.0, key="Aesthetics", label_visibility="collapsed")
with col3:
    st.write("Instagram-able")
# traffic slider
st.subheader("Traffic Level",divider="#FFD100")
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.write("Empty")
with col2:
    traf_val = st.slider("traffic_slider", 0.0, 1.0, key="Traffic", label_visibility="collapsed")
with col3:
    st.write("Full")
# outlets & food toggle
st.subheader("Amenities",divider="#FFD100")
col1, col2 = st.columns(2)
with col1:
    outlets_on = st.toggle("Need outlets?",key="Outlets")
with col2:
    food_on = st.toggle("Need food or coffee?",key="Food")
# reset button
st.button("Reset to Text", on_click=reset_to_text)

# CALCULATE SCORE ================================================
# calculation of closest point in 3D space of noise, aesthetics, traffic
df['dist'] = np.linalg.norm(
    df[['Noise', 'Aesthetics', 'Traffic']].values - [noise_val, aes_val, traf_val], 
    axis=1
)
# convert distance to a score (smaller distance = higher score)
df['Final_Score'] = (1 / (1 + df['dist'])) * 100
# bonus scores for outlets and food
if food_on:
    df.loc[df['Food'] == 1, 'Final_Score'] += 10
    df.loc[df['Final_Score']> 100, 'Final_Score'] = 100
if outlets_on:
    df.loc[df['Outlets'] == 1, 'Final_Score'] += 10
    df.loc[df['Final_Score']> 100, 'Final_Score'] = 100

# RESULTS ================================================
st.header("Step 3: Study!")
st.subheader("Your Best Matches")
# sort for final results
results = df.sort_values(by='Final_Score', ascending=False).head(3)
# display scores for each of the top 3
for i, spot in results.iterrows():
    st.info(f"{spot['Location']} — Match: {int(spot['Final_Score'])}%")