import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
from Q_Learning_Class import Q_Learning

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Initialize Session State
if 'q_matrix' not in st.session_state.keys():
    st.session_state['q_matrix'] = None

# Main Page
col1, col2 = st.columns(2)

with col1:

    with st.form('Init', clear_on_submit=False):
        n = st.slider('Set Range', 0, 50, step=5)
        perc_walls = st.slider('Number of Walls (as percentage of total points)', 0,100) / 100

        x = np.random.uniform(0, n)
        y = np.random.uniform(0, n)

        submit = st.form_submit_button()
    
    if submit:
        st.session_state['n'] = n
        st.session_state['perc_walls'] = perc_walls 
        st.session_state['ql'] = Q_Learning(n, perc_walls)


    with st.form('Training', clear_on_submit=False):
        num_episodes = st.slider('Number of Episodes', 0, 100)

        train = st.form_submit_button()
    
    if train:
        st.session_state['num_episodes'] = num_episodes


with col2:

    c = st.empty()

    if submit:
        fig = st.session_state['ql'].plot_environment()
        c.pyplot(fig=fig)


    if train:
        fig, ax = plt.subplots()
        ax.scatter(x, y) 
        ax.axis([0,n,0,n])

        ql = st.session_state['ql']
        num_episodes = st.session_state['num_episodes']
        
        ql.train(num_episodes)
        path = ql.get_optimal_path()
        
        fig = ql.plot_path(path)
        c.pyplot(fig=fig)