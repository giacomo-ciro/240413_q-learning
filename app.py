import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
from Q_Learning_Class import Q_Learning

st.set_page_config(
    page_title="Q-Learning Playground",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://giacomo-ciro.github.io/',
        'Report a bug': "https://giacomo-ciro.github.io/",
        'About': "Q-Learning Playground!"
    }
)

# Initialize Session State
if 'q_matrix' not in st.session_state.keys():
    st.session_state['q_matrix'] = None

# Main Page
st.write('''
# Q-Learning Playground  
### Help the Agent find the shortest path to reach its Target, while avoiding Bombs!  

How it works:
1) Initialize the environment by setting the grid size and the number of bombs, defined as percentage of total space available (100% means the entire space is covered by bombs).
2) Update the Q-Values Matrix by setting the number of training episodes (Q-Values are reset only when a new environment is initialized).        
''')

# Columns
col1, col2 = st.columns(2)

with col1:

    with st.form('Init', clear_on_submit=False):
        n = st.slider('Square Grid Size (side length)', 0, 50, step=5)
        perc_bombs = st.slider('Percentage of Bombs', 0,100) / 100

        x = np.random.uniform(0, n)
        y = np.random.uniform(0, n)

        submit = st.form_submit_button('Initialize Environment')
    
    if submit:
        st.session_state['n'] = n
        st.session_state['perc_bombs'] = perc_bombs 
        st.session_state['ql'] = Q_Learning(n, perc_bombs)


    with st.form('Training', clear_on_submit=False):
        num_episodes = st.slider('Number of Episodes', 0, 1000)

        train = st.form_submit_button('Train!')
    
    if train:
        st.session_state['num_episodes'] = num_episodes


with col2:

    c = st.empty()

    if submit:
        fig = st.session_state['ql'].plot()
        c.pyplot(fig=fig)


    if train:
        fig, ax = plt.subplots()
        ax.scatter(x, y) 
        ax.axis([0,n,0,n])

        ql = st.session_state['ql']
        num_episodes = st.session_state['num_episodes']
        
        ql.train(num_episodes)
        path = ql.get_optimal_path() # Greedy path
        
        fig = ql.plot(path)
        c.pyplot(fig=fig)
