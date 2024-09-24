# run with "streamlit run Home.py"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ressources.functions import *
from ressources.functions import about_items

st.set_page_config(
    page_title="MOFSONG Optimizer",
    page_icon=":hammer_and_pick:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=about_items
)

style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

about_markdown = read_markdown_file("pages/about.md")
st.markdown(about_markdown, unsafe_allow_html=True)

