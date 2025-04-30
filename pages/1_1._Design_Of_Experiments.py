# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 

import streamlit as st
import numpy as np
import pandas as pd
from ressources.functions import about_items
from io import BytesIO
import xlsxwriter
from ressources.functions import *
from optimeo.doe import *
from datetime import datetime

help = read_markdown_file("pages/help-doe.md")

st.set_page_config(page_title="Design Of Experiment", 
                   page_icon="ressources/icon.png",
                   layout="wide", menu_items=about_items)
style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

defaultParNames = ["Temperature", "ConcentrationA", "ConcentrationB", "Reaction_time",
                   "Flux", "Parameter_6", "Parameter_7", "Parameter_8", "Parameter_9", 
                   "Parameter_10", "Parameter_11", "Parameter_12", "Parameter_13", 
                   "Parameter_14", "Parameter_15", "Parameter_16", "Parameter_17", 
                   "Parameter_18", "Parameter_19", "Parameter_20"]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Design Of Experiment
""")

design_type = st.sidebar.selectbox("Design type",
        ['Sobol sequence','Full Factorial', 'Fractional Factorial', 
         'Definitive Screening Design', 'Space Filling Latin Hypercube', 
         'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman', 'Box-Behnken', 'Central Composite'])

cols = st.sidebar.columns(2)
cols[0].write("N parameters")
Npars = cols[0].number_input("N parameters", min_value=1, max_value=20, value=2, label_visibility="collapsed")
parameters = np.array([{'name': '', 'actual': 0., 'coded': 0.} for i in range(Npars)])
cols[1].write("Random order")
randomize = cols[1].checkbox("Randomize", value=True, label_visibility="collapsed")
reduction = 2
order = 2
Nexp = 4
center = (1,1)
alpha = 'o'
face = 'ccc'
feature_constraints = []
# # # # # # # # # # # # # # # # 
tab1, tab2, tab3 = st.tabs(["How to choose the proper design?", "Parameters Ranges", "Experimental Design"])

with tab1:
    st.markdown(help, unsafe_allow_html=True)

with tab2:
    for par in range(Npars):
        cols = st.columns(5)
        parameters[par]['name'] = cols[0].text_input(f"**Parameter {par+1}:**", 
                                            key=f"par{1+par}", 
                                            value=defaultParNames[par]).replace(" ","_")
        parameters[par]['type'] = cols[1].selectbox("Parameter Type", 
                                ("Float", "Integer", "Categorical"), key=f"cat{1+par}")
        if parameters[par]['type'] != "Categorical":
            low = cols[2].number_input("Low", value=-1., key=f"low{1+par}")
            high = cols[3].number_input("High", value=1., key=f"high{1+par}")
            other = cols[4].text_input("""Other (comma separated)""", value="", key=f"other{1+par}")
            other = [float(item) for item in other.split(",") if item]
            parameters[par]['values'] = np.array([low] + other + [high])
        else:
            other = cols[2].text_input("""Values (comma separated)""", value="A,B", key=f"other{1+par}")
            parameters[par]['values'] = other.split(",")


if design_type in ['Sobol sequence', 'Space Filling Latin Hypercube', 
                   'Randomized Latin Hypercube', 'Optimal']:
    Nexp = st.sidebar.number_input("Number of experiments:",
                                    min_value=1, max_value=1000, value = 1)
    feature_constraints = st.sidebar.text_input("""Add **linear** constraints on the features (if any). Use a comma to separate multiple constraints.""",
                help="""The constraints should be in the form of inequalities such as:

- `x1 >= 0`
- `x2 <= 10, x4 >= -0.5`
- `x1 + 3*x2 <= 5`""")
    if len(feature_constraints)>0:
        feature_constraints = feature_constraints.replace("+", " + ")
        feature_constraints = feature_constraints.replace("<", "<=")
        feature_constraints = feature_constraints.replace(">", ">=")
        feature_constraints = feature_constraints.replace("<==", "<=")
        feature_constraints = feature_constraints.replace("<=", " <= ")
        feature_constraints = feature_constraints.replace(">==", ">=")
        feature_constraints = feature_constraints.replace(">=", " >= ")
        feature_constraints = feature_constraints.split(",")
    else:
        feature_constraints = []
elif design_type=='Fractional Factorial':
    reduction = st.sidebar.number_input("Reduction:", min_value=2, max_value=Npars+1, value=2)
elif design_type=='Optimal':
    model_order = st.sidebar.selectbox("Model order:", 
            ['quadratic','linear','cubic','constant'])
    if model_order == 'linear':
        order = 1
    elif model_order == 'quadratic':
        order = 2
    elif model_order == 'cubic':
        order = 3
    else:
        order = 0
elif design_type=='Central Composite':
    center = st.sidebar.slider("Number of replications of the center point:",
            value=1, min_value=0, max_value=10, step=1)
    center = (0, center)
    face = st.sidebar.selectbox("Type of design:", ['Circumscribed', 'Inscribed', 'Faced'])
    if face == 'Circumscribed':
        face = 'ccc'
    elif face == 'Inscribed':
        face = 'cci'
    elif face == 'Faced':
        face = 'ccf'
    # alpha = st.sidebar.selectbox("Alpha", ['orthogonal', 'rotatable'])
    # if alpha == 'orthogonal':
    #     alpha = 'o'
    # elif alpha == 'rotatable':
    #     alpha = 'r'
    # change parameter values to list
    for par in range(Npars):
        parameters[par]['values'] = parameters[par]['values'].tolist()

doe = DesignOfExperiments(
    type                = design_type,
    parameters          = parameters,
    Nexp                = Nexp,
    reduction           = reduction,
    order               = order,
    randomize           = randomize,
    feature_constraints = feature_constraints,
    center              = center,
    alpha               = alpha,
    face                = face
)
design = doe.design

# # # # # # # # # # # # # # # # 

with tab3:
    # add an empty "response" column to the design
    design['response'] = ''
    timestamp = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    outfile = writeout(design)
    cols= st.columns([4,1,1,7])
    cols[0].write(f"Download Experimental Design with {len(design)} runs:")
    cols[1].download_button(
        label     = f"CSV",
        data      = outfile,
        file_name = f'DOE_{timestamp}.csv',
        mime      = 'text/csv',
        key       = 'download-csv'
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df = clean_names(design)
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        writer.close()
        cols[2].download_button(
            label=f"XLSX",
            data=buffer,
            file_name=f'DOE_{timestamp}.xlsx',
            mime="application/vnd.ms-excel",
        )
    st.dataframe(design, hide_index=True)
    # plot the design
    Npars = len(parameters)
    figs = doe.plot()
    cols = st.columns(1 if Npars <= 3 else 2)
    count = 0
    if len(design.values) > 0:
        for i in range(len(figs)):
            cols[i%2].plotly_chart(figs[i], use_container_width=True)