# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 

import streamlit as st
import numpy as np
from dexpy.optimal import build_optimal
from dexpy.model import ModelOrder
from dexpy.design import coded_to_actual
from doepy import build 
from datetime import datetime
from ressources.bo import *
from ressources.functions import *
from sklearn.preprocessing import LabelEncoder
from pyDOE3 import *
from ressources.functions import about_items
import definitive_screening_design as dsd
import plotly.express as px
from itertools import combinations
import plotly.graph_objects as go
from io import BytesIO
import xlsxwriter

help = read_markdown_file("pages/help-doe.md")

st.set_page_config(page_title="Design Of Experiment", 
                   page_icon="ressources/icon.png",
                   layout="wide", menu_items=about_items)
style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

defaultParNames = ["Temperature", "Concentration A", "Concentration B", "Reaction time", "Flux", "Parameter 6", "Parameter 7", "Parameter 8", "Parameter 9", "Parameter 10", "Parameter 11", "Parameter 12", "Parameter 13", "Parameter 14", "Parameter 15", "Parameter 16", "Parameter 17", "Parameter 18", "Parameter 19", "Parameter 20"]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Design Of Experiment
""")

design_type = st.sidebar.selectbox("Design type",
        ['Sobol sequence','Full Factorial', 'Fractional Factorial', 'Definitive Screening Design', 'Space Filling Latin Hypercube', 'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman', 'Box-Behnken'])

if design_type == 'Optimal':
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

cols = st.sidebar.columns(2)
cols[0].write("N parameters")
Npars = cols[0].number_input("N parameters", min_value=1, max_value=20, value=2, label_visibility="collapsed")
parameters = np.array([{'name': '', 'actual': 0., 'coded': 0.} for i in range(Npars)])
cols[1].write("Random order")
randomize = cols[1].checkbox("Randomize", value=True, label_visibility="collapsed")

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
        parameters[par]['partype'] = cols[1].selectbox("Parameter Type", 
                                ("Numerical", "Categorical"), key=f"cat{1+par}")
        if parameters[par]['partype'] == "Numerical":
            low = cols[2].number_input("Low", value=-1., key=f"low{1+par}")
            high = cols[3].number_input("High", value=1., key=f"high{1+par}")
            other = cols[4].text_input("""Other (comma separated)""", value="", key=f"other{1+par}")
            other = [float(item) for item in other.split(",") if item]
            parameters[par]['values'] = np.array([low] + other + [high])
        else:
            other = cols[2].text_input("""Values (comma separated)""", value="A,B", key=f"other{1+par}")
            if design_type!='Sobol sequence':
                le = LabelEncoder()
                label = le.fit_transform(other.split(","))
                parameters[par]['values'] = label
                parameters[par]['encoder'] = le
            else:
                parameters[par]['values'] = other.split(",")
                parameters[par]['encoder'] = None

# # # # # # # # # # # # # # # # 

pars = {par['name']:par['values'] for par in parameters}

if design_type=='Full Factorial':
    design = build.full_fact(pars)
elif design_type=='Sobol sequence':
    Nexp = st.sidebar.number_input("Number of experiments:",
                                    min_value=1, max_value=1000, value = 1)
    ax_client = AxClient()
    params = []
    for par in parameters:
        if par['partype'] == "Numerical":
            params.append({'name': par['name'],
                           'type': 'range', 
                           'value_type': 'float',
                           'bounds': [np.min(par['values']), np.max(par['values'])]})
        else:
            params.append({'name': par['name'],
                           'type': 'choice', 
                           'values': par['values']})
    # Create the Ax experiment
    ax_client.create_experiment(
        name="DOE",
        parameters=params,
        objectives={"response": ObjectiveProperties(minimize=False)}
    )
    gs = GenerationStrategy(
                steps=[GenerationStep(
                            model=Models.SOBOL,
                            num_trials=-1,
                            should_deduplicate=True,  # Deduplicate the trials
                            model_kwargs={"seed": 165478},
                            model_gen_kwargs={},
                        )
                    ]
                )
    generator_run = gs.gen(
                experiment=ax_client.experiment,
                data=None,
                n=Nexp,
                # fixed_features=_fixed_features, 
                pending_observations=get_pending_observation_features(
                    ax_client.experiment
                )
            )
    if Npars == 1:
        trial = ax_client.experiment.new_trial(generator_run)
    else:
        trial = ax_client.experiment.new_batch_trial(generator_run)
    trials = ax_client.get_trials_data_frame()
    design = trials[trials['trial_status'] == 'CANDIDATE']
    design = design.drop(columns=['trial_index', 
                                  'trial_status', 
                                  'arm_name',
                                  'generation_method',
                                  'generation_node'
                                  ])
elif design_type=='Fractional Factorial':
    # make all parameters categorical
    for par in range(Npars):
        if parameters[par]['partype'] == "Numerical":
            parameters[par]['partype'] = "Categorical"
            le = LabelEncoder()
            label = le.fit_transform(parameters[par]['values'])
            parameters[par]['values'] = label
            parameters[par]['encoder'] = le
    reduction = st.sidebar.number_input("Reduction:", min_value=2, max_value=Npars+1, value=2)
    design = gsd([len(par['values']) for par in parameters], reduction)
    design = pd.DataFrame(design, columns=[par['name'] for par in parameters])
elif design_type=='Definitive Screening Design':
    params = {par['name']:[np.min(par['values']), np.max(par['values'])] for par in parameters}
    design = dsd.generate(factors_dict = params)
elif design_type=='Space Filling Latin Hypercube':
    design_base = build.space_filling_lhs(pars)
    Nmin = len(design_base)
    Nruns = st.sidebar.number_input("Number of experiments:",
                                    min_value=Nmin, max_value=1000, value = Nmin)
    design = build.space_filling_lhs(pars, num_samples = Nruns)
elif design_type=='Randomized Latin Hypercube':
    design_base = build.lhs(pars)
    Nmin = len(design_base)
    Nruns = st.sidebar.number_input("Number of experiments:",
                                    min_value=Nmin, max_value=1000, value = Nmin)
    design = build.lhs(pars, num_samples = Nruns)
elif design_type=='Optimal':
    reaction_design_base = build_optimal(
            Npars, 
            order=ModelOrder(order))
    Nmin = len(reaction_design_base)
    Nruns = st.sidebar.number_input("Number of experiments:", 
                                    min_value=Nmin, max_value=1000, value=Nmin)
    names = [par['name'] for par in parameters]
    lows = {par['name']:np.min(par['values']) for par in parameters}
    highs = {par['name']:np.max(par['values']) for par in parameters}
    reaction_design = build_optimal(
            Npars, 
            order=ModelOrder(order),
            run_count=Nruns)
    reaction_design.columns = names
    design = coded_to_actual(reaction_design, lows, highs)
elif design_type=='Plackett-Burman':
    design = build.plackett_burman(pars)
elif design_type=='Box-Behnken':
    if Npars<3 or any([len(par['values'])<3 for par in parameters]):
        st.error("Box-Behnken design is not possible with less than 3 parameters and with less than 3 levels for any parameter..")
        design = pd.DataFrame({})
    else:
        design = build.box_behnken(d=pars, center=1)


for par in parameters:
    if par['partype'] == "Categorical" and design_type!='Sobol sequence':
        vals = design[par['name']].to_numpy()
        design[par['name']] = par['encoder'].inverse_transform([int(v) for v in vals])
        
# # # # # # # # # # # # # # # # 

with tab3:
    design['run_order'] = np.arange(len(design))+1
    if randomize:
        ord = design['run_order'].to_numpy()
        design['run_order'] = np.random.permutation(ord)
    cols = design.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    design = design[cols]
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
    cols = st.columns(1 if Npars <= 3 else 2)
    count = 0
    if len(design.values) > 0:
        if Npars <= 2:
            # Create 2D scatter plots
            for i, faci in enumerate(parameters):
                for j, facj in enumerate(parameters):
                    if j > i:
                        fig = px.scatter(
                            design,
                            x=facj['name'],
                            y=faci['name'],
                            title=f"""{faci['name']} vs {facj['name']}""",
                            labels={facj['name']: facj['name'], faci['name']: faci['name']}
                        )
                        fig.update_traces(marker=dict(size=10))
                        fig.update_layout(
                            plot_bgcolor="white",  # White background
                            margin=dict(l=10, r=10, t=50, b=50),
                            xaxis=dict(
                                showgrid=True,  # Enable grid
                                gridcolor="lightgray",  # Light gray grid lines
                                zeroline=False,
                                showline=True,
                                linewidth=1,
                                linecolor="black",  # Black border
                                mirror=True
                            ),
                            yaxis=dict(
                                showgrid=True,  # Enable grid
                                gridcolor="lightgray",  # Light gray grid lines
                                zeroline=False,
                                showline=True,
                                linewidth=1,
                                linecolor="black",  # Black border
                                mirror=True
                            ),
                        )
                        cols[j - 1].plotly_chart(fig, use_container_width=True)
        else:
            # Create 3D scatter plots
            for k, (faci, facj, fack) in enumerate(combinations(parameters, 3)):
                fig = go.Figure(data=[go.Scatter3d(
                    x=design[facj['name']],
                    y=design[faci['name']],
                    z=design[fack['name']],
                    mode='markers',
                    marker=dict(size=10)
                )])
                fig.update_layout(
                    scene=dict(
                        xaxis_title=facj['name'],
                        yaxis_title=faci['name'],
                        zaxis_title=fack['name']
                    ),
                    title=f"{faci['name']} vs {facj['name']}<br>vs {fack['name']}",
                    margin=dict(l=10, r=10, t=50, b=50),
                    plot_bgcolor="white"
                )
                cols[count%2].plotly_chart(fig, use_container_width=True)
                count += 1
