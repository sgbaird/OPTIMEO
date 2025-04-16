# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 

import streamlit as st
import numpy as np
from ressources.functions import *
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optimeo.analysis import *

st.set_page_config(page_title="Data Analysis and Modeling", 
                   page_icon="ressources/icon.png", 
                   layout="wide", menu_items=about_items)

style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

if not 'analysis' in st.session_state:
    st.session_state.analysis = None
if not 'figml' in st.session_state:
    st.session_state.figml = None
if not 'modml' in st.session_state:
    st.session_state.modml = None
if not 'modlin' in st.session_state:
    st.session_state.modlin = None
if not 'figlin' in st.session_state:
    st.session_state.figlin = None

def data_changed():
    st.session_state.analysis = None
    st.session_state.figml = None
    st.session_state.modml = None
    st.session_state.modlin = None
    st.session_state.figlin = None

def recompute_ML_fig():
    st.session_state.figml = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Data Analysis and Modeling
""")

tabs = st.tabs(["Data Loading", "Visual Assessment", "Linear Regression Model", 'Machine Learning Model'])

with tabs[0]: # data loading
    datafile = st.sidebar.file_uploader("""Upload data file (csv, xls, xlsx, xlsm, xlsb, odf, ods and odt).""", 
                type=["csv",'xlsx','xls', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'],
                help="The data file should contain the factors and the response variable.")
    if datafile is None:
        st.markdown(
        """⚠️ The data must be in tidy format, meaning that each column is a variable and each row is an observation. We usually place the factors in the first columns and the response(s) in the last column(s). Data type can be float, integer, or text, and you can only specify one response. Spaces and special characters in the column names will be automatically removed. The first row of the file will be used as the header.

For Excel-like files, the first sheet will be used, and data should start in the A1 cell, and no unnecessary rows or columns should be present. 
        """
        )
        cols=st.columns([1,4,1])
        cols[1].image("ressources/tidy_data.jpg", caption="Example of tidy data format")
    if datafile is not None:
        left,right=st.columns([1,1])
        if Path(datafile.name).suffix == '.csv':
            data = pd.read_csv(datafile)
        else:
            data = pd.read_excel(datafile)
        data = clean_names(data, remove_special=True, case_type='preserve')
        cols = data.columns.to_numpy()
        st.dataframe(data, hide_index=True)
        mincol = 1 if 'run_order' in cols else 0
        factors = left.multiselect("Select the **factors** columns:", 
                data.columns, default=cols[mincol:-1], on_change=data_changed)
        # response cannot be a factor, so default are all unselected columns in factor
        available = [col for col in cols if col not in factors]
        response = [right.selectbox("Select the **response** column:", 
                available, index=len(available)-1, on_change=data_changed)]
        if len(response) > 0:
            response = response[0]
        dtypes = data.dtypes
        st.session_state.analysis = DataAnalysis(data, factors, response)
        encoders = st.session_state.analysis.encoders


with tabs[1]: # visual assessment
    if datafile is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the "Data Loading" tab.""")
    if datafile is not None and len(factors) > 0 and len(response) > 0:

        # Set up the layout for Streamlit
        ncols = min(len(factors), 4)
        cols = st.columns(4)

        # Q-Q Plot
        fig = st.session_state.analysis.plot_qq()
        cols[0].plotly_chart(fig)

        # Box Plot
        fig = st.session_state.analysis.plot_boxplot()
        cols[1].plotly_chart(fig)

        # Histogram
        fig = st.session_state.analysis.plot_histogram()
        cols[2].plotly_chart(fig)

        # Scatter Plot of Response Values
        fig = st.session_state.analysis.plot_scatter_response()
        cols[3].plotly_chart(fig)

        # Scatter Plots for Each Factor
        cols = st.columns(ncols)
        for i, factor in enumerate(factors):
            fig = px.scatter(x=data[factor], 
                             y=data[response], 
                             labels={'x': factor, 'y': response}, 
                             title=f'{response} vs {factor}')
            fig.update_layout(
                plot_bgcolor="white",  # White background
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=10, r=10, t=100, b=10),
                xaxis=dict(
                    showgrid=True,  # Enable grid
                    gridcolor="lightgray",  # Light gray grid lines
                    zeroline=False,
                    zerolinecolor="black",  # Black zero line
                    showline=True,
                    linewidth=1,
                    linecolor="black",  # Black border
                    mirror=True
                ),
                yaxis=dict(
                    showgrid=True,  # Enable grid
                    gridcolor="lightgray",  # Light gray grid lines
                    zeroline=False,
                    zerolinecolor="black",  # Black zero line
                    showline=True,
                    linewidth=1,
                    linecolor="black",  # Black border
                    mirror=True
                ),
            )
            fitorder = cols[i % ncols].number_input(f"Polynomial fit order for {response} vs {factor}:",
                                        min_value=0, value=2, max_value=10, step=1)
            # Add linear regression with red line and equation
            if dtypes[factor] != 'object':
                p = np.polyfit(data[factor], data[response], fitorder)
                x_range = np.linspace(np.min(data[factor]), np.max(data[factor]), 100)
                fig.add_trace(go.Scatter(x=x_range, y=np.polyval(p, x_range), 
                                         mode='lines', 
                                         name='Polynomial Fit', 
                                         line=dict(color='red')))
                fig.update_layout(title_subtitle_text=rf'{write_poly(p)}',
                                  title_subtitle_font_size=16,
                                  showlegend=False, height=400)

            cols[i % ncols].plotly_chart(fig)
        # Pairplot for all factors
        st.write("##### Pairplot of all factors")
        if st.button("Compute the pairplot"):
            # Create a pairplot of the dataset
            # Use a copy of the data to avoid modifying the original
            fig = st.session_state.analysis.plot_pairplot()
            st.pyplot(fig, use_container_width=True)


with tabs[2]: # simple model
    if datafile is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the "Data Loading" tab.""")
    if datafile is not None and len(factors) > 0 and len(response) > 0:
        cols = st.columns([1,1,4])
        order = cols[0].number_input("Interactions order:", 
                                        min_value=1, value=1, 
                                        max_value=min([4,len(factors)]))
        quadratic = cols[1].multiselect("Quadratic terms?", factors)
        def_eqn = st.session_state.analysis.write_equation(order, quadratic)
        st.session_state.analysis.equation = cols[2].text_input("Model equation:", key="eqn", value=def_eqn, help="""Interactions are written as `factor1:factor2`.  
Powered terms are written as `I(factor**power)`.  
Categorical variables are written as `C(factor)`.  
You can also use `np.log(factor)` or `np.exp(factor)` for transformations.  
To remove the intercept, add `-1` at the end of the equation.""")
        if st.button("Compute the linear model"):
            st.session_state.analysis.compute_linear_model()
            st.session_state.modlin = st.session_state.analysis.linear_model
            st.session_state.figlin = st.session_state.analysis.plot_linear_model()
            # Plot: Actual vs Predicted
        if st.session_state.modlin is not None:
            st.write(st.session_state.modlin.summary())
            cols = st.columns([1, 1])
            cols[0].plotly_chart(st.session_state.figlin[0])
            cols[1].plotly_chart(st.session_state.figlin[1])
            # # # # # # # # # # # # # # # 
            st.write("##### Predict the response for a set of factors with this linear model:")
            Xnew = {factor: 0 for factor in factors}
            left, right = st.columns(2)
            for i,factor in enumerate(factors):
                colsinput = left.columns(2)
                colsinput[0].write(f"<p style='text-align:right;font-size:1.1em'><b>{factor}</b></p>", unsafe_allow_html=True)
                if dtypes[factor] == 'object':
                    # non encoded factor
                    possible = np.unique(encoders[factor].inverse_transform(data[factor].values))
                    Xnew[factor] = str(colsinput[1].selectbox(f"{factor}", possible, key=f"{factor}lm", label_visibility='collapsed'))
                else:
                    Xnew[factor] = colsinput[1].number_input(f"{factor}", 
                                                value=np.mean(data[factor]), key=f"{factor}lm", label_visibility='collapsed')
            # encode the factors if they are categorical
            for i,factor in enumerate(factors):
                if dtypes[factor] == 'object':
                    toencode = Xnew[factor]
                    Xnew[factor] = encoders[factor].transform([toencode])[0]
            # st.write(Xnew)
            right.container(border=True).write(f"<p style='text-align:center;font-size:1.5em'>Predicted {response}:<br><br><b>{st.session_state.modlin.predict(exog=Xnew)[0]:.4g}</b></p>",
                        unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")

with tabs[3]: # machine learning model
    if datafile is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the "Data Loading" tab.""")
    if datafile is not None and len(factors) > 0 and len(response) > 0:
        # Choose machine learning model
        cols = st.columns(3)
        model_sel = cols[0].selectbox("Select the machine learning model:", 
                ["ElasticNetCV", "RidgeCV", "LinearRegression", 
                 "RandomForest", "GaussianProcess", "GradientBoosting"])
        split_size = cols[1].number_input("Validation set size:", 
                                             min_value=0.0, value=0.2,
                                             max_value=1., step=.1)
        features_in_log = cols[2].toggle("Log scale for features importance", value=True)
        if st.button("Compute the machine learning model and plot the results"):
            st.session_state.analysis.model_type = model_sel
            st.session_state.analysis.split_size = split_size
            st.session_state.analysis.compute_ML_model()
            st.session_state.modml = st.session_state.analysis.model
            st.session_state.figml = st.session_state.analysis.plot_ML_model(features_in_log)
        
        if st.session_state.figml is not None:
            # make plot of predicted versus actual
            cols = st.columns([1, 1])
            cols[0].plotly_chart(st.session_state.figml[0])
            if st.session_state.figml[1] is not None:
                cols[1].plotly_chart(st.session_state.figml[1])
            else:
                cols[1].warning(f"The {model_sel} model does not support feature importance.")
            st.write(f"##### Predict the response for a set of factors with this {model_sel} model:")
            Xnew = []
            left, right = st.columns(2)
            for i, factor in enumerate(factors):
                colsinput = left.columns(2)
                colsinput[0].write(f"<p style='text-align:right;font-size:1.1em'><b>{factor}</b></p>", unsafe_allow_html=True)
                if dtypes[factor] == 'object':
                    # Non-encoded factor
                    possible = np.unique(encoders[factor].inverse_transform(data[factor].values))
                    Xnew.append(str(colsinput[1].selectbox(f"{factor}", possible, key=f"{factor}ml", label_visibility='collapsed')))
                else:
                    Xnew.append(colsinput[1].number_input(f"{factor}",
                                                        value=np.mean(data[factor]), key=f"{factor}ml", label_visibility='collapsed'))

            # Encode the factors if they are categorical
            for i, factor in enumerate(factors):
                if dtypes[factor] == 'object':
                    toencode = Xnew[i]
                    Xnew[i] = encoders[factor].transform([toencode])[0]

            # Convert Xnew to a numpy array and reshape
            Xnew = np.array(Xnew).reshape(1, -1)

            # Make prediction
            prediction = st.session_state.modml.predict(Xnew)[0]

            right.container(border=True).write(f"<p style='text-align:center;font-size:1.5em'>Predicted {response}:<br><br><b>{prediction:.4g}</b></p>",
                        unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")