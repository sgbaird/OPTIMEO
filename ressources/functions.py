# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 


import streamlit as st
from pathlib import Path
import sys
from io import StringIO 
import pandas as pd
import numpy as np
from janitor import clean_names
import pickle
import os
from sklearn.preprocessing import LabelEncoder

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def read_markdown_file(markdown_file):
    """Read a md file and return the content"""
    return Path(markdown_file).read_text()

def writeout(df: pd.DataFrame, format='csv'):
    """Write a pd.DataFrame to csv. To use with st.download_button()"""
    df = clean_names(df)
    return df.to_csv(index=False).encode('utf-8')

def write_poly(pp):
    signs = [np.sign(x) for x in pp]
    p = np.array([f"{x:.2e}" for x in pp])
    p = [x.split("e") for x in p]
    p = [f"{x[0]}•10<sup>{int(x[1])}</sup>" for x in p]
    p = [x.replace(f'•10<sup>{0}</sup>', '') for x in p]
    out = "y = "
    order = len(p) - 1
    for i,n in zip(range(len(p)), range(order, -1, -1)):
        if i==0 and n>1:
            out += f"{p[0]}x<sup>{n}</sup>"
        elif i==0 and n==1:
            out += f"{p[0]}x"
        elif i==0 and n==0:
            out += f"{p[0]}"
        elif n == 1:
            out += f"{'' if signs[i]<0 else ' +'} {p[i]}x"
        elif n == 0:
            out += f"{'' if signs[i]<0 else ' +'} {p[i]}"
        else:
            out += f"{'' if signs[i]<0 else ' +'} {p[i]}x<sup>{n}</sup>"
    out = out.replace('-', '–')
    return(out)

def write_equation(factors, response, order=1, quadratic=[], dtypes=None):
    """Write R-style equation for multivariate fitting procedure using the statsmodels package"""
    myfactors = factors.copy()
    if dtypes is not None:
        print(dtypes)
        for i in range(len(factors)):
            if dtypes[factors[i]] == 'object':
                myfactors[i] = f'C({myfactors[i]})'
    eqn = f'{response} ~ {myfactors[0]} '
    for factor in myfactors[1:]:
        eqn += f'+ {factor} ' 
    if order>1:
        for i in range(len(myfactors)): 
            for j in range(i + 1, len(myfactors)): 
                eqn += f'+ {myfactors[i]}:{myfactors[j]} '
    if order>2:
        for i in range(len(myfactors)): 
            for j in range(i + 1, len(myfactors)): 
                for k in range(j + 1, len(myfactors)): 
                    eqn += f'+ {myfactors[i]}:{myfactors[j]}:{myfactors[k]} '
    if order>3:
        for i in range(len(myfactors)): 
            for j in range(i + 1, len(myfactors)): 
                for k in range(j + 1, len(myfactors)): 
                    for l in range(k + 1, len(myfactors)): 
                        eqn += f'+ {myfactors[i]}:{myfactors[j]}:{myfactors[k]}:{myfactors[l]} '
    if order>4:
        st.warning("Only orders below 5 can be consedired.")
    if len(quadratic)>0:
        for factor in quadratic:
            eqn += f'+ I({factor}**2) '
    return eqn

about_items={
        'Get Help': 'mailto:colin.bousige@cnrs.fr',
        'Report a bug': "mailto:colin.bousige@cnrs.fr",
        'About': """
        ## DOE-DOA
        Version date 2024-10-01.

        This app was made by [Colin Bousige](https://lmi.cnrs.fr/author/colin-bousige/). Contact me for support, requests, or to signal a bug.
        """
    }



def train_model(X_train, y_train, model, model_name):
    model.fit(X_train, y_train)
    # saving the trained model
    out = f"./trained_model/{model_name}.pkl"
    out = out.replace(" ", "_")
    with open(out, 'wb') as file:
        pickle.dump(model, file)
    return model


@st.cache_resource
def load_model(model_name):
    infile = f"{model_name}".replace(" ", "_")
    print(f"./trained_model/{infile}")
    if infile in os.listdir("./trained_model"):
        with open(f"./trained_model/{infile}", 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"The model {infile} was not found. Please train the model first.")
        return None


def encode_data(data, factors, response):
    """
    Read experimental data from a CSV file and format it into features and outcomes dictionaries.

    Parameters:
    - file_path (str): Path to the CSV file containing experimental data.
    - out_pos (list of int): Column indices of the outcome variables. Default is the last column.

    Returns:
    - Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]: Formatted features and outcomes dictionaries.
    """
    features = data[factors].copy()
    outcomes = data[response].copy()

    formatted_features = {}
    for column in features.columns:
        if features[column].dtype == 'object':
            unique_values = features[column].unique()
            formatted_features[column] = {'type': 'text',
                                          'data': [str(val) for val in features[column].tolist()],
                                          'range': [str(val) for val in unique_values.tolist()]}
        elif 'int' in str(features[column].dtype):
            min_val = int(features[column].min())
            max_val = int(features[column].max())
            formatted_features[column] = {'type': 'int',
                                          'data': [int(val) for val in features[column].tolist()],
                                          'range': (min_val, max_val)}
        elif 'float' in str(features[column].dtype):
            min_val = float(features[column].min())
            max_val = float(features[column].max())
            formatted_features[column] = {'type': 'float',
                                          'data': [float(val) for val in features[column].tolist()],
                                          'range': (min_val, max_val)}

    # same for outcomes with just type and data
    formatted_outcomes = {}
    for column in outcomes.columns:
        if outcomes[column].dtype == 'object':
            unique_values = outcomes[column].unique()
            formatted_outcomes[column] = {'type': 'text',
                                          'data': unique_values.tolist()}
        elif outcomes[column].dtype in ['int64', 'float64']:
            min_val = outcomes[column].min()
            max_val = outcomes[column].max()
            outcome_type = 'int' if outcomes[column].dtype == 'int64' else 'float'
            formatted_outcomes[column] = {'type': outcome_type,
                                          'data': outcomes[column].tolist()}

    return formatted_features, formatted_outcomes

def encode_data2(data, factors):
    dtypes = data.dtypes
    encoders = {}
    for factor in factors:
        if dtypes[factor] == 'object':
            le = LabelEncoder()
            encoders[factor] = le
            data[factor] = le.fit_transform([str(d) for d in data[factor]])
    return data, encoders, dtypes


# def decode_data(data, factors, dtypes, encoders):
#     for factor in factors:
#         if dtypes[factor] == 'object':
#             data[factor] = encoders[factor].inverse_transform([round(f) for f in data[factor]])
#     return data


def clear_models():
    st.cache_resource.clear()
    available_models = os.listdir("./trained_model")
    for model_name in available_models:
        os.remove(f"./trained_model/{model_name}")
        


# Function to check constraints
def check_constraints(df, constraints):
    results = {}
    for constraint in constraints:
        # Evaluate the constraint as a boolean expression
        results[constraint] = df.eval(constraint)
    return results