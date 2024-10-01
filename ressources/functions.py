import streamlit as st
from pathlib import Path
import sys
from io import StringIO 
import pandas as pd
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

def writeout(df: pd.DataFrame):
    """Write a pd.DataFrame to csv. To use with st.download_button()"""
    df = clean_names(df)
    return df.to_csv(index=False).encode('utf-8')

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
        'Get Help': 'https://lmi.cnrs.fr/author/colin-bousige/',
        'Report a bug': "https://lmi.cnrs.fr/author/colin-bousige/",
        'About': """
        ## MOFSONG Optimizer
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


def encode_data(data, factors):
    dtypes = data.dtypes
    encoders = {}
    for factor in factors:
        if dtypes[factor] == 'object':
            le = LabelEncoder()
            encoders[factor] = le
            data[factor] = le.fit_transform(data[factor])
    return data, encoders, dtypes


def decode_data(data, factors, dtypes, encoders):
    for factor in factors:
        if dtypes[factor] == 'object':
            data[factor] = encoders[factor].inverse_transform([round(f) for f in data[factor]])
    return data


def clear_models():
    st.cache_resource.clear()
    available_models = os.listdir("./trained_model")
    for model_name in available_models:
        os.remove(f"./trained_model/{model_name}")