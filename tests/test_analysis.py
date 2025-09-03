# test_analysis.py
import sys
import os

# Add the parent directory of optimeo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
import pandas as pd
from optimeo.analysis import DataAnalysis

data = pd.read_csv("tests/data/dataML.csv")

models_list = ["ElasticNetCV", "RidgeCV", "LinearRegression", 
               "RandomForest", "GaussianProcess", "GradientBoosting"]

def test_initialization():
    """Test if the class initializes correctly with valid inputs."""
    factors = data.columns[:-1].tolist()
    response = data.columns[-1]
    for mod in models_list:
        analysis = DataAnalysis(data, factors, response)
        analysis.model_type = mod
        MLmodel = analysis.compute_ML_model()
        analysis.encode_data()
        eq = analysis.write_equation(order=1, quadratic=[])
        assert len(eq) > 0
        linmod = analysis.compute_linear_model(order=1, quadratic=[])
        assert linmod is not None
        MLmodel = analysis.compute_ML_model()
        assert MLmodel is not None
        fig = analysis.plot_qq()
        assert fig is not None
        fig = analysis.plot_boxplot()
        assert fig is not None
        fig = analysis.plot_histogram()
        assert fig is not None
        fig = analysis.plot_scatter_response()
        assert fig is not None
        fig = analysis.plot_corr()
        assert fig is not None
        fig = analysis.plot_pairplot_seaborn()
        assert fig is not None
        fig = analysis.plot_pairplot_plotly()
        assert fig is not None
        fig = analysis.plot_ML_model(features_in_log=False)
        assert fig is not None
        fig = analysis.plot_linear_model()
        assert fig is not None

def test_invalid_model_type():
    """Test if an invalid model type raises an error."""
    factors = data.columns[:-1].tolist()
    response = data.columns[-1]
    analysis = DataAnalysis(data, factors, response)
    with pytest.raises(ValueError):
        analysis.model_type = "InvalidModel"

