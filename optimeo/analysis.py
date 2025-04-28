# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 

"""
The analysis module provides tools for data analysis and regression modeling.
The main workhorse is the `DataAnalysis` class, which allows for encoding categorical variables, performing regression analysis, and visualizing results.

It supports both linear regression using the `statsmodels` package and machine learning models from `sklearn`.
The class also provides methods for plotting Q-Q plots, box plots, histograms, and scatter plots.
It includes functionality for bootstrap resampling to estimate the variability of model coefficients.
The `DataAnalysis` class is designed to be flexible and extensible, allowing users to customize the regression analysis process.
"""


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.gofplots import qqplot
# import ML models and scaling functions
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import seaborn as sns


class DataAnalysis:
    """
    This class is used to analyze the data and perform regression analysis.
    
    ### Example
    
    ```python
    from optimeo.analysis import * 

    data = pd.read_csv('dataML.csv')
    factors = data.columns[:-1]
    response = data.columns[-1]
    analysis = DataAnalysis(data, factors, response)
    analysis.model_type = "ElasticNetCV"
    MLmodel = analysis.compute_ML_model()
    figs = analysis.plot_ML_model()
    for fig in figs:
        fig.show()
    ```
    """

    def __init__(self, 
                 data: pd.DataFrame, 
                 factors: list, 
                 response: str, 
                 split_size=.2, 
                 model_type=None):
        self._dtypes = None
        self._encoders = {}
        self._linear_model = None
        self._model = None
        self._equation = ''
        self._data = data
        self._factors = factors
        self._response = response
        self._model_type = model_type
        self._split_size = split_size
        self.encode_data()

    @property
    def data(self):
        """The input `pandas.DataFrame`."""
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self._data = value
    
    def encode_data(self):
        """
        Called during initialization: encodes categorical variables in the data if there are any. 
        Uses `LabelEncoder()` from `sklearn` to convert categorical variables to numerical values.
        """
        self._dtypes = self._data.dtypes
        for factor in self._factors:
            if self._dtypes[factor] == 'object':
                le = LabelEncoder()
                self._encoders[factor] = le
                self._data[factor] = le.fit_transform([str(d) for d in self._data[factor]])

    @property
    def factors(self):
        """The list of names of the columns of the `data` DataFrame that contain factor variables."""
        return self._factors

    @factors.setter
    def factors(self, value):
        if not isinstance(value, list):
            raise ValueError("Factors must be a list.")
        self._factors = value
    
    @property
    def encoders(self):
        """The list of encoders for categorical variables."""
        return self._encoders
    
    @encoders.setter
    def encoders(self, value):
        if not isinstance(value, dict):
            raise ValueError("Encoders must be a dictionary.")
        self._encoders = value

    @property
    def dtypes(self):
        """Get the data types of the columns."""
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value):
        self._dtypes = value

    @property
    def response(self):
        """The name of the column of the `data` DataFrame that contain the response variable."""
        return self._response

    @response.setter
    def response(self, value):
        if not isinstance(value, str):
            raise ValueError("Response must be a string.")
        self._response = value

    @property
    def linear_model(self):
        """Get the linear model."""
        return self._linear_model

    @linear_model.setter
    def linear_model(self, value):
        self._linear_model = value

    @property
    def equation(self):
        """The equation for the linear model, in the form `response ~ var1 + var2 + var1:var2`. This is based on the [statsmodels package](https://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html)."""
        return self._equation

    @equation.setter
    def equation(self, value):
        if not isinstance(value, str):
            raise ValueError("Equation must be a string.")
        self._equation = value

    @property
    def model_type(self):
        """The type of machine learning model to use. Default is None. 
            Must be one of the following: `"ElasticNetCV"`, `"RidgeCV"`,
            `"LinearRegression"`, `"RandomForest"`, `"GaussianProcess"`, `"GradientBoosting"`."""
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        if value is not None and value not in ["ElasticNetCV", "RidgeCV",
                                                "LinearRegression", "RandomForest",
                                                "GaussianProcess", "GradientBoosting"]:
            raise ValueError("Model must be one of the following: "
                             "ElasticNetCV, RidgeCV, LinearRegression, "
                             "RandomForest, GaussianProcess, GradientBoosting.")
        self._model_type = value

    @property
    def model(self):
        """The machine learning model object."""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def split_size(self):
        """The proportion of the dataset to include in the test split. Default is `0.2`."""
        return self._split_size

    @split_size.setter
    def split_size(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Split size must be a number.")
        self._split_size = value

    def __str__(self):
        """Return a string representation of the DataAnalysis object."""
        return (f"DataAnalysis(data={self._data.shape}, "
                f"factors={self._factors}, "
                f"response={self._response}, "
                f"model_type={self._model_type}, "
                f"split_size={self._split_size}, "
                f"encoders={self._encoders})"
                )

    def __repr__(self):
        """Return a string representation of the DataAnalysis object."""
        return self.__str__()

    def plot_qq(self):
        """
        Plot a Q-Q plot for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The Q-Q plot figure.
        """
        qqplot_data = pd.Series(self._data[self._response], copy=True)
        qqplot_data = qqplot(qqplot_data, line='s').gca().lines
        fig = go.Figure()

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[0].get_xdata(),
            'y': qqplot_data[0].get_ydata(),
            'mode': 'markers',
        })

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[1].get_xdata(),
            'y': qqplot_data[1].get_ydata(),
            'mode': 'lines',
            'line': {
                'color': 'red'
            }
        })

        fig.update_layout(
            title='Q-Q Plot',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
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
            height=300,
        )
        return fig

    def plot_boxplot(self):
        """
        Plot a boxplot for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The boxplot figure.
        """
        fig = px.box(self._data, y=self._response, points="all", title='Box Plot')
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
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
            ), height=300,
        )
        return fig

    def plot_histogram(self):
        """
        Plot a histogram for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The histogram figure.
        """
        fig = px.histogram(self._data, x=self._response, title='Histogram')
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
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
            ), height=300,
        )
        return fig

    def plot_scatter_response(self):
        """
        Plot a scatter plot for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The scatter plot figure.
        """
        fig = px.scatter(x=np.arange(1, len(self._data[self._response]) + 1),
                         y=self._data[self._response],
                         labels={'x': 'Measurement number', 'y': self._response},
                         title='Scatter Plot')
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
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
            ), height=300,
        )
        return fig

    def plot_pairplot(self):
        """
        Plot a pairplot for the data.

        Returns
        -------
        fig : seaborn.axisgrid.PairGrid
            The pairplot figure.
        """
        fig = sns.pairplot(
            self._data,
            kind="reg",
            diag_kind="kde",
            plot_kws={"scatter_kws": {"alpha": 0.1}},
        )
        return fig

    def write_equation(self, order=1, quadratic=[]):
        """
        Write R-style equation for multivariate fitting procedure using the statsmodels package.

        Parameters
        ----------
        order : int, optional
            The order of the polynomial. Default is 1.
        quadratic : list, optional
            The list of quadratic factors. Default is an empty list.

        Returns
        -------
        str
            The R-style equation.
        """
        myfactors = self._factors.copy()
        if self._dtypes is not None:
            if not isinstance(myfactors, list):
                myfactors = myfactors.tolist()  # Convert to list to allow mutable operations
            for i in range(len(self._factors)):
                if self._dtypes[self._factors[i]] == 'object':
                    myfactors[i] = f'C({myfactors[i]})'
        eqn = f'{self._response} ~ {myfactors[0]} '
        for factor in myfactors[1:]:
            eqn += f'+ {factor} '
        if order > 1:
            for i in range(len(myfactors)):
                for j in range(i + 1, len(myfactors)):
                    eqn += f'+ {myfactors[i]}:{myfactors[j]} '
        if order > 2:
            for i in range(len(myfactors)):
                for j in range(i + 1, len(myfactors)):
                    for k in range(j + 1, len(myfactors)):
                        eqn += f'+ {myfactors[i]}:{myfactors[j]}:{myfactors[k]} '
        if order > 3:
            for i in range(len(myfactors)):
                for j in range(i + 1, len(myfactors)):
                    for k in range(j + 1, len(myfactors)):
                        for l in range(k + 1, len(myfactors)):
                            eqn += f'+ {myfactors[i]}:{myfactors[j]}:{myfactors[k]}:{myfactors[l]} '
        if len(quadratic) > 0:
            for factor in quadratic:
                eqn += f'+ I({factor}**2) '
        self._equation = eqn
        return self._equation

    def compute_linear_model(self, order=1, quadratic=[]):
        """
        Compute the linear model using the statsmodels package.

        Parameters
        ----------
        order : int, optional
            The order of the polynomial. Default is 1. The parameter 
            is not used if the equation is already set.
        quadratic : list, optional
            The list of quadratic factors. Default is an empty list.
            The parameter is not used if the equation is already set.

        Returns
        -------
        statsmodels.regression.linear_model.RegressionResultsWrapper
            The fitted linear model.
        """
        if self._equation == '':
            eqn = self.write_equation(order=order, quadratic=quadratic)
        else:
            eqn = self._equation
        model = smf.ols(formula=eqn, data=self._data)
        self._linear_model = model.fit()
        return self._linear_model

    def plot_linear_model(self):
        """
        Plot the linear model using plotly.

        Returns
        -------
        fig : list
            The list of plotly figures.
        """
        if self._linear_model is None:
            raise Warning("Linear model has not been computed yet.")
        fig = [None] * 2
        fig[0] = go.Figure()
        # Fig[0]: actual vs predicted line
        fig[0].add_trace(go.Scatter(x=self._data[self._response],
                                    y=self._linear_model.predict(),
                                    mode='markers',
                                    marker=dict(size=12),
                                    name='Actual vs Predicted'))
        # Add 1:1 line
        fig[0].add_shape(type="line",
                      x0=min(self._data[self._response]),
                      y0=min(self._data[self._response]),
                      x1=max(self._data[self._response]),
                      y1=max(self._data[self._response]),
                      line=dict(color="Gray", width=1, dash="dash"))
        fig[0].update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis_title=f'Actual {self._response}',
            yaxis_title=f'Predicted {self._response}',
            height=500,  # Adjust height as needed
            margin=dict(l=10, r=10, t=50, b=50),
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
            )
        )
        # # # # # # # # # # # # # # # # # # # # #
        # Fig[1]: slope values for each factor
        # # # # # # # # # # # # # # # # # # # # #
        # Plot: Slope values for each factor
        res = self._linear_model.params.rename_axis('terms').reset_index()[1:]
        res.columns = ['terms', 'slope']
        error = self._linear_model.bse.rename_axis('terms').reset_index()[1:]
        error.columns = ['terms', 'error']
        res['error'] = error['error']
        res['pvalue'] = [self._linear_model.pvalues[res['terms']].iloc[i] for i in range(len(res))]

        # Sort by p-values
        res = res.sort_values(by='pvalue', ascending=False)
        # Prepare colors and labels
        colors = ['red' if x < 0 else 'green' for x in res['slope']]
        res['slope'] = res['slope'].abs()

        fig[1] = go.Figure()

        # Add bar plot
        fig[1].add_trace(go.Bar(
            y=[term.replace('I(', '').replace('C(', '').replace(')', '').replace(' ** ', '^') for term in res['terms']],
            x=res['slope'],
            error_x=dict(type='data', array=res['error']),
            marker_color=colors,
            orientation='h',
            name='Slope',
            showlegend=False  # Hide the legend entry for the bar trace
        ))

        # Update layout for log scale and labels
        fig[1].update_layout(
            xaxis_title='Magnitude of effect',
            xaxis_type="log",
            height=500,  # Adjust height as needed
            plot_bgcolor="white",  # White background
            legend=dict(
                orientation="h",  # Horizontal orientation
                y=1.1  # Place legend on top
            ),
            margin=dict(l=10, r=150, t=50, b=50),  # Increase right margin for annotations
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
            )
        )

        # Add legend for Negative and Positive
        fig[1].add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='red'),
            name='Negative'
        ))
        fig[1].add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='green'),
            name='Positive'
        ))

        # Add p-values as annotations outside the plot
        for i, p in enumerate(res['slope']):
            fig[1].add_annotation(
                x=1.025,  # Place annotations outside the plot
                y=i,
                text=f"p = {self._linear_model.pvalues[res['terms'].iloc[i]]:.2g}",
                showarrow=False,
                xref="paper",  # Use paper coordinates for x
                xanchor='left'
            )

        return fig

    def compute_ML_model(self, **kwargs):
        """
        Compute the machine learning model.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for the model.

        Returns
        -------
        object
            The fitted machine learning model.
        """
        if self._model_type is None:
            raise ValueError("Model must be provided.")

        X = self._data[self._factors]
        y = self._data[self._response]
        if self._split_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._split_size)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        if self._model_type == "ElasticNetCV":
            self._model = make_pipeline(StandardScaler(),
                                        ElasticNetCV(**kwargs))
        elif self._model_type == "RidgeCV":
            self._model = make_pipeline(StandardScaler(),
                                        RidgeCV(**kwargs))
        elif self._model_type == "LinearRegression":
            self._model = make_pipeline(StandardScaler(),
                                        LinearRegression(**kwargs))
        elif self._model_type == "RandomForest":
            self._model = make_pipeline(StandardScaler(),
                                        RandomForestRegressor(**kwargs))
        elif self._model_type == "GaussianProcess":
            self._model = make_pipeline(StandardScaler(),
                                        GaussianProcessRegressor(**kwargs))
        elif self._model_type == "GradientBoosting":
            self._model = make_pipeline(StandardScaler(),
                                        GradientBoostingRegressor(**kwargs))
        # Fit the self._model
        self._model.fit(X_train, y_train)
        return self._model

    def plot_ML_model(self, features_in_log=False):
        """
        Plot the machine learning model using plotly.

        Parameters
        ----------
        features_in_log : bool, optional
            Whether to plot the feature importances in log scale. Default is False.

        Returns
        -------
        fig : list
            The list of plotly figures.
        """
        coef_names = self._data[self._factors].columns
        X = self._data[self._factors]
        y = self._data[self._response]
        if self._split_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._split_size)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        bootstrap_coefs = bootstrap_coefficients(
            self._model[1], X, y, n_bootstrap=100, random_state=42)
        # Calculate mean and standard deviation of coefficients
        mean_coefs = np.mean(bootstrap_coefs, axis=0)
        std_coefs = np.std(bootstrap_coefs, axis=0)
        # make the fig
        fig = [None] * 2
        fig[0] = go.Figure()
        # Add actual vs predicted line
        fig[0].add_trace(go.Scatter(x=y_train,
                                 y=self._model.predict(X_train),
                                 mode='markers',
                                 marker=dict(size=12, color='royalblue'),
                                 name='Training'))
        if self._split_size > 0:
            fig[0].add_trace(go.Scatter(x=y_test,
                                    y=self._model.predict(X_test),
                                    mode='markers',
                                    marker=dict(size=12, color='orange'),
                                    name='Validation'))
        # Add 1:1 line
        fig[0].add_shape(type="line",
                      x0=min(y_train), y0=min(y_train),
                      x1=max(y_train), y1=max(y_train),
                      line=dict(color="Gray", width=1, dash="dash"))
        fig[0].update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis_title=f'Actual {self._response}',
            yaxis_title=f'Predicted {self._response}',
            height=500,  # Adjust height as needed
            margin=dict(l=10, r=10, t=120, b=50),
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
            )
        )
        # add the score in title
        fig[0].update_layout(
            title={
                'text': f"Predicted vs Actual for {str(self._model[1])}<br>R²_test = {r2_score(y_test, self._model.predict(X_test)):.4f}  -  R²_train = {r2_score(y_train, self._model.predict(X_train)):.4f}<br>RMSE_test = {root_mean_squared_error(y_test, self._model.predict(X_test)):.4f}  -  RMSE_train = {root_mean_squared_error(y_train, self._model.predict(X_train)):.4f}",
                'x': 0.45,
                'xanchor': 'center'
            }
        )
        # make feature importance plot
        if self._model_type != "GaussianProcess":
            fig[1] = go.Figure()
            pos_coefs = mean_coefs[mean_coefs > 0]
            pos_coefs_names = coef_names[mean_coefs > 0]
            neg_coefs = mean_coefs[mean_coefs < 0]
            neg_coefs_names = coef_names[mean_coefs < 0]
            # Add bars for positive mean coefficients
            fig[1].add_trace(go.Bar(
                y=[f"{pos_coefs_names[i]}" for i in range(len(pos_coefs))],
                x=mean_coefs[mean_coefs > 0],
                error_x=dict(type='data', array=std_coefs[mean_coefs > 0], visible=True),
                orientation='h',
                marker_color='royalblue',
                name='Positive'
            ))
            fig[1].add_trace(go.Bar(
                y=[f"{neg_coefs_names[i]}" for i in range(len(neg_coefs))],
                x=-mean_coefs[mean_coefs < 0],
                error_x=dict(type='data', array=std_coefs[mean_coefs < 0], visible=True),
                orientation='h',
                marker_color='orange',
                name='Negative'
            ))
            # Update layout
            fig[1].update_layout(
                title={
                    'text': f"Features importance for {str(self._model[1])}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Coefficient Value",
                yaxis_title="Features",
                barmode='relative',
                margin=dict(l=150)
            )
            if features_in_log:
                fig[1].update_xaxes(type="log")
            else:
                fig[1].update_xaxes(type="linear")
            fig[1].update_layout(
                plot_bgcolor="white",  # White background
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=500,  # Adjust height as needed
                margin=dict(l=10, r=10, t=120, b=50),
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
                )
            )
        else:
            print('GaussianProcess does not have coefficients or feature importances')
        return fig

    def predict(self, X=None, model='all'):
        """
        Predict using the machine learning model and the linear model,
        if they are trained. Use the encoders to encode the data.
        If X is not provided, use the original data.
        If the model has not been trained, raise a warning.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The input features. Default is None, which uses the original data.

        Returns
        -------
        pd.DataFrame
            The predicted values with a column indicating the model used.
        """
        if X is None:
            X = self._data[self._factors].copy()
        else:
            X = X.copy()

        # Encode the input data using the same encoders
        for factor in self._factors:
            if factor in self._encoders:
                X[factor] = self._encoders[factor].transform(X[factor].astype(str))

        predML = None
        predlin = None

        if self._model is not None:
            # Check if the model is fitted
            if not hasattr(self._model, 'predict'):
                raise Warning("Model has not been trained yet.")
            predML = self._model.predict(X)
            predML = pd.DataFrame(predML, columns=['prediction'])
            predML['model'] = self._model_type

        if self._linear_model is not None:
            # Check if the linear model is fitted
            if not hasattr(self._linear_model, 'predict'):
                raise Warning("Linear model has not been trained yet.")
            predlin = self._linear_model.predict(X)
            predlin = pd.DataFrame(predlin, columns=['prediction'])
            predlin['model'] = 'Linear Model'

        if predML is not None and predlin is not None and model == 'all':
            # Combine the predictions on top of each other
            pred = pd.concat([predML, predlin], axis=0)
            pred = pred.reset_index(drop=True)
            return pred
        elif predML is not None and model == 'ML':
            return predML
        elif predlin is not None and model == 'linear':
            return predlin
        else:
            raise Warning("No model has been trained yet.")




def bootstrap_coefficients(mod, X, y, n_bootstrap=100, random_state=None):
    """
    Perform bootstrap resampling to estimate the variability of model coefficients.

    Parameters
    ----------
    mod : object
        The machine learning model.
    X : pd.DataFrame
        The input features.
    y : pd.Series
        The target variable.
    n_bootstrap : int, optional
        The number of bootstrap samples. Default is 100.
    random_state : int, optional
        The seed for the random number generator.

    Returns
    -------
    results : np.ndarray
        The bootstrapped coefficients.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    results = []

    for _ in range(n_bootstrap):
        # Resample the data
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        X_resample, y_resample = X.values[indices], y.values[indices]

        # Fit the model
        if isinstance(mod, RidgeCV):
            model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_resample, y_resample)
            results.append(model.coef_)
        elif isinstance(mod, ElasticNetCV):
            model = ElasticNetCV(alphas=np.logspace(-3, 3, 10), l1_ratio=0.5).fit(X_resample, y_resample)
            results.append(model.coef_)
        elif isinstance(mod, LinearRegression):
            model = LinearRegression().fit(X_resample, y_resample)
            results.append(model.coef_)
        elif isinstance(mod, RandomForestRegressor):
            model = RandomForestRegressor().fit(X_resample, y_resample)
            results.append(model.feature_importances_)
        elif isinstance(mod, GradientBoostingRegressor):
            model = GradientBoostingRegressor().fit(X_resample, y_resample)
            results.append(model.feature_importances_)
        elif isinstance(mod, GaussianProcessRegressor):
            # Gaussian Process does not have coefficients or feature importances
            model = GaussianProcessRegressor().fit(X_resample, y_resample)
            results.append(np.zeros(X.shape[1]))  # Placeholder for Gaussian Process
        else:
            raise ValueError(f"Unsupported model type: {mod}")

    return np.array(results)