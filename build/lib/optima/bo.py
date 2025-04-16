# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 
import streamlit as st
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeError)

import numpy as np
import pandas as pd
from janitor import clean_names
from typing import Any, Dict, List, Optional, Union

from ax.core.observation import ObservationFeatures, TrialStatus
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.modelbridge.registry import Models
from ax.plot.contour import interact_contour, plot_contour
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.slice import plot_slice
from ax.service.ax_client import AxClient, ObjectiveProperties
import plotly.graph_objects as go

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def ordered_dict_to_dataframe(data):
    """
    Convert an OrderedDict with arbitrary nesting to a DataFrame.
    """
    dflat = flatten_dict(data)
    out = []

    for key, value in dflat.items():
        main_dict = value[0]
        sub_dict = value[1][0]
        out.append([value for value in main_dict.values()] +
                   [value for value in sub_dict.values()])

    df = pd.DataFrame(out, columns=[key for key in main_dict.keys()] +
                                   [key for key in sub_dict.keys()])
    return df

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def read_experimental_data(file_path: str, out_pos=[-1]) -> (Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]):
    """
    Read experimental data from a CSV file and format it into features and outcomes dictionaries.

    Parameters:
    - file_path (str): Path to the CSV file containing experimental data.
    - out_pos (list of int): Column indices of the outcome variables. Default is the last column.

    Returns:
    - Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]: Formatted features and outcomes dictionaries.
    """
    data = pd.read_csv(file_path)
    data = clean_names(data, remove_special=True, case_type='preserve')
    outcome_column_name = data.columns[out_pos]
    features = data.loc[:, ~data.columns.isin(outcome_column_name)].copy()
    outcomes = data[outcome_column_name].copy()

    feature_definitions = {}
    for column in features.columns:
        if features[column].dtype == 'object':
            unique_values = features[column].unique()
            feature_definitions[column] = {'type': 'text',
                                           'range': unique_values.tolist()}
        elif features[column].dtype in ['int64', 'float64']:
            min_val = features[column].min()
            max_val = features[column].max()
            feature_type = 'int' if features[column].dtype == 'int64' else 'float'
            feature_definitions[column] = {'type': feature_type,
                                           'range': [min_val, max_val]}

    formatted_features = {name: {'type': info['type'],
                                 'data': features[name].tolist(),
                                 'range': info['range']}
                          for name, info in feature_definitions.items()}
    # same for outcomes with just type and data
    outcome_definitions = {}
    for column in outcomes.columns:
        if outcomes[column].dtype == 'object':
            unique_values = outcomes[column].unique()
            outcome_definitions[column] = {'type': 'text',
                                           'data': unique_values.tolist()}
        elif outcomes[column].dtype in ['int64', 'float64']:
            min_val = outcomes[column].min()
            max_val = outcomes[column].max()
            outcome_type = 'int' if outcomes[column].dtype == 'int64' else 'float'
            outcome_definitions[column] = {'type': outcome_type,
                                           'data': outcomes[column].tolist()}
    formatted_outcomes = {name: {'type': info['type'],
                                 'data': outcomes[name].tolist()}
                           for name, info in outcome_definitions.items()}
    return formatted_features, formatted_outcomes

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class BOExperiment:
    """
    BOExperiment is a class designed to facilitate Bayesian Optimization experiments using the Ax platform.
    It encapsulates the experiment setup, including features, outcomes, constraints, and optimization methods.

    Attributes
    ----------
    
    features: Dict[str, Dict[str, Any]]
        A dictionary defining the features of the experiment, including their types and ranges.

    outcomes: Dict[str, Dict[str, Any]]
        A dictionary defining the outcomes of the experiment, including their types and observed data.

    N: int
        The number of trials to suggest in each optimization step. Must be a positive integer.

    maximize: Union[bool, List[bool]]
        A boolean or list of booleans indicating whether to maximize the outcomes.
        If a single boolean is provided, it is applied to all outcomes.

    outcome_constraints: Optional[Dict[str, Dict[str, float]]]
        Constraints on the outcomes, specified as a dictionary or list of dictionaries.

    feature_constraints: Optional[List[Dict[str, Any]]]
        Constraints on the features, specified as a list of dictionaries.

    optim: str
        The optimization method to use, either 'bo' for Bayesian Optimization or 'sobol' for Sobol sequence.

    data: pd.DataFrame
        A DataFrame representing the current data in the experiment, including features and outcomes.

    Methods
    -------
    
    initialize_ax_client():
        Initializes the AxClient with the experiment's parameters, objectives, and constraints.

    suggest_next_trials():
        Suggests the next set of trials based on the current model and optimization strategy.
        Returns a DataFrame containing the suggested trials and their predicted outcomes.

    predict(params: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        Predicts the outcomes for a given set of parameters using the current model.
        Returns a list of predicted outcomes for the given parameters.

    update_experiment(params: Dict[str, Any], outcomes: Dict[str, Any]):
        Updates the experiment with new parameters and outcomes, and reinitializes the AxClient.

    plot_model(metricname: Optional[str] = None, slice_values: Optional[Dict[str, Any]]  None, linear: bool = False)`:
        Plots the model's predictions for the experiment's parameters and outcomes.
        If metricname is None, the first outcome metric is used.
        If slice_values is provided, it slices the plot at those values.
        If linear is True, it plots a linear slice plot.
        If the experiment has only one feature, it plots a slice plot.
        If the experiment has multiple features, it plots a contour plot.
        Returns a Plotly figure of the model's predictions.

    plot_optimization_trace(optimum: Optional[float] = None):
        Plots the optimization trace, showing the progress of the optimization over trials.
        If the experiment has multiple outcomes, it raises a warning and returns None.
        Returns a Plotly figure of the optimization trace.

    plot_pareto_frontier():
        Plots the Pareto frontier for multi-objective optimization experiments.
        If the experiment has only one outcome, it raises a warning and returns None.
        Returns a Plotly figure of the Pareto frontier.

    get_best_parameters() -> pd.DataFrame:
        Returns the best parameters found by the optimization process.
        If the experiment has multiple outcomes, it returns a DataFrame of the Pareto optimal parameters.
        If the experiment has only one outcome, it returns a DataFrame of the best parameters and their outcomes.
        The DataFrame contains the best parameters and their corresponding outcomes.

    Properties
    ----------

    features, outcomes, N, maximize, outcome_constraints, feature_constraints, optim, data:
        Getter and setter methods for the respective attributes, with validation to ensure data integrity.
    
    Example
    -------
    >>> features, outcomes = read_experimental_data('data.csv', out_pos=[-2, -1])
    >>> experiment = BOExperiment(features, outcomes, N=5, maximize={'out1':True, 'out2':False})
    >>> experiment.suggest_next_trials()
    >>> experiment.plot_model(metricname='outcome1')
    >>> experiment.plot_model(metricname='outcome2', linear=True)
    >>> experiment.plot_model(metricname='outcome1', slice_values={'feature1': 5})
    >>> experiment.plot_optimization_trace()
    >>> experiment.plot_pareto_frontier()
    >>> experiment.get_best_parameters()
    >>> experiment.update_experiment({'feature1': [4]}, {'outcome1': [0.4]})
    >>> experiment.plot_model()
    >>> experiment.plot_optimization_trace()
    >>> experiment.plot_pareto_frontier()
    >>> experiment.get_best_parameters()
    
    """

    def __init__(self,
                 features: Dict[str, Dict[str, Any]],
                 outcomes: Dict[str, Dict[str, Any]],
                 ranges: Optional[Dict[str, Dict[str, Any]]] = None,
                 N=1,
                 maximize: Union[bool, Dict[str, bool]] = True,
                 fixed_features: Optional[Dict[str, Any]] = None,
                 outcome_constraints: Optional[Dict[str, Dict[str, float]]] = None,
                 feature_constraints: Optional[List[Dict[str, Any]]] = None,
                 optim='bo') -> None:
        """
        Initialize the BOExperiment with features, outcomes, and optimization settings.

        Parameters
        ----------

        features: Dict[str, Dict[str, Any]]
            A dictionary defining the features of the experiment, including their types and ranges.

        outcomes: Dict[str, Dict[str, Any]]
            A dictionary defining the outcomes of the experiment, including their types and observed data.
        
        ranges: Optional[Dict[str, Dict[str, Any]]], optional
            A dictionary defining the ranges of the features. Default is None.
            If not provided, the ranges will be inferred from the features data.
            The ranges should be in the format {'feature_name': [minvalue,maxvalue]}.

        N: int, optional
            The number of trials to suggest in each optimization step. Must be a positive integer. Default is 1.

        maximize: Union[bool, Dict[str, bool]] = True
            A boolean or dict indicating whether to maximize the outcomes.
            If a single boolean is provided, it is applied to all outcomes. Default is True.
        
        fixed_features: Optional[Dict[str, Any]], optional
            A dictionary defining fixed features with their values. Default is None.
            If provided, the fixed features will be treated as fixed parameters in the generation process.
            The fixed features should be in the format {'feature_name': value}.
            The values should be the fixed values for the respective features.

        outcome_constraints: Optional[Dict[str, Dict[str, float]]], optional
            Constraints on the outcomes, specified as a dictionary or list of dictionaries. Default is None.

        feature_constraints: Optional[List[Dict[str, Any]]], optional
            Constraints on the features, specified as a list of dictionaries. Default is None.

        optim: str, optional
            The optimization method to use, either 'bo' for Bayesian Optimization or 'sobol' for Sobol sequence. Default is 'bo'.
        """
        self.first_initialization_done = False
        self.fixed_features      = fixed_features
        self.ranges              = ranges
        self.features            = features
        self.outcomes            = outcomes
        self.N                   = N
        self.maximize            = maximize
        self.outcome_constraints = outcome_constraints
        self.feature_constraints = feature_constraints
        self.optim               = optim
        self.candidate = None
        self.ax_client = None
        self.model = None
        self.parameters = None
        self.generator_run = None
        self.gs = None
        self.initialize_ax_client()
        self.first_initialization_done = True

    @property
    def features(self):
        """
        Get the features of the experiment.
        """
        return self._features

    @features.setter
    def features(self, value):
        """
        Set the features of the experiment with validation.
        """
        if not isinstance(value, dict):
            raise ValueError("features must be a dictionary")
        self._features = value
        self.names = list(value.keys())
        for name in self._features.keys():
            if self.ranges and name in self.ranges.keys():
                self._features[name]['range'] = self.ranges[name]
            else:
                if self._features[name]['type'] == 'text':
                    self._features[name]['range'] = list(set(self._features[name]['data']))
                elif self._features[name]['type'] == 'int':
                    self._features[name]['range'] = [int(np.min(self._features[name]['data'])),
                                                     int(np.max(self._features[name]['data']))]
                elif self._features[name]['type'] == 'float':
                    self._features[name]['range'] = [float(np.min(self._features[name]['data'])),
                                                     float(np.max(self._features[name]['data']))]
        if self.first_initialization_done:
            self.initialize_ax_client()

    @property
    def outcomes(self):
        """
        Get the outcomes of the experiment.
        """
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value):
        """
        Set the outcomes of the experiment with validation.
        """
        if not isinstance(value, dict):
            raise ValueError("outcomes must be a dictionary")
        self._outcomes = value
        self.out_names = list(value.keys())
        if self.first_initialization_done:
            self.initialize_ax_client()
    
    @property
    def fixed_features(self):
        """
        Get the fixed features of the experiment.
        """
        return self._fixed_features

    @outcomes.setter
    def fixed_features(self, value):
        """
        Set the fixed features of the experiment.
        """
        self._fixed_features = None
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("fixed_features must be a dictionary")
            for name in value.keys():
                if name not in self.names:
                    raise ValueError(f"Fixed feature '{name}' not found in features")
            # fixed_features should be an ObservationFeatures object
            self._fixed_features = ObservationFeatures(parameters=value)
        if self.first_initialization_done:
            self.set_gs()

    @property
    def N(self):
        """
        Get the number of trials to suggest in each optimization step.
        """
        return self._N

    @N.setter
    def N(self, value):
        """
        Set the number of trials to suggest in each optimization step with validation.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("N must be a positive integer")
        self._N = value
        if self.first_initialization_done:
            self.set_gs()

    @property
    def maximize(self):
        """
        Get the maximization setting for the outcomes.
        """
        return self._maximize

    @maximize.setter
    def maximize(self, value):
        """
        Set the maximization setting for the outcomes with validation.
        """
        if isinstance(value, bool):
            self._maximize = {out: value for out in self.out_names}
        elif isinstance(value, dict) and len(value) == len(self._outcomes):
            self._maximize = value
        else:
            raise ValueError("maximize must be a boolean or a list of booleans with the same length as outcomes")
        if self.first_initialization_done:
            self.initialize_ax_client()

    @property
    def outcome_constraints(self):
        """
        Get the outcome constraints of the experiment.
        """
        return self._outcome_constraints

    @outcome_constraints.setter
    def outcome_constraints(self, value):
        """
        Set the outcome constraints of the experiment with validation.
        """
        if isinstance(value, dict):
            self._outcome_constraints = [value]
        elif isinstance(value, list):
            self._outcome_constraints = value
        else:
            self._outcome_constraints = None
        if self.first_initialization_done:
            self.initialize_ax_client()

    @property
    def feature_constraints(self):
        """
        Get the feature constraints of the experiment.
        """
        return self._feature_constraints

    @feature_constraints.setter
    def feature_constraints(self, value):
        """
        Set the feature constraints of the experiment with validation.
        """
        if isinstance(value, dict):
            self._feature_constraints = [value]
        elif isinstance(value, list):
            self._feature_constraints = value
        elif isinstance(value, str):
            self._feature_constraints = [value]
        else:
            self._feature_constraints = None
        if self.first_initialization_done:
            self.initialize_ax_client()

    @property
    def optim(self):
        """
        Get the optimization method.
        """
        return self._optim

    @optim.setter
    def optim(self, value):
        """
        Set the optimization method with validation.
        """
        value = value.lower()
        if value not in ['bo', 'sobol']:
            raise ValueError("Optimization method must be either 'bo' or 'sobol'")
        self._optim = value
        if self.first_initialization_done:
            self.set_gs()

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns a DataFrame of the current data in the experiment, including features and outcomes.
        """
        feature_data = {name: info['data'] for name, info in self._features.items()}
        outcome_data = {name: info['data'] for name, info in self._outcomes.items()}
        data_dict = {**feature_data, **outcome_data}
        return pd.DataFrame(data_dict)

    @data.setter
    def data(self, value: pd.DataFrame):
        """
        Sets the features and outcomes data from a given DataFrame.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        feature_columns = [col for col in value.columns if col in self._features]
        outcome_columns = [col for col in value.columns if col in self._outcomes]

        for col in feature_columns:
            self._features[col]['data'] = value[col].tolist()

        for col in outcome_columns:
            self._outcomes[col]['data'] = value[col].tolist()

        if self.first_initialization_done:
            self.initialize_ax_client()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Return a string representation of the BOExperiment instance.
        """
        return f"""
BOExperiment(
    N={self.N},
    maximize={self.maximize},
    outcome_constraints={self.outcome_constraints},
    feature_constraints={self.feature_constraints},
    optim={self.optim}
)

Input data:

{self.data}
        """

    def initialize_ax_client(self):
        """
        Initialize the AxClient with the experiment's parameters, objectives, and constraints.
        """
        print('\n========   INITIALIZING MODEL   ========\n')
        self.ax_client = AxClient(verbose_logging=False, suppress_storage_errors=True)
        self.parameters = []
        for name, info in self._features.items():
            if info['type'] == 'text':
                self.parameters.append({
                    "name": name,
                    "type": "choice",
                    "values": [str(val) for val in info['range']],
                    "value_type": "str"})
            elif info['type'] == 'int':
                self.parameters.append({
                    "name": name,
                    "type": "range",
                    "bounds": [int(np.min(info['range'])),
                               int(np.max(info['range']))],
                    "value_type": "int"})
            elif info['type'] == 'float':
                self.parameters.append({
                    "name": name,
                    "type": "range",
                    "bounds": [float(np.min(info['range'])),
                               float(np.max(info['range']))],
                    "value_type": "float"})

        self.ax_client.create_experiment(
            name="bayesian_optimization",
            parameters=self.parameters,
            objectives={self.out_names[i]:
                ObjectiveProperties(minimize=not self._maximize[self.out_names[i]])
                    for i in range(len(self.out_names))},
            parameter_constraints=self._feature_constraints,
            outcome_constraints=self._outcome_constraints,
            overwrite_existing_experiment=True
        )

        if len(next(iter(self._outcomes.values()))['data']) > 0:
            for i in range(len(next(iter(self._outcomes.values()))['data'])):
                params = {name: info['data'][i] for name, info in self._features.items()}
                outcomes = {name: info['data'][i] for name, info in self._outcomes.items()}
                self.ax_client.attach_trial(params)
                self.ax_client.complete_trial(trial_index=i, raw_data=outcomes)

        self.set_model()
        self.set_gs()

    def set_model(self):
        """
        Set the model to be used for predictions.
        This method is called after initializing the AxClient.
        """
        self.model = Models.BOTORCH_MODULAR(
                experiment=self.ax_client.experiment,
                data=self.ax_client.experiment.fetch_data()
                )
    
    def set_gs(self):
        """
        Set the generation strategy for the experiment.
        This method is called after initializing the AxClient.
        """
        self.clear_trials()
        if self._optim == 'bo':
            if not self.model:
                self.set_model()
            self.gs = GenerationStrategy(
                steps=[GenerationStep(
                            model=Models.BOTORCH_MODULAR,
                            num_trials=-1,  # No limitation on how many trials should be produced from this step
                            max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
                        )
                    ]
                )
        elif self._optim == 'sobol':
            self.gs = GenerationStrategy(
                steps=[GenerationStep(
                            model=Models.SOBOL,
                            num_trials=-1,  # How many trials should be produced from this generation step
                            
                            should_deduplicate=True,  # Deduplicate the trials
                            # model_kwargs={"seed": 165478},  # Any kwargs you want passed into the model
                            model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
                        )
                    ]
                )
        self.generator_run = self.gs.gen(
                experiment=self.ax_client.experiment,  # Ax `Experiment`, for which to generate new candidates
                data=None,  # Ax `Data` to use for model training, optional.
                n=self._N,  # Number of candidate arms to produce
                fixed_features=self._fixed_features, 
                pending_observations=get_pending_observation_features(
                    self.ax_client.experiment
                ),  # Points that should not be re-generated
            )
    
    def clear_trials(self):
        """
        Clear all trials in the experiment.
        """
        # Get all pending trial indices
        pending_trials = [k for k,i in self.ax_client.experiment.trials.items() 
                            if i.status==TrialStatus.CANDIDATE]
        for i in pending_trials:
            self.ax_client.experiment.trials[i].mark_abandoned()
    
    def suggest_next_trials(self, with_predicted=True):
        """
        Suggest the next set of trials based on the current model and optimization strategy.

        Returns
        -------

        pd.DataFrame: 
            DataFrame containing the suggested trials and their predicted outcomes.
        """
        self.clear_trials()
        if self.ax_client is None:
            self.initialize_ax_client()
        if self._N == 1:
            trial = self.ax_client.experiment.new_trial(self.generator_run)
        else:
            trial = self.ax_client.experiment.new_batch_trial(self.generator_run)
        trials = self.ax_client.get_trials_data_frame()
        trials = trials[trials['trial_status'] == 'CANDIDATE']
        trials = trials[[name for name in self.names]]
        if with_predicted:
            topred = [trials.iloc[i].to_dict() for i in range(len(trials))]
            preds = pd.DataFrame(self.predict(topred))
            # add 'predicted_' to the names of the pred dataframe
            preds.columns = [f'Predicted_{col}' for col in preds.columns]
            preds = preds.reset_index(drop=True)
            trials = trials.reset_index(drop=True)
            return pd.concat([trials, preds], axis=1)
        else:
            return trials

    def predict(self, params):
        """
        Predict the outcomes for a given set of parameters using the current model.

        Parameters
        ----------

        params : List[Dict[str, Any]]
            List of parameter dictionaries for which to predict outcomes.

        Returns
        -------

        List[Dict[str, float]]: 
            List of predicted outcomes for the given parameters.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        obs_feats = [ObservationFeatures(parameters=p) for p in params]
        f, _ = self.model.predict(obs_feats)
        return f

    def update_experiment(self, params, outcomes):
        """
        Update the experiment with new parameters and outcomes, and reinitialize the AxClient.

        Parameters
        ----------

        params : Dict[str, Any]
            Dictionary of new parameters to update the experiment with.

        outcomes : Dict[str, Any]
            Dictionary of new outcomes to update the experiment with.
        """
        # append new data to the features and outcomes dictionaries
        for k, v in zip(params.keys(), params.values()):
            if k not in self._features:
                raise ValueError(f"Parameter '{k}' not found in features")
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if not isinstance(v, list):
                v = [v]
            self._features[k]['data'] += v
        for k, v in zip(outcomes.keys(), outcomes.values()):
            if k not in self._outcomes:
                raise ValueError(f"Outcome '{k}' not found in outcomes")
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if not isinstance(v, list):
                v = [v]
            self._outcomes[k]['data'] += v
        self.initialize_ax_client()

    def plot_model(self, metricname=None, slice_values={}, linear=False):
        """
        Plot the model's predictions for the experiment's parameters and outcomes.

        Parameters
        ----------

        metricname : Optional[str]
            The name of the metric to plot. If None, the first outcome metric is used.

        slice_values : Optional[Dict[str, Any]]
            Dictionary of slice values for plotting.

        linear : bool
            Whether to plot a linear slice plot. Default is False.

        Returns
        -------

        plotly.graph_objects.Figure: 
            Plotly figure of the model's predictions.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
            self.suggest_next_trials()

        cand_name = 'Candidate' if self._N == 1 else 'Candidates'
        mname = self.out_names[0] if metricname is None else metricname
        
        param_name = [name for name in self.names if name not in slice_values.keys()]
        par_numeric = [name for name in param_name if self._features[name]['type'] in ['int', 'float']]
        if len(par_numeric)==1:
            fig = plot_slice(
                    model=self.model,
                    metric_name=mname,
                    density=100,
                    param_name=par_numeric[0],
                    generator_runs_dict={cand_name: self.generator_run},
                    slice_values=slice_values
                    )
        elif len(par_numeric)==2:
            fig = plot_contour(
                    model=self.model,
                    metric_name=mname,
                    param_x=par_numeric[0],
                    param_y=par_numeric[1],
                    generator_runs_dict={cand_name: self.generator_run},
                    slice_values=slice_values
                    )
        else:
            fig = interact_contour(
                    model=self.model,
                    generator_runs_dict={cand_name: self.generator_run},
                    metric_name=mname,
                    slice_values=slice_values,
                )

        # Turn the figure into a plotly figure
        plotly_fig = go.Figure(fig.data)

        # Modify only the "In-sample" markers
        trials = self.ax_client.get_trials_data_frame()
        trials = trials[trials['trial_status'] == 'CANDIDATE']
        trials = trials[[name for name in self.names]]
        for trace in plotly_fig.data:
            if trace.type == "contour":  # Check if it's a contour plot
                trace.colorscale = "viridis"  # Apply Viridis colormap
            if 'marker' in trace:  # Modify only the "In-sample" markers
                trace.marker.color = "white"  # Change marker color
                trace.marker.symbol = "circle"  # Change marker style
                trace.marker.size = 10
                trace.marker.line.width = 2
                trace.marker.line.color = 'black'
                trace.text = [t.replace('Arm', '<b>Sample').replace("_0","</b>") for t in trace.text]
            if trace.legendgroup == cand_name:  # Modify only the "Candidate" markers
                trace.marker.color = "red"  # Change marker color
                trace.name = cand_name
                trace.marker.symbol = "x"
                trace.marker.size = 12
                trace.marker.opacity = 1
                # Add hover info
                trace.hoverinfo = "text"  # Enable custom text for hover
                trace.hoverlabel = dict(bgcolor="#f8d5cd", font_color='black')
                trace.text = [t.replace("<i>","").replace("</i>","") for t in trace.text]
                trace.text = [
                    f"<b>Candidate {i+1}</b><br>{'<br>'.join([f'{col}: {val}' for col, val in trials.iloc[i].items()])}"
                    for t in trace.text
                    for i in range(len(trials))
                ]
        plotly_fig.update_layout(
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
            xaxis2=dict(
                showgrid=True,  # Enable grid
                gridcolor="lightgray",  # Light gray grid lines
                zeroline=False,
                zerolinecolor="black",  # Black zero line
                showline=True,
                linewidth=1,
                linecolor="black",  # Black border
                mirror=True
            ),
            yaxis2=dict(
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
        return plotly_fig

    def plot_optimization_trace(self, optimum=None):
        """
        Plot the optimization trace, showing the progress of the optimization over trials.

        Parameters
        ----------

        optimum : Optional[float]
            The optimal value to plot on the optimization trace.

        Returns
        -------

        plotly.graph_objects.Figure: 
            Plotly figure of the optimization trace.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if len(self._outcomes) > 1:
            print("Optimization trace is not available for multi-objective optimization.")
            return None
        fig = self.ax_client.get_optimization_trace(objective_optimum=optimum)
        fig = go.Figure(fig.data)
        for trace in fig.data:
            # add hover info
            trace.hoverinfo = "x+y"
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
            ),
        )
        return fig

    def plot_pareto_frontier(self):
        """
        Plot the Pareto frontier for multi-objective optimization experiments.

        Returns
        -------

        plotly.graph_objects.Figure: 
            Plotly figure of the Pareto frontier.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if len(self._outcomes) < 2:
            print("Pareto frontier is not available for single-objective optimization.")
            return None
        objectives = self.ax_client.experiment.optimization_config.objective.objectives
        frontier = compute_posterior_pareto_frontier(
            experiment=self.ax_client.experiment,
            data=self.ax_client.experiment.fetch_data(),
            primary_objective=objectives[1].metric,
            secondary_objective=objectives[0].metric,
            absolute_metrics=self.out_names,
            num_points=20,
        )
        fig = plot_pareto_frontier(frontier)
        fig = go.Figure(fig.data)
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
            ),
        )
        return fig

    def get_best_parameters(self):
        """
        Return the best parameters found by the optimization process.

        Returns
        -------

        pd.DataFrame: 
            DataFrame containing the best parameters and their outcomes.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if len(self._outcomes) == 1:
            best_parameters = self.ax_client.get_best_parameters()[0]
            best_outcomes = self.ax_client.get_best_parameters()[1]
            best_parameters.update(best_outcomes[0])
            best = pd.DataFrame(best_parameters, index=[0])
        else:
            best_parameters = self.ax_client.get_pareto_optimal_parameters()
            best = ordered_dict_to_dataframe(best_parameters)
        return best

