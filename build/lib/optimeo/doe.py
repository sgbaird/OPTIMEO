# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeError)
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from dexpy.optimal import build_optimal
from dexpy.model import ModelOrder
from dexpy.design import coded_to_actual
from doepy import build
from sklearn.preprocessing import LabelEncoder
from pyDOE3 import *
import definitive_screening_design as dsd
import plotly.express as px
from itertools import combinations
import plotly.graph_objects as go

class DesignOfExperiments:
    """
    Class to create a design of experiments (DoE) for a given model.
    This class allows the user to specify the type of design, the parameters,
    and various options for the design generation.
    The design can be visualized using scatter plots.
    
    Parameters
    ----------
    type : str
        The type of design to create. Must be one of:
        'Full Factorial', 'Sobol sequence', 'Fractional Factorial',
        'Definitive Screening Design', 'Space Filling Latin Hypercube',
        'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman',
        'Box-Behnken'.
    parameters : List[Dict[str, Dict[str, Any]]]
        List of parameters for the design, each with a dictionary of properties.
        Each dictionary should contain 'name', 'type', and 'values'.
        'values' should be a list of possible values for the parameter.
        'type' should be either "int", "integer", "float", "<other>". 
        Any <other> will be considered as "categorical".
        'values' should be a list of possible values for the parameter.
    Nexp : int, optional
        Number of experiments in the design, when applicable. Default is 4.
    order : int, optional
        Order of the model (for 'Optimal' design). Default is 2.
    randomize : bool, optional
        Whether to randomize the run order. Default is True.
    reduction : int, optional
        Reduction factor for 'Fractional Factorial' designs. Default is 2.
    
    Attributes
    ----------
    type : str
        The type of design. Must be one of:
        'Full Factorial', 'Sobol sequence', 'Fractional Factorial',
        'Definitive Screening Design', 'Space Filling Latin Hypercube',
        'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman',
        'Box-Behnken'.
    parameters : List[Dict[str, Dict[str, Any]]]
        The parameters for the design.
    Nexp : int
        Number of experiments in the design.
    order : int
        Order of the model.
    randomize : bool
        Whether to randomize the run order.
    reduction : int
        Reduction factor for 'Fractional Factorial' designs.
    design : pd.DataFrame
        The design DataFrame.
    lows : Dict[str, float]
        Lower bounds for the parameters.
    highs : Dict[str, float]
        Upper bounds for the parameters.
    
    Methods
    -------
    create_design()
        Create the design of experiments based on the specified type and parameters.
    plot()
        Plot the design of experiments.
    
    Examples
    --------
    >>> from doe import DesignOfExperiments
    >>> parameters = [
    ...     {'name': 'Temperature', 'type': 'integer', 'values': [20, 30, 40]},
    ...     {'name': 'Pressure', 'type': 'float', 'values': [1, 2, 3]},
    ...     {'name': 'Catalyst', 'type': 'categorical', 'values': ['A', 'B', 'C']}
    ... ]
    >>> doe = DesignOfExperiments(
    ...     type='Full Factorial',
    ...     parameters=parameters
    ... )
    >>> design = doe.design
    >>> print(design)
    >>> figs = doe.plot()
    >>> for fig in figs:
    ...     fig.show()
    

    """

    def __init__(self, type: str,
                 parameters: List[Dict[str, Dict[str, Any]]],
                 Nexp: int = 4, order: int = 2, 
                 randomize: bool = True, reduction: int = 2):
        """
        Initialize the DesignOfExperiments class.

        Parameters
        ----------
        type : str
            The type of design to create. Must be one of:
            'Full Factorial', 'Sobol sequence', 'Fractional Factorial',
            'Definitive Screening Design', 'Space Filling Latin Hypercube',
            'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman',
            'Box-Behnken'.
        parameters : List[Dict[str, Dict[str, Any]]]
            List of parameters for the design, each with a dictionary of properties.
            Each dictionary should contain 'name', 'type', and 'values'.
            'values' should be a list of possible values for the parameter.
            'type' should be either "int", "integer", "float", "<other>". 
            Any <other> will be considered as "categorical".
            'values' should be a list of possible values for the parameter.
        Nexp : int, optional
            Number of experiments in the design, when applicable. Default is 4.
        order : int, optional
            Order of the model (for 'Optimal' design). Default is 2.
        randomize : bool, optional
            Whether to randomize the run order. Default is True.
        reduction : int, optional
            Reduction factor for 'Fractional Factorial' designs. Default is 2.
        """
        self._type = type
        self._parameters = parameters
        self._Nexp = Nexp
        self._order = order
        self._randomize = randomize
        self._reduction = reduction
        self._design = None
        self._lows = {}
        self._highs = {}
        self.create_design()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Return a string representation of the DesignOfExperiments instance.
        """
        return f"""
- Design of Experiments type: {self.type}
- Parameters:
{"\n".join([str(par) for par in self.parameters])}
- Lows: {self._lows}
- Highs: {self._highs}
- If applicable:
    - Randomize: {self.randomize}
    - Number of Experiments: {self.Nexp}
    - Order: {self.order}
    - Reduction: {self.reduction}
- Design:
{self.design}
"""

    @property
    def type(self) -> str:
        """Get the type of design."""
        return self._type

    @type.setter
    def type(self, value: str):
        """Set the type of design."""
        self._type = value

    @property
    def parameters(self) -> List[Dict[str, Dict[str, Any]]]:
        """Get the parameters for the design."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: List[Dict[str, Dict[str, Any]]]):
        """Set the parameters for the design."""
        self._parameters = value

    @property
    def Nexp(self) -> int:
        """Get the number of experiments."""
        return self._Nexp

    @Nexp.setter
    def Nexp(self, value: int):
        """Set the number of experiments."""
        self._Nexp = value

    @property
    def order(self) -> int:
        """Get the order of the model."""
        return self._order

    @order.setter
    def order(self, value: int):
        """Set the order of the model."""
        self._order = value

    @property
    def randomize(self) -> bool:
        """Get the randomize flag."""
        return self._randomize

    @randomize.setter
    def randomize(self, value: bool):
        """Set the randomize flag."""
        self._randomize = value

    @property
    def reduction(self) -> int:
        """Get the reduction factor."""
        return self._reduction

    @reduction.setter
    def reduction(self, value: int):
        """Set the reduction factor."""
        self._reduction = value

    @property
    def design(self) -> pd.DataFrame:
        """Get the design DataFrame."""
        return self._design

    def create_design(self):
        """
        Create the design of experiments based on the specified type and parameters.
        """
        for par in self.parameters:
            if par['type'].lower() == "categorical":
                if self.type != 'Sobol sequence':
                    le = LabelEncoder()
                    label = le.fit_transform(par['values'])
                    par['values'] = label
                    par['encoder'] = le
                    self._lows[par['name']] = np.min(par['values'])
                    self._highs[par['name']] = np.max(par['values'])
                else:
                    par['encoder'] = None
            else:
                self._lows[par['name']] = np.min(par['values'])
                self._highs[par['name']] = np.max(par['values'])

        pars = {par['name']: par['values'] for par in self.parameters}

        if self.type == 'Full Factorial':
            self._design = build.full_fact(pars)
        elif self.type == 'Sobol sequence':
            from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
            from ax.modelbridge.registry import Models
            from ax.service.ax_client import AxClient, ObjectiveProperties

            ax_client = AxClient()
            params = []
            for par in self.parameters:
                if par['type'].lower() == "float":
                    params.append({'name': par['name'],
                                   'type': 'range',
                                   'value_type': 'float',
                                   'bounds': [float(np.min(par['values'])), float(np.max(par['values']))]})
                elif par['type'].lower() in ["integer", 'int']:
                    params.append({'name': par['name'],
                                   'type': 'range',
                                   'value_type': 'int',
                                   'bounds': [int(np.min(par['values'])), int(np.max(par['values']))]})
                else:
                    params.append({'name': par['name'],
                                   'type': 'choice',
                                   'values': par['values']})

            ax_client.create_experiment(
                name="DOE",
                parameters=params,
                objectives={"response": ObjectiveProperties(minimize=False)}
            )
            gs = GenerationStrategy(
                steps=[GenerationStep(
                    model=Models.SOBOL,
                    num_trials=-1,
                    should_deduplicate=True,
                    model_kwargs={"seed": 165478},
                    model_gen_kwargs={},
                )]
            )
            generator_run = gs.gen(
                experiment=ax_client.experiment,
                data=None,
                n=self.Nexp
            )
            if self.Nexp == 1:
                ax_client.experiment.new_trial(generator_run)
            else:
                ax_client.experiment.new_batch_trial(generator_run)
            trials = ax_client.get_trials_data_frame()
            self._design = trials[trials['trial_status'] == 'CANDIDATE']
            self._design = self._design.drop(columns=['trial_index',
                                                      'trial_status',
                                                      'arm_name',
                                                      'generation_method',
                                                      'generation_node'])
        elif self.type == 'Fractional Factorial':
            for par in range(len(self.parameters)):
                if self.parameters[par]['type'] == "Numerical":
                    self.parameters[par]['type'] = "Categorical"
                    le = LabelEncoder()
                    label = le.fit_transform(self.parameters[par]['values'])
                    self.parameters[par]['values'] = label
                    self.parameters[par]['encoder'] = le
            design = gsd([len(par['values']) for par in self.parameters], self.reduction)
            self._design = pd.DataFrame(design, columns=[par['name'] for par in self.parameters])
        elif self.type == 'Definitive Screening Design':
            params = {par['name']: [np.min(par['values']), np.max(par['values'])] for par in self.parameters}
            self._design = dsd.generate(factors_dict=params)
        elif self.type == 'Space Filling Latin Hypercube':
            self._design = build.space_filling_lhs(pars, num_samples=self.Nexp)
        elif self.type == 'Randomized Latin Hypercube':
            self._design = build.lhs(pars, num_samples=self.Nexp)
        elif self.type == 'Optimal':
            reaction_design = build_optimal(
                len(self.parameters),
                order=ModelOrder(self.order),
                run_count=self.Nexp)
            reaction_design.columns = [par['name'] for par in self.parameters]
            self._design = coded_to_actual(reaction_design, self._lows, self._highs)
        elif self.type == 'Plackett-Burman':
            self._design = build.plackett_burman(pars)
        elif self.type == 'Box-Behnken':
            if len(self.parameters) < 3 or any([len(par['values']) < 3 for par in self.parameters]):
                self._design = pd.DataFrame({})
                raise Warning("Box-Behnken design is not possible with less than 3 parameters and with less than 3 levels for any parameter.")
            else:
                self._design = build.box_behnken(d=pars, center=1)
        else:
            raise Warning("Unknown design type. Must be one of: 'Full Factorial', 'Sobol sequence', 'Fractional Factorial', 'Definitive Screening Design', 'Space Filling Latin Hypercube', 'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman', 'Box-Behnken'.")

        for par in self.parameters:
            if par['type'] == "Categorical" and self.type != 'Sobol sequence':
                vals = self._design[par['name']].to_numpy()
                self._design[par['name']] = par['encoder'].inverse_transform([int(v) for v in vals])

        # randomize the run order
        self._design['run_order'] = np.arange(len(self._design)) + 1
        if self.randomize:
            ord = self._design['run_order'].to_numpy()
            self._design['run_order'] = np.random.permutation(ord)
        cols = self._design.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self._design = self._design[cols]
        # apply the column types
        for col in self._design.columns:
            for par in self.parameters:
                if col == par['name']:
                    if par['type'].lower() == "float":
                        self._design[col] = self._design[col].astype(float)
                    elif par['type'].lower() in ["int", "integer"]:
                        self._design[col] = self._design[col].astype(int)
                    else:
                        self._design[col] = self._design[col].astype(str)
        return self._design

    def plot(self):
        """
        Plot the design of experiments.

        Returns
        -------
        List of plotly.graph_objs._figure.Figure
            A list of Plotly figures representing the design of experiments.
        """
        fig = []
        count = 0
        if len(self.design) > 0:
            if len(self.parameters) <= 2:
                # Create 2D scatter plots
                for i, faci in enumerate(self.parameters):
                    for j, facj in enumerate(self.parameters):
                        if j > i:
                            fig.append(px.scatter(
                                self.design,
                                x=facj['name'],
                                y=faci['name'],
                                title=f"""{faci['name']} vs {facj['name']}""",
                                labels={facj['name']: facj['name'], faci['name']: faci['name']}
                            ))
                            fig[count].update_traces(marker=dict(size=10))
                            fig[count].update_layout(
                                margin=dict(l=10, r=10, t=50, b=50),
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor="lightgray",
                                    zeroline=False,
                                    showline=True,
                                    linewidth=1,
                                    linecolor="black",
                                    mirror=True
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor="lightgray",
                                    zeroline=False,
                                    showline=True,
                                    linewidth=1,
                                    linecolor="black",
                                    mirror=True
                                ),
                            )
                            count += 1
            else:
                # Create 3D scatter plots
                for k, (faci, facj, fack) in enumerate(combinations(self.parameters, 3)):
                    fig.append(go.Figure(data=[go.Scatter3d(
                        x=self.design[facj['name']],
                        y=self.design[faci['name']],
                        z=self.design[fack['name']],
                        mode='markers',
                        marker=dict(size=10, color='royalblue', opacity=0.7),
                    )]))
                    fig[count].update_layout(
                        template='ggplot2',
                        height=500,
                        width=500,
                        scene=dict(
                            xaxis_title=facj['name'],
                            yaxis_title=faci['name'],
                            zaxis_title=fack['name'],
                        ),
                        title=f"{faci['name']} vs {facj['name']}<br>vs {fack['name']}",
                        margin=dict(l=10, r=10, t=50, b=50)
                    )
                    count += 1
        return fig

