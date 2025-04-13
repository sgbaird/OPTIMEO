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
from ressources.functions import *
from ressources.bo import *
import pandas as pd
import numpy as np
from ressources.functions import about_items

st.set_page_config(page_title="New experiments – Bayesian Optimisation",
                   page_icon="📈", layout="wide", menu_items=about_items)

style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

if "bo" not in st.session_state:
    st.session_state['bo'] = None
if "next" not in st.session_state:
    st.session_state['next'] = None
if "best" not in st.session_state:
    st.session_state['best'] = None
if "model_up_to_date" not in st.session_state:
    st.session_state['model_up_to_date'] = False
if "plot_up_to_date" not in st.session_state:
    st.session_state['plot_up_to_date'] = False

def model_changed():
    st.session_state.model_up_to_date = False
    st.session_state.plot_up_to_date = False
def model_updated():
    st.session_state.model_up_to_date = True
def plot_changed():
    st.session_state.plot_up_to_date = False
def plot_updated():
    st.session_state.plot_up_to_date = True


# if "data" not in st.session_state:
#     st.session_state['data'] = None
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# New experiments – Bayesian Optimisation
""")


tabs = st.tabs(["Data Loading", "Bayesian Optimization", 'Predictions'])

with tabs[0]:
    colos = st.columns([2,3])
    dataf = colos[0].file_uploader("""Upload data file (csv, xls, xlsx, xlsm, xlsb, odf, ods and odt).

For Excel-like files, make sure the data start in the A1 cell.""", type=["csv",'xlsx','xls', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'],
                help="The data file should contain the factors and the response variable.",
                on_change=model_changed)
    if dataf is None:
        colos[1].markdown(
        "⚠️ The data must be in tidy format, meaning that each column is a variable and each row is an observation. We usually place the factors in the first columns and the response(s) in the last column(s). You can specify up to two responses. Try to avoid spaces and special characters in the column names. The first row of the file will be used as the header."
        )
        colos[1].image("ressources/tidy_data.jpg", caption="Example of tidy data format")
    if dataf is not None:
        if Path(dataf.name).suffix == '.csv':
            data = pd.read_csv(dataf)
        else:
            data = pd.read_excel(dataf)
        data = clean_names(data)
        left, right = st.columns([3,2])
        cols = data.columns.to_numpy()
        colos[1].dataframe(data, hide_index=False)
        mincol = 1 if 'run_order' in cols else 0
        factors = colos[0].multiselect("Select the **factors** columns:", 
                data.columns, default=cols[mincol:-1],
                on_change=model_changed)
        # response cannot be a factor, so default are all unselected columns in factor
        available = [col for col in cols if col not in factors]
        responses = colos[0].multiselect("Select the **response** column:", 
                available, max_selections=2, default=available[-1],
                on_change=model_changed)
        # add option to change type of columns
        dtypesF = data[factors].dtypes
        placeholder = st.empty()
        st.write("""##### Select the type and range of each factor
Except for categorical factors, you can increase the ranges to allow the optimization algorithm to explore values outside the current range of measures.""")
        factor_types = {factor: dtypesF[factor] for factor in factors}
        factor_ranges = {factor: [np.min(data[factor]), np.max(data[factor])] for factor in factors}
        type_choice = {'object':0, 'int64':1, 'float64':2}
        colos = st.columns(5)
        colos[1].write("<p style='text-align:center;'><b>Type</b></p>", unsafe_allow_html=True)
        colos[2].write("<p style='text-align:center;'><b>Min</b></p>", unsafe_allow_html=True)
        colos[3].write("<p style='text-align:center;'><b>Max</b></p>", unsafe_allow_html=True)
        for factor in factors:
            colos = st.columns(5)
            colos[0].write(f"<p style='text-align:right;'><b>{factor}</b></p>", unsafe_allow_html=True)
            factype = type_choice[f"{factor_types[factor]}"]
            factor_types[factor] = colos[1].selectbox(f"Type of **{factor}**", 
                ['Categorical', 'Integer', 'Float'], key=f"type_{factor}", 
                index = factype, label_visibility='collapsed', on_change=model_changed)
            if factor_types[factor] == 'Categorical':
                factor_types[factor] = 'object'
            elif factor_types[factor] == 'Integer':
                factor_types[factor] = 'int64'
            else:
                factor_types[factor] = 'float64'
            data[factor] = data[factor].astype(factor_types[factor])
            if factor_types[factor] != 'object':
                factor_ranges[factor][0] = colos[2].number_input(f"Min value of **{factor}**",
                    value=factor_ranges[factor][0], key=f"min_{factor}", label_visibility='collapsed',
                    on_change=model_changed)
                factor_ranges[factor][1] = colos[3].number_input(f"Max value of **{factor}**",
                    value=factor_ranges[factor][1], key=f"max_{factor}", label_visibility='collapsed',
                    on_change=model_changed)
        if data is not None and len(factors) > 0 and len(responses) > 0:
            features, outcomes, messages = encode_data(
                data, factors, responses, factor_ranges)
        if len(messages) > 0:
            key, value = list(messages.items())[0]
            messages[key] = '⚠️   '+messages[key]
            message = '''

⚠️   '''.join(messages.values())
            placeholder.error(message)
            for name,messsage in messages.items():
                # drop factors[name]
                factors.remove(name)
        st.write("")
        st.write("")
        st.write("")
        st.write("")


with tabs[1]:
    if dataf is not None and len(factors) > 0 and len(responses) > 0:
        cols = st.sidebar.columns([1,1])
        maximize = {}
        for i in range(len(responses)):
            temp = cols[i].radio(f"Direction for **{responses[i]}**:", ["Maximize", "Minimize"], 
                                 on_change=model_changed)
            maximize[responses[i]] = True if temp == "Maximize" else False
        Nexp = st.sidebar.number_input("Number of experiments", 
                min_value=1, value=1, max_value=100, 
                help="Number of experiments to look for the optimum response.", 
                on_change=model_changed)
        samplerchoice = st.sidebar.selectbox("Select the generator", ["Sobol pseudo-random", "Bayesian Optimization"], help="""### Select the generator to use for the optimization.  
- **Sobol pseudo-random:** Sobol sequence generator. This will tend to explore the parameter space more uniformly (exploration).
- **Bayesian Optimization:** Bayesian optimization. This will tend to exploit the parameter space more (exploitation).  

It is recommended to use the Sobol generator for the first few (5-10) iterations, and then switch to Bayesian optimization for the last iterations.
""", on_change=model_changed)
        sampler_list = {"Sobol pseudo-random": 'sobol',
                        "Bayesian Optimization": 'bo'}
        # fix a parameter value
        fixed_features_names = st.sidebar.multiselect("""Select the fixed parameters (if any)""", 
                factors, help="Select one or more features to fix during generation.", on_change=model_changed)
        fixed_features_values = [None]*len(fixed_features_names)
        if len(fixed_features_names) > 0:
            for i,feature in enumerate(fixed_features_names):
                if factor_types[feature] == 'object':
                    cases = data[feature].unique()
                    fixed_features_values[i] = st.sidebar.selectbox(f"Value of **{feature}**", 
                                                                    cases, 
                                                                    key=f"fixpar{i}", 
                                                                    on_change=model_changed)
                else:
                    fixed_features_values[i] = st.sidebar.number_input(f"Value of **{feature}**", 
                                                                        value=np.mean(data[feature]), key=f"fixpar{i}", 
                                                                        on_change=model_changed)
        # regroup the fixed features in a dict
        fixed_features = {}
        for i,feature in enumerate(fixed_features_names):
            fixed_features[feature] = fixed_features_values[i]
                # feature_constraints += [f'{par} >= {fixparval[i]}']
                # feature_constraints += [f'{par} <= {fixparval[i]}']
        
        # add a text input to add constraints
        feature_constraints = st.sidebar.text_input("""Add constraints on the features (if any). Use a comma to separate multiple constraints.""",
                help="""The constraints should be in the form of inequalities such as:

- `x1 >= 0`
- `x2 <= 10, x4 >= -0.5`
- `x1 + 3*x2 <= 5`""", on_change=model_changed)
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
        
        # same for outcome_constraints (not working?)
        # I get ValueError: Cannot constrain on objective metric. Need to to some digging.
#         outcome_constraints = st.sidebar.text_input("""Add constraints on the outcomes (if any). Use a comma to separate multiple constraints.""",
#                 help="""The constraints should be in the form of inequalities such as:
# - `constrained_metric <= some_bound`""")
#         if len(outcome_constraints)>0:
#             outcome_constraints = outcome_constraints.replace("+", " + ")
#             outcome_constraints = outcome_constraints.replace("<", "<=")
#             outcome_constraints = outcome_constraints.replace(">", ">=")
#             outcome_constraints = outcome_constraints.replace("<==", "<=")
#             outcome_constraints = outcome_constraints.replace("<=", " <= ")
#             outcome_constraints = outcome_constraints.replace(">==", ">=")
#             outcome_constraints = outcome_constraints.replace(">=", " >= ")
#             outcome_constraints = outcome_constraints.split(",")
#         else:
#             outcome_constraints = []
        
        # Perform Bayesian optimization
        colos = st.columns([2,4])
        if samplerchoice == "Bayesian Optimization":
            colos[0].success("**Bayesian optimization** is a probabilistic model. The results may vary slightly each time you run it.", icon=":material/info:")
        else:
            colos[0].warning("""You are using the **Sobol pseudo-random generator**. The results will vary each time you run it.
                            
**You are _not_ performing an optimization**, but an uniform sampling of the parameter space. This is suitable for the first few iterations of the optimization (exploration), then switch to Bayesian optimization.""", icon="⚠️")
        buttons = colos[0].columns([1,1])
        modelbutton = buttons[0].empty()
        plotbutton = buttons[1].empty()
        # Check constraints
        if len(feature_constraints) > 0:
            constraint_results = check_constraints(data, feature_constraints)
            all_valid = all(result.all() for result in constraint_results.values())
            if not all_valid:
                # print which constraints are not valid
                for feature, result in constraint_results.items():
                    if not result.all():
                        whichfails = result[result == False].index.tolist()
                        colos[0].error(f"Constraint **{feature}** is invalid for the given data (see lines: {whichfails}). It was discarded.")
                        # drop feature_constraints[i]
                        feature_constraints = [f for f in feature_constraints if f != feature]
        # check outcome_constraints
        # if len(outcome_constraints) > 0:
        #     constraint_results = check_constraints(data, outcome_constraints)
        #     all_valid = all(result.all() for result in constraint_results.values())
        #     if not all_valid:
        #         # print which constraints are not valid
        #         for feature, result in constraint_results.items():
        #             if not result.all():
        #                 whichfails = result[result == False].index.tolist()
        #                 colos[0].error(f"Constraint **{feature}** is invalid for the given data (see lines: {whichfails}). It was discarded.")
        #                 # drop feature_constraints[i]
        #                 outcome_constraints = [f for f in outcome_constraints if f != feature]
        
        # bo = AxBOExperiment(features=features, 
        #                     outcomes=outcomes,
        #                     ranges=factor_ranges,
        #                     N = Nexp,
        #                     maximize=maximize,
        #                     # outcome_constraints=outcome_constraints,
        #                     outcome_constraints=None,
        #                     fixed_features=fixed_features,
        #                     feature_constraints=feature_constraints,
        #                     optim = sampler_list[samplerchoice])
        if modelbutton.button("Compute / Update model", type="primary", 
                              disabled=st.session_state['model_up_to_date'], 
                              on_click=model_updated):
            update_model(
                    features, outcomes,
                    factor_ranges, Nexp, maximize, 
                    fixed_features, feature_constraints, 
                    sampler_list[samplerchoice]
                    )
            st.session_state.plot_up_to_date = False
            st.session_state['next'] = st.session_state['bo'].suggest_next_trials()
            st.session_state['best'] = st.session_state['bo'].get_best_parameters()
        if (st.session_state['bo'] is not None and 
            st.session_state['next'] is not None and 
            st.session_state['best'] is not None):
            colos[1].write("**Next experiments to perform:**")
            colos[1].dataframe(st.session_state['next'], hide_index=True)
            colos[1].write("**Best parameters found:**")
            colos[1].dataframe(st.session_state['best'], hide_index=True)
        figmod = []
        figopt = None
        # add a button to launch pareto frontiers plotting
        st.sidebar.write("---")
        st.sidebar.write("#### Plot options: slices")
        parslice = {}
        for f in factors:
            if features[f]['type'] == 'float':
                temp = st.sidebar.number_input(f"Slice for **{f}**", key=f"parslice{f}", 
                                               on_change=plot_changed,
                                               value=None, min_value=features[f]['range'][0],
                                               max_value=features[f]['range'][1])
                if temp is not None:
                    parslice[f] = temp
            if features[f]['type'] == 'text':
                temp = st.sidebar.multiselect(f"Slice for **{f}**", max_selections=1,
                                              on_change=plot_changed,
                                              options=features[f]['range'], key=f"parslice{f}")
                if len(temp) > 0:
                    parslice[f] = temp[0]
            if features[f]['type'] == 'int':
                temp = st.sidebar.number_input(f"Slice for **{f}**", key=f"parslice{f}", 
                                               on_change=plot_changed,
                                               value=None, 
                                               min_value=int(features[f]['range'][0]),
                                               max_value=int(features[f]['range'][1]))
                if temp is not None:
                    parslice[f] = temp

        count = len([name for name, info in features.items() if info['type']=='float' or info['type']=='int'])-len(parslice)
        if st.session_state['bo'] is not None and (plotbutton.button("Plot model / Update plots", type="primary",
                             on_click=plot_updated,
                             disabled=st.session_state['plot_up_to_date']) or
            st.session_state['plot_up_to_date'] == True):
            if count>0:
                for i in range(len(responses)):
                    figmod.append(st.session_state['bo'].plot_model(metricname=responses[i], 
                                                slice_values=parslice,
                                                linear=False if count > 1 else True,
                                                ))
            if len(responses) == 1:
                figopt = st.session_state['bo'].plot_optimization_trace()
            if figmod is not None and count>0:
                for i in range(len(responses)):
                    st.plotly_chart(figmod[i], key=f"figmod{i}")
            elif figmod is not None and count==0:
                st.write("Can't plot a model with no features with type `float`.")
            if figopt is not None:
                st.plotly_chart(figopt, key=f"figopt")
        if (st.session_state['bo'] is not None and 
            len(responses) == 2 and
            st.session_state['plot_up_to_date'] == True and
            st.session_state['bo'].model is not None and
            plotbutton.button("Plot Pareto frontiers", type="primary")):
            figpareto = st.session_state['bo'].plot_pareto_frontier()
            if figpareto is not None:
                st.plotly_chart(figpareto, key=f"figpareto")


with tabs[2]:
    if dataf is not None and len(factors) > 0 and len(responses) > 0:
        st.write("#### Select the parameters for prediction of the response")
        cols = st.columns([1,1])
        # add a button to launch predictions
        parslice = {}
        for f in factors:
            if features[f]['type'] == 'float':
                parslice[f] = cols[0].number_input(f"Value of **{f}**", 
                                                    value='min', 
                                                    min_value=float(features[f]['range'][0]),
                                                    max_value=float(features[f]['range'][1]))
            elif features[f]['type'] == 'text':
                parslice[f] = str(cols[0].selectbox(f"Value of **{f}**",
                                        options=features[f]['range']))
            elif features[f]['type'] == 'int':
                parslice[f] = cols[0].number_input(f"Value of **{f}**", 
                                                      value=int(np.mean(features[f]['range'])), 
                                                      min_value=int(features[f]['range'][0]),
                                                      max_value=int(features[f]['range'][1]))
        if len(parslice) > 0 and st.session_state['bo'] is not None:
            pred = st.session_state['bo'].predict([parslice])
            pred = pd.DataFrame(pred)
            # append "Predicted_" to the response names
            pred.columns = [f"Predicted_{col}" for col in pred.columns]
            cols[0].dataframe(pred, hide_index=True)
            # actual = {}
            # cols[1].write("Update model with actual value of the response for these parameters")
            # for i in range(len(responses)):
            #     temp = cols[1].number_input(f"Actual value of **{responses[i]}**", 
            #                                                 value=None, step=0.1)
            #     if temp is not None:
            #         actual[responses[i]] = temp
            # if cols[1].button("Update model"):
            #     if len(actual) == len(responses):
            #         # update the model with the new data
            #         newdata = pd.DataFrame({**parslice, **actual}, index=[0])
            #         st.session_state['data'] = pd.concat([st.session_state['data'], newdata], ignore_index=True)
            #         st.success("Model updated with actual value of the response.")
            #     else:
            #         st.error("Please provide actual values for all responses.")