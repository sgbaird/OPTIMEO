import streamlit as st
from ressources.functions import *
import optuna
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from ressources.functions import about_items
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

st.set_page_config(page_title="New set of experiments using Bayesian Optimisation",
                   page_icon="📈", layout="wide", menu_items=about_items)

style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# New set of experiments using Bayesian Optimisation
""")


data = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

tabs = st.tabs(["Data Loading", "Bayesian Optimization"])

with tabs[0]:
    if data is not None:
        data = pd.read_csv(data)
        left, right = st.columns([3,2])
        cols = data.columns.to_numpy()
        left.dataframe(data, hide_index=True)
        mincol = 1 if 'run_order' in cols else 0
        factors = right.multiselect("Select the factors columns:", 
                data.columns, default=cols[mincol:-1])
        # response cannut be a factor, so default are all unselected columns in factor
        available = [col for col in cols if col not in factors]
        response = right.multiselect("Select the response column:", 
                available, default=available[-1], max_selections=1)
        if len(response) > 0:
            response = response[0]
        # add option to change type of columns
        dtypesF = data[factors].dtypes
        categ = right.multiselect("Categorical factors", factors, 
            default=[dtypesF.index[i] for i in range(len(dtypesF)) if dtypesF[i] == 'object'])
        floats = right.multiselect("Float factors", factors, 
            default=[dtypesF.index[i] for i in range(len(dtypesF)) if dtypesF[i] == 'float64'])
        integers = right.multiselect("Integer factors", factors, 
            default=[dtypesF.index[i] for i in range(len(dtypesF)) if dtypesF[i] == 'int64'])
        # change the type of the columns accordingly
        data[categ] = data[categ].astype('object')
        data[floats] = data[floats].astype('float64')
        data[integers] = data[integers].astype('int64')
        data, encoders, dtypes = encode_data(data, factors)


with tabs[1]:
    if data is not None and len(factors) > 0 and len(response) > 0:
        Nexp = st.sidebar.number_input("Number of experiments", 
                min_value=1, value=1, max_value=100, 
                help="Number of experiments to look for the optimum response.")
        # fix a parameter value
        samplerchoice = st.sidebar.selectbox("Select the sampler", ["TPE", "NSGAII", "Base"], help="""### Select the sampler to use for the optimization.  
- **TPE:** Tree-structured Parzen Estimator. This will tend to explore the parameter space more efficiently (exploitation).
- **NSGAII:** Non-dominated Sorting Genetic Algorithm II. This will tend to explore the parameter space more uniformly (exploration).
- **Base:** Base sampler. This will tend to explore the parameter space uniformly.""")
        sampler_list = {"TPE": optuna.samplers.TPESampler,
                        "NSGAII": optuna.samplers.NSGAIISampler,
                        "Base": optuna.samplers.BaseSampler}
        fixpar = st.sidebar.multiselect("Fix a parameter value", factors,
                help="Select a parameter to fix its value in the optimization.")
        fixparval = [None]*len(fixpar)
        if len(fixpar)>0:
            for i,par in enumerate(fixpar):
                if dtypes[par] == 'object':
                    cases = encoders[par].inverse_transform([round(f) for f in data[par].unique()])
                    fixparval[i] = st.sidebar.selectbox(f"Value of {par}", cases, 
                                                        key=f"fixpar{i}")
                    fixparval[i] = encoders[par].transform([fixparval[i]])[0]
                else:
                    fixparval[i] = st.sidebar.number_input(f"Value of {par}", 
                                    value=np.mean(data[par]), key=f"fixpar{i}")
        fixedparval = {par: val for par,val in zip(fixpar, fixparval)}
        X = data[factors].values
        y = data[response].values
        # Standardize the input feature
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Train Gaussian Process Regressor
        # kernel = RBF(length_scale_bounds='fixed', length_scale=0.25) 
        gp_model = GaussianProcessRegressor(n_restarts_optimizer=10, 
                                            # kernel=kernel,
                                            random_state=12345)
        gp_model.fit(X_scaled, y)

        # Objective function to maximize
        def evaluate_objective(X):
            # Prepare input for prediction and scale
            X_pred = scaler.transform(X)
            # Predict B/C using Gaussian Process Regressor
            response_pred, _ = gp_model.predict(X_pred, return_std=True)
            return response_pred[0]

        def objective(trial):
            # Search space for parameters
            trials = [None]*len(factors)
            for i,factor in enumerate(factors):
                if dtypes[factor] == 'object':
                    suggestions = data[factor].unique() if factor not in fixpar else [fixedparval[factor]]
                    trials[i] = trial.suggest_categorical(factor, suggestions)
                elif dtypes[factor] == 'int':
                    Min = np.min(data[factor]) if factor not in fixpar else fixedparval[factor]
                    Max = np.max(data[factor]) if factor not in fixpar else fixedparval[factor]
                    trials[i] = trial.suggest_int(factor, Min, Max)
                else:
                    Min = np.min(data[factor]) if factor not in fixpar else fixedparval[factor]
                    Max = np.max(data[factor]) if factor not in fixpar else fixedparval[factor]
                    trials[i] = trial.suggest_float(factor, Min, Max)
            trials = np.array(trials).reshape(1, -1)
            # Evaluate the objective function
            resp = evaluate_objective(trials)
            return resp

        # Perform Bayesian optimization
        cols = st.columns([1,3])
        direction = cols[0].radio("Select the direction to optimize:", ["Maximize", "Minimize"])
        if samplerchoice == "Base":
            study = optuna.create_study(direction=direction.lower())
        else:
            sampler = sampler_list[samplerchoice]()
            study = optuna.create_study(direction=direction.lower(), sampler=sampler)
        study.optimize(objective, n_trials=100, n_jobs=-1)

        # Get the best hyperparameters
        res = study.trials_dataframe()
        res = res[res['state']=='COMPLETE']
        res = res.sort_values('value', ascending=False)
        # remove the strin 'params_' from the column names
        res.columns = [col.replace('params_', '') for col in res.columns]
        # take the first Nexp best parameters
        best_params = res.head(Nexp)[factors+['value']]
        # rename the value column to the response
        best_params = best_params.rename(columns={'value': f"Expected {response}"})

        outdf = best_params.copy()
        # make the output more readable
        for col in outdf.columns:
            outdf[col] = np.round(outdf[col], 2)
        outdf = decode_data(outdf, factors, dtypes, encoders)
        cols[1].write("New parameters to try and expected response:")
        cols[1].dataframe(outdf, hide_index=True)
        
        timestamp = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        df_new = outdf.copy()
        df_new['run_order'] = np.arange(1, len(df_new)+1)
        df_new['run_order'] = np.random.permutation(df_new['run_order'])
        colos = df_new.columns.tolist()
        colos = colos[-1:] + colos[:-1]
        df_new = df_new[colos]
        # add an empty "response" column to the design
        df_new['response'] = ''
        outfile = writeout(df_new)
        cols[0].download_button(
            label     = f"Download new Experimental Design with {len(df_new)} runs",
            data      = outfile,
            file_name = f'newDOE_{timestamp}.csv',
            mime      = 'text/csv',
            key       = 'download-csv'
        )

        ncols = np.min([len(factors),4])
        cols = st.columns(int(ncols))
        for i,factor in enumerate(factors):
            fig, ax = plt.subplots()
            Xr = pd.DataFrame(columns=factors)
            for f in factors:
                if f == factor:
                    Xr[f] = np.linspace(np.min(data[f]), np.max(data[f]), 50)
                else:
                    Xr[f] = np.repeat(best_params[f].values[0], 50)
            Xr = Xr.values
            Xr = scaler.transform(Xr)
            yp, ys = gp_model.predict(Xr, return_std=True)
            Xr = scaler.inverse_transform(Xr)
            Xr = pd.DataFrame(Xr, columns=factors)
            Xr = Xr[factor].values.reshape(-1, 1)
            ax.plot(Xr, yp)
            ax.fill_between(Xr[:, 0], yp - ys, yp + ys, alpha=0.1, 
                            color='k', label = "Uncertainty")
            ax.scatter(data[factor], data[response], s=100)
            ax.scatter(best_params[factor], 
                       best_params[f"Expected {response}"], s=200, color='red')
            ax.set_xlabel(factor)
            ax.set_ylabel(response)
            # if factor is categorical, change the xticks to the categories
            if dtypes[factor] == 'object':
                ax.set_xticks(np.arange(len(data[factor].unique())))
                labels = encoders[factor].inverse_transform([round(f) for f in data[factor].unique()])
                ax.set_xticklabels(labels)
            fig.tight_layout()
            cols[i%ncols].pyplot(fig)
            
        plt.rcParams.update({'font.size': 22})
        cols = st.columns(len(factors)-1)
        for i,faci in enumerate(factors):
            for j,facj in enumerate(factors):
                if j>i:
                    fig, ax = plt.subplots()
                    ax.scatter(data[facj], data[faci], s=100)
                    ax.scatter(best_params[facj], best_params[faci], s=100, color='red')
                    ax.set_ylabel(faci)
                    ax.set_xlabel(facj)
                    fig.tight_layout()
                    cols[j-1].pyplot(fig)


