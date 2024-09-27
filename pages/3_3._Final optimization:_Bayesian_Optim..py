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
        data, encoders, dtypes = encode_data(data, factors)


with tabs[1]:
    if data is not None and len(factors) > 0 and len(response) > 0:
        Nexp = st.sidebar.number_input("Number of experiments", 
                min_value=1, value=1, max_value=100, 
                help="Number of experiments to look for the optimum response.")
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
                    trials[i] = trial.suggest_categorical(factor, data[factor].unique())
                elif dtypes[factor] == 'int':
                    trials[i] = trial.suggest_int(factor, np.min(data[factor]), np.max(data[factor]))
                else:
                    trials[i] = trial.suggest_float(factor, np.min(data[factor]), np.max(data[factor]))
            trials = np.array(trials).reshape(1, -1)
            # Evaluate the objective function
            resp = evaluate_objective(trials)
            return resp

        # Perform Bayesian optimization
        cols = st.columns([1,3])
        direction = cols[0].radio("Select the direction to optimize:", ["Maximize", "Minimize"])
        study = optuna.create_study(direction=direction.lower())
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


