import streamlit as st
import numpy as np
from ressources.functions import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ressources.functions import about_items
# import ML models and scaling functions
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Data Analysis and Modeling", 
                   page_icon="📈", layout="wide", menu_items=about_items)

style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

if 'firstlaunch' not in st.session_state:
    st.session_state['firstlaunch'] = 1
    if not os.path.exists("./trained_model"):
        os.mkdir("./trained_model")
    available_models = os.listdir("./trained_model")
    for model_name in available_models:
        os.remove(f"./trained_model/{model_name}")
if 'launch' not in st.session_state:
    st.session_state['launch'] = 0
if 'fitted' not in st.session_state:
    st.session_state['fitted'] = 0

def write_poly(pp):
    signs = [np.sign(x) for x in pp]
    p = np.array([f"{x:.2e}" for x in pp])
    p = [x.split("e") for x in p]
    p = [f"{x[0]}\cdot10^{{{int(x[1])}}}" for x in p]
    p = [x.replace(f'\cdot10^{{0}}', '') for x in p]
    out = f"$y = {p[0]}x^2"
    out += f"{'' if signs[1]<0 else '+'} {p[1]}x"
    out += f"{'' if signs[2]<0 else '+'} {p[2]}$"
    return(out)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Data Analysis and Modeling
""")

tabs = st.tabs(["Data Loading", "Visual Assessment", "Linear Regression Model", "Machine Learning Models"])

with tabs[0]: # data loading
    left, right = st.columns([2,3])
    datafile = left.file_uploader("Upload a CSV file (comma separated values)", type=["csv"], 
                    help="The data file should contain the factors and the response variable.")
    if datafile is not None:
        data = pd.read_csv(datafile)
        cols = data.columns.to_numpy()
        right.dataframe(data, hide_index=True)
        mincol = 1 if 'run_order' in cols else 0
        factors = left.multiselect("Select the **factors** columns:", 
                data.columns, default=cols[mincol:-1])
        # response cannut be a factor, so default are all unselected columns in factor
        available = [col for col in cols if col not in factors]
        response = [left.selectbox("Select the **response** column:", 
                available, index=len(available)-1)]
        if len(response) > 0:
            response = response[0]
        dtypes = data.dtypes
        responsevals = data[response].values
        dtypesF = data[factors].dtypes
        factor_carac = {factor: dtypesF[factor] for factor in factors}
        type_choice = {'object':0, 'int64':1, 'float64':2}
        colos = st.columns(3)
        colos[1].write("<p style='text-align:center;'><b>Type</b></p>", unsafe_allow_html=True)
        for factor in factors:
            colos = st.columns(3)
            colos[0].write(f"<p style='text-align:right;'><b>{factor}</b></p>", unsafe_allow_html=True)
            factype = type_choice[f"{factor_carac[factor]}"]
            factor_carac[factor] = colos[1].selectbox(f"Type of **{factor}**", 
                ['Categorical', 'Integer', 'Float'], key=f"type_{factor}", index = factype, label_visibility='collapsed')
            if factor_carac[factor] == 'Categorical':
                factor_carac[factor] = 'object'
            elif factor_carac[factor] == 'Integer':
                factor_carac[factor] = 'int64'
            else:
                factor_carac[factor] = 'float64'
            data[factor] = data[factor].astype(factor_carac[factor])
        data, encoders, dtypes = encode_data(data, factors)


with tabs[1]: # visual assessment
    if datafile is not None and len(factors) > 0 and len(response) > 0:
        toplot = data.copy() # for whatever reason, the data order is changed if we don't do that
        plt.rcParams.update({'font.size': 22})
        ncols = np.min([len(factors),4])
        cols = st.columns(4)
        fig, ax = plt.subplots()
        sm.qqplot(toplot[response], line='s', ax=ax) 
        fig.tight_layout()
        cols[0].pyplot(fig)
        fig, ax = plt.subplots()
        plt.boxplot(toplot[response], vert=False) 
        plt.xlabel(response)
        fig.tight_layout()
        cols[1].pyplot(fig)
        fig, ax = plt.subplots()
        plt.hist(toplot[response]) 
        plt.xlabel(response) 
        fig.tight_layout()
        cols[2].pyplot(fig)
        fig, ax = plt.subplots()
        toplot[response] = responsevals
        plt.scatter(range(1,len(toplot[response])+1), toplot[response], s=100)
        plt.xlabel('Measurement number') 
        plt.ylabel(response)
        fig.tight_layout()
        cols[3].pyplot(fig)
        
        cols = st.columns(int(ncols))
        for i,factor in enumerate(factors):
            fig, ax = plt.subplots()
            ax.scatter(toplot[factor], toplot[response], s=100)
            # add linear regression with red line and equation
            if dtypes[factor] != 'object':
                p = np.polyfit(toplot[factor], toplot[response], 2)
                ax.plot(np.linspace(np.min(toplot[factor]),np.max(toplot[factor]),100), np.polyval(p, np.linspace(np.min(toplot[factor]),np.max(toplot[factor]),100)), color='red')
                ax.set_title(write_poly(p), fontsize=20)
            ax.set_xlabel(factor)
            ax.set_ylabel(response)
            fig.tight_layout()
            cols[i%ncols].pyplot(fig)


with tabs[2]: # simple model
    if datafile is not None and len(factors) > 0 and len(response) > 0:
        cols = st.columns([1,1,4])
        order = cols[0].number_input("Interactions order:", 
                                        min_value=1, value=1, max_value=len(factors))
        quadratic = cols[1].multiselect("Quadratic terms?", factors)
        def_eqn = write_equation(factors, response, order, quadratic, dtypes)
        eqn = cols[2].text_input("Model equation:", key="eqn", value=def_eqn, help="""Interactions are written as `factor1:factor2`.  
Powered terms are written as `I(factor**power)`.  
Categorical variables are written as `C(factor)`.  
You can also use `np.log(factor)` or `np.exp(factor)` for transformations.  
To remove the intercept, add `-1` at the end of the equation.""")
        model = smf.ols(formula=eqn, data=data)
        result = model.fit()
        st.write(result.summary())
        plt.rcParams.update({'font.size': 14})
        cols = st.columns([1,1])
        fig, ax = plt.subplots()
        ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=.5)
        plt.plot(data[response], result.predict(), 'o') 
        plt.xlabel(f'Actual {response}') 
        plt.ylabel(f'Predicted {response}')
        fig.tight_layout()
        cols[0].pyplot(fig)
        # plot the slope values for each factor
        fig, ax = plt.subplots()
        res = result.params.rename_axis('terms').reset_index()[1:]
        res.columns = ['terms', 'slope']
        error = result.bse.rename_axis('terms').reset_index()[1:]
        error.columns = ['terms', 'error']
        res['error'] = error['error']
        res['pvalue'] = [result.pvalues[res['terms']].iloc[i] for i in range(len(res))]
        # sort the res coloumn by increasing order of pvalues
        res = res.sort_values(by='pvalue', ascending=False)
        # res = res[order]
        colors = ['red' if x < 0 else 'green' for x in res['slope']]
        res['slope'] = res['slope'].abs()
        ax.barh([term.replace('I(', '').replace('C(', '').replace(')', '').replace(' ** ', '^') for term in res['terms']], res['slope'], xerr=res['error'], color=colors)
        ax.set_xlabel('Magnitude of effect')
        ax.set_xscale('log')
        red_patch = mpatches.Patch(color='red', label='Negative')
        blue_patch = mpatches.Patch(color='green', label='Positive')
        ax.legend(handles=[red_patch, blue_patch])
        # add pvalues on the right of each bar
        for i, p in enumerate(res['slope']):
            ax.text(p, i+.2, f"p={result.pvalues[res['terms'].iloc[i]]:.2g}", va='center')
        fig.tight_layout()
        cols[1].pyplot(fig)
        # # # # # # # # # # # # # # # 
        st.write("#### Predict the response for a set of factors")
        Xnew = {factor: 0 for factor in factors}
        left, right = st.columns(2)
        for i,factor in enumerate(factors):
            colsinput = left.columns(2)
            colsinput[0].write(f"<p style='text-align:right;font-size:1.5em'><b>{factor}</b></p>", unsafe_allow_html=True)
            if dtypes[factor] == 'object':
                # non encoded factor
                possible = np.unique(encoders[factor].inverse_transform(data[factor].values))
                Xnew[factor] = colsinput[1].selectbox(f"{factor}", possible, key=f"{factor}lm", label_visibility='collapsed')
            else:
                Xnew[factor] = colsinput[1].number_input(f"{factor}", 
                                            value=np.mean(data[factor]), key=f"{factor}lm", label_visibility='collapsed')
        # encode the factors if they are categorical
        for i,factor in enumerate(factors):
            if dtypes[factor] == 'object':
                toencode = Xnew[factor].values[0]
                Xnew[factor] = encoders[factor].transform([toencode])[0]
        # st.write(Xnew)
        right.write(f"<p style='text-align:center;font-size:1.5em'>Predicted {response}:<br><br><b>{result.predict(exog=Xnew)[0]:.4g}</b></p>",
                    unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")

with tabs[3]: # machine learning model
    if datafile is not None and len(factors) > 0 and len(response) > 0:
        # Choose machine learning model
        model_sel = st.sidebar.selectbox("Select the machine learning model:", 
                ["Random Forest", "Gaussian Process"])#, "Gradient Boosting", "Linear Regression"])
        split_size = st.sidebar.number_input("Test size:", min_value=0.0, value=0.2, max_value=1.)
        X = data[factors].values
        y = data[response].values
        if split_size>0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if model_sel == "Linear Regression":
            model = LinearRegression(n_jobs=-1)
        elif model_sel == "Random Forest":
            nestim = st.sidebar.number_input("Number of estimators:", min_value=1, value=100, max_value=1000)
            model = RandomForestRegressor(n_estimators=nestim, n_jobs=-1, random_state=12345)
        elif model_sel == "Gradient Boosting":
            nestim = st.sidebar.number_input("Number of estimators:", min_value=1, value=100, max_value=1000)
            model = GradientBoostingRegressor(n_estimators=nestim, random_state=12345)
        elif model_sel == "Gaussian Process":
            model = GaussianProcessRegressor(n_restarts_optimizer=10, random_state=12345)
        # add a Go button to launch the fit
        cols = st.columns([1,1])
        if st.sidebar.button(f"Launch the fitting process with the {model_sel} model"):
            model = train_model(X_train, y_train, model, model_sel)
            st.session_state['fitted'] = 1
        if st.sidebar.button(f"Clear all saved models"):
            clear_models()
        available_models = os.listdir("./trained_model")
        if model_sel in [av.replace('_', ' ').replace('.pkl', '') for av in available_models]:
            model = load_model(f"{model_sel}.pkl")
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2tr = r2_score(y_train, model.predict(X_train))
            if split_size>0:
                r2te = r2_score(y_test, y_pred)
            plt.rcParams.update({'font.size': 14})
            fig, ax = plt.subplots()
            ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=.5)
            plt.scatter(y_train, model.predict(X_train), label='Train data', s=100)
            if split_size>0:
                plt.scatter(y_test, y_pred, label='Test data', s=100)
            plt.legend()
            plt.xlabel(f'Actual {response}')
            plt.ylabel(f'Predicted {response}')
            plt.title(f"{model_sel}")
            fig.tight_layout()
            cols[0].pyplot(fig)
            if split_size>0:
                cols[0].dataframe(pd.DataFrame({'RMSE':[rmse], 'R2_train': [r2tr], 'R2_test': [r2te]}), hide_index=True)
            else:
                cols[0].dataframe(pd.DataFrame({'RMSE':[rmse], 'R2_train': [r2tr]}), hide_index=True)
            # plot the feature importance
            if model_sel in ["Random Forest", "Gradient Boosting"]:
                fig, ax = plt.subplots()
                importance = model.feature_importances_
                importance = importance / np.max(importance)
                importance, factors = zip(*sorted(zip(importance, factors)))
                ax.barh(factors, importance)
                ax.set_xlabel('Normalized Importance')
                fig.tight_layout()
                cols[0].pyplot(fig)
            else:
                cols[0].write(f"Feature importance is not available for the {model_sel} model.")
            cols[1].write("#### Predict the response for a set of factors")
            Xnew = pd.DataFrame(columns=factors)
            for i,factor in enumerate(factors):
                colsinput = cols[1].columns([2,2,1])
                colsinput[0].write(f"<p style='text-align:right;font-size:1.4em'><b>{factor}</b></p>", unsafe_allow_html=True)
                if dtypes[factor] == 'object':
                    # non encoded factor
                    possible = np.unique(encoders[factor].inverse_transform(data[factor].values))
                    Xnew[factor] = [colsinput[1].selectbox(f"**{factor}**", possible, label_visibility='collapsed')]
                else:
                    Xnew[factor] = [colsinput[1].number_input(f"**{factor}**", 
                                                    value=np.mean(data[factor]),label_visibility='collapsed')]
            # encode the factors if they are categorical
            for i,factor in enumerate(factors):
                if dtypes[factor] == 'object':
                    toencode = Xnew[factor].values[0]
                    Xnew[factor] = encoders[factor].transform([toencode])[0]
            Xnew = Xnew.values.reshape(1, -1)
            Xnew = scaler.transform(Xnew)
            cols[1].write(f"<p style='text-align:center;font-size:1.5em'><br>Predicted {response}:<br><b>{model.predict(Xnew)[0]:.4g}</br></p>",unsafe_allow_html=True)