# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 

import streamlit as st
import numpy as np
from ressources.functions import *
import pandas as pd
import statsmodels.formula.api as smf
from ressources.functions import about_items, bootstrap_coefficients
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
from sklearn.pipeline import make_pipeline
import seaborn as sns

st.set_page_config(page_title="Data Analysis and Modeling", 
                   page_icon="ressources/icon.png", 
                   layout="wide", menu_items=about_items)

style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Data Analysis and Modeling
""")

tabs = st.tabs(["Data Loading", "Visual Assessment", "Linear Regression Model", 'Machine Learning Model'])

with tabs[0]: # data loading
    left, right = st.columns([2,3])
    datafile = left.file_uploader("""Upload data file (csv, xls, xlsx, xlsm, xlsb, odf, ods and odt).

For Excel-like files, make sure the data start in the A1 cell.""", 
                type=["csv",'xlsx','xls', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'],
                help="The data file should contain the factors and the response variable.")
    if datafile is None:
        container = right.container(border=True)
        container.markdown(
        "⚠️ The data must be in tidy format, meaning that each column is a variable and each row is an observation. We usually place the factors in the first columns and the response(s) in the last column(s). Data type can be float, integer, or text, and you can only specify one response. Spaces and special characters in the column names will be automatically removed. The first row of the file will be used as the header."
        )
        container.image("ressources/tidy_data.jpg", caption="Example of tidy data format")
    if datafile is not None:
        if Path(datafile.name).suffix == '.csv':
            data = pd.read_csv(datafile)
        else:
            data = pd.read_excel(datafile)
        data = clean_names(data, remove_special=True, case_type='preserve')
        cols = data.columns.to_numpy()
        right.dataframe(data, hide_index=True)
        mincol = 1 if 'run_order' in cols else 0
        factors = left.multiselect("Select the **factors** columns:", 
                data.columns, default=cols[mincol:-1])
        # response cannot be a factor, so default are all unselected columns in factor
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
        data, encoders, dtypes = encode_data2(data, factors)


with tabs[1]: # visual assessment
    if datafile is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the "Data Loading" tab.""")
    if datafile is not None and len(factors) > 0 and len(response) > 0:
        # Assuming 'data', 'response', 'factors', 'responsevals', and 'dtypes' are defined
        toplot = data.copy()

        # Set up the layout for Streamlit
        ncols = min(len(factors), 4)
        cols = st.columns(4)

        # Q-Q Plot
        qqplot_data = qqplot(toplot[response], line='s').gca().lines
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
        fig.update_layout(title='Q-Q Plot',
                          xaxis_title='Theoretical Quantiles',
                          showlegend=False,
                          yaxis_title='Sample Quantiles')
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
        cols[0].plotly_chart(fig)

        # Box Plot
        fig = px.box(toplot, y=response, points="all", title='Box Plot')
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
        cols[1].plotly_chart(fig)

        # Histogram
        fig = px.histogram(toplot, x=response, title='Histogram')
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
        cols[2].plotly_chart(fig)

        # Scatter Plot of Response Values
        toplot[response] = responsevals
        fig = px.scatter(x=np.arange(1, len(toplot[response]) + 1), 
                         y=toplot[response], 
                         labels={'x': 'Measurement number', 'y': response}, 
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
        cols[3].plotly_chart(fig)

        # Scatter Plots for Each Factor
        cols = st.columns(ncols)
        for i, factor in enumerate(factors):
            fig = px.scatter(x=toplot[factor], 
                             y=toplot[response], 
                             labels={'x': factor, 'y': response}, 
                             title=f'{factor} vs {response}')
            fig.update_layout(
                plot_bgcolor="white",  # White background
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=10, r=10, t=100, b=10),
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
            fitorder = cols[i % ncols].number_input(f"Polynomial fit order for {response} vs {factor}:",
                                        min_value=0, value=2, max_value=10, step=1)
            # Add linear regression with red line and equation
            if dtypes[factor] != 'object':
                p = np.polyfit(toplot[factor], toplot[response], fitorder)
                x_range = np.linspace(np.min(toplot[factor]), np.max(toplot[factor]), 100)
                fig.add_trace(go.Scatter(x=x_range, y=np.polyval(p, x_range), 
                                         mode='lines', 
                                         name='Polynomial Fit', 
                                         line=dict(color='red')))
                fig.update_layout(title_subtitle_text=rf'{write_poly(p)}',
                                  title_subtitle_font_size=16,
                                  showlegend=False, height=400)

            cols[i % ncols].plotly_chart(fig)
        # Pairplot for all factors
        st.write("##### Pairplot of all factors")
        # Create a pairplot of the dataset
        # Use a copy of the data to avoid modifying the original
        train_dataset = data.copy()
        fig = sns.pairplot(
            train_dataset,
            kind="reg",
            diag_kind="kde",
            plot_kws={"scatter_kws": {"alpha": 0.1}},
        )
        st.pyplot(fig, use_container_width=True)


with tabs[2]: # simple model
    if datafile is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the "Data Loading" tab.""")
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
        # Plot: Actual vs Predicted
        cols = st.columns([1, 1])
        fig = go.Figure()

        # Add actual vs predicted line
        fig.add_trace(go.Scatter(x=data[response], 
                                 y=result.predict(), 
                                 mode='markers', 
                                 marker=dict(size=12),
                                 name='Actual vs Predicted'))

        # Add 1:1 line
        fig.add_shape(type="line", 
                      x0=min(data[response]), y0=min(data[response]), 
                      x1=max(data[response]), y1=max(data[response]),
                      line=dict(color="Gray", width=1, dash="dash"))

        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis_title=f'Actual {response}',
            yaxis_title=f'Predicted {response}',
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

        cols[0].plotly_chart(fig)

        # Plot: Slope values for each factor
        res = result.params.rename_axis('terms').reset_index()[1:]
        res.columns = ['terms', 'slope']
        error = result.bse.rename_axis('terms').reset_index()[1:]
        error.columns = ['terms', 'error']
        res['error'] = error['error']
        res['pvalue'] = [result.pvalues[res['terms']].iloc[i] for i in range(len(res))]

        # Sort by p-values
        res = res.sort_values(by='pvalue', ascending=False)

        # Prepare colors and labels
        colors = ['red' if x < 0 else 'green' for x in res['slope']]
        res['slope'] = res['slope'].abs()

        fig = go.Figure()

        # Add bar plot
        fig.add_trace(go.Bar(
            y=[term.replace('I(', '').replace('C(', '').replace(')', '').replace(' ** ', '^') for term in res['terms']],
            x=res['slope'],
            error_x=dict(type='data', array=res['error']),
            marker_color=colors,
            orientation='h',
            name='Slope',
            showlegend=False  # Hide the legend entry for the bar trace
        ))

        # Update layout for log scale and labels
        fig.update_layout(
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
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='red'),
            name='Negative'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='green'),
            name='Positive'
        ))

        # Add p-values as annotations outside the plot
        for i, p in enumerate(res['slope']):
            fig.add_annotation(
                x=1.025,  # Place annotations outside the plot
                y=i,
                text=f"p = {result.pvalues[res['terms'].iloc[i]]:.2g}",
                showarrow=False,
                xref="paper",  # Use paper coordinates for x
                xanchor='left'
            )

        cols[1].plotly_chart(fig)
        # # # # # # # # # # # # # # # 
        st.write("##### Predict the response for a set of factors with this linear model:")
        Xnew = {factor: 0 for factor in factors}
        left, right = st.columns(2)
        for i,factor in enumerate(factors):
            colsinput = left.columns(2)
            colsinput[0].write(f"<p style='text-align:right;font-size:1.5em'><b>{factor}</b></p>", unsafe_allow_html=True)
            if dtypes[factor] == 'object':
                # non encoded factor
                possible = np.unique(encoders[factor].inverse_transform(data[factor].values))
                Xnew[factor] = str(colsinput[1].selectbox(f"{factor}", possible, key=f"{factor}lm", label_visibility='collapsed'))
            else:
                Xnew[factor] = colsinput[1].number_input(f"{factor}", 
                                            value=np.mean(data[factor]), key=f"{factor}lm", label_visibility='collapsed')
        # encode the factors if they are categorical
        for i,factor in enumerate(factors):
            if dtypes[factor] == 'object':
                toencode = Xnew[factor]
                Xnew[factor] = encoders[factor].transform([toencode])[0]
        # st.write(Xnew)
        right.write(f"<p style='text-align:center;font-size:1.5em'>Predicted {response}:<br><br><b>{result.predict(exog=Xnew)[0]:.4g}</b></p>",
                    unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")

with tabs[3]: # machine learning model
    if datafile is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the "Data Loading" tab.""")
    if datafile is not None and len(factors) > 0 and len(response) > 0:
        # Choose machine learning model
        model_sel = st.sidebar.selectbox("Select the machine learning model:", 
                ["ElasticNetCV", "RidgeCV", "LinearRegression", "Random Forest", "Gaussian Process", "Gradient Boosting"])
        split_size = st.sidebar.number_input("Validation set size:", 
                                             min_value=0.0, value=0.2, max_value=1.)
        X = data[factors]
        y = data[response]
        if split_size>0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        if model_sel == "ElasticNetCV":
            model = make_pipeline(StandardScaler(), ElasticNetCV())
        elif model_sel == "RidgeCV":
            model = make_pipeline(StandardScaler(), RidgeCV())
        elif model_sel == "LinearRegression":
            model = make_pipeline(StandardScaler(), LinearRegression())
        elif model_sel == "Random Forest":
            model = make_pipeline(StandardScaler(), RandomForestRegressor())
        elif model_sel == "Gaussian Process":
            model = make_pipeline(StandardScaler(), GaussianProcessRegressor())
        elif model_sel == "Gradient Boosting":
            model = make_pipeline(StandardScaler(), GradientBoostingRegressor())
        # Fit the model
        coef_names = X_train.columns
        model.fit(X_train, y_train)
        bootstrap_coefs = bootstrap_coefficients(model[1], X, y, n_bootstrap=100, random_state=42)
        # Calculate mean and standard deviation of coefficients
        mean_coefs = np.mean(bootstrap_coefs, axis=0)
        std_coefs = np.std(bootstrap_coefs, axis=0)
        # make plot of predicted versus actual
        cols = st.columns([1, 1])
        fig = go.Figure()
        # Add actual vs predicted line
        fig.add_trace(go.Scatter(x=y_train, 
                                 y=model.predict(X_train), 
                                 mode='markers', 
                                 marker=dict(size=12, color='royalblue'),
                                 name='Training'))
        fig.add_trace(go.Scatter(x=y_test, 
                                 y=model.predict(X_test), 
                                 mode='markers', 
                                 marker=dict(size=12, color='orange'),
                                 name='Validation'))
        # Add 1:1 line
        fig.add_shape(type="line", 
                      x0=min(y_train), y0=min(y_train), 
                      x1=max(y_train), y1=max(y_train),
                      line=dict(color="Gray", width=1, dash="dash"))
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis_title=f'Actual {response}',
            yaxis_title=f'Predicted {response}',
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
        fig.update_layout(
            title={
                'text': f"Predicted vs Actual for {str(model[1])}<br>R²_test = {r2_score(y_test, model.predict(X_test)):.4f}  -  R²_train = {r2_score(y_train, model.predict(X_train)):.4f}<br>RMSE_test = {root_mean_squared_error(y_test, model.predict(X_test)):.4f}  -  RMSE_train = {root_mean_squared_error(y_train, model.predict(X_train)):.4f}",
                'x': 0.45,
                'xanchor': 'center'
            }
        )
        cols[0].plotly_chart(fig)
        
        # make feature importance plot
        if model_sel != "Gaussian Process":
            fig = go.Figure()
            pos_coefs = mean_coefs[mean_coefs > 0]
            pos_coefs_names = coef_names[mean_coefs > 0]
            neg_coefs = mean_coefs[mean_coefs < 0]
            neg_coefs_names = coef_names[mean_coefs < 0]
            # Add bars for positive mean coefficients
            fig.add_trace(go.Bar(
                y=[f"{pos_coefs_names[i]}" for i in range(len(pos_coefs))],
                x=mean_coefs[mean_coefs > 0],
                error_x=dict(type='data', array=std_coefs[mean_coefs > 0], visible=True),
                orientation='h',
                marker_color='royalblue',
                name='Positive'
            ))
            fig.add_trace(go.Bar(
                y=[f"{neg_coefs_names[i]}" for i in range(len(neg_coefs))],
                x=-mean_coefs[mean_coefs < 0],
                error_x=dict(type='data', array=std_coefs[mean_coefs < 0], visible=True),
                orientation='h',
                marker_color='orange',
                name='Negative'
            ))
            # Update layout
            features_in_log = st.sidebar.toggle("Log scale for features importance", value=True)
            fig.update_layout(
                title={
                    'text': f"Features importance for {str(model[1])}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Coefficient Value",
                yaxis_title="Features",
                barmode='relative',
                margin=dict(l=150)
            )
            if features_in_log:
                fig.update_xaxes(type="log")
            else:
                fig.update_xaxes(type="linear")
            fig.update_layout(
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
            cols[1].plotly_chart(fig)
        # make prediction plot
        if model is not None:
            st.write("##### Predict the response for a set of factors with this model:")
            Xnew = []
            left, right = st.columns(2)
            for i, factor in enumerate(factors):
                colsinput = left.columns(2)
                colsinput[0].write(f"<p style='text-align:right;font-size:1.1em'><b>{factor}</b></p>", unsafe_allow_html=True)
                if dtypes[factor] == 'object':
                    # Non-encoded factor
                    possible = np.unique(encoders[factor].inverse_transform(data[factor].values))
                    Xnew.append(str(colsinput[1].selectbox(f"{factor}", possible, key=f"{factor}ml", label_visibility='collapsed')))
                else:
                    Xnew.append(colsinput[1].number_input(f"{factor}",
                                                        value=np.mean(data[factor]), key=f"{factor}ml", label_visibility='collapsed'))

            # Encode the factors if they are categorical
            for i, factor in enumerate(factors):
                if dtypes[factor] == 'object':
                    toencode = Xnew[i]
                    Xnew[i] = encoders[factor].transform([toencode])[0]

            # Convert Xnew to a numpy array and reshape
            Xnew = np.array(Xnew).reshape(1, -1)

            # Make prediction
            prediction = model.predict(Xnew)[0]

            right.container(border=True).write(f"<p style='text-align:center;font-size:1.5em'>Predicted {response}:<br><br><b>{prediction:.4g}</b></p>",
                        unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.write("")