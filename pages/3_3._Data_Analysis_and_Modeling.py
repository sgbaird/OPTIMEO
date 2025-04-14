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
from ressources.functions import about_items
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.gofplots import qqplot


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

tabs = st.tabs(["Data Loading", "Visual Assessment", "Linear Regression Model"])

with tabs[0]: # data loading
    left, right = st.columns([2,3])
    datafile = left.file_uploader("Upload a CSV file (comma separated values)", type=["csv"], 
                    help="The data file should contain the factors and the response variable.")
    if datafile is not None:
        data = pd.read_csv(datafile)
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
                             title=f'Scatter Plot: {factor} vs {response}')
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
            name='Slope'
        ))

        # Update layout for log scale and labels
        fig.update_layout(
            xaxis_title='Magnitude of effect',
            xaxis_type="log",
            height=400  # Adjust height as needed
        )

        # Add legend
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

        # Add p-values as annotations
        for i, p in enumerate(res['slope']):
            fig.add_annotation(
                x=p, y=i,
                text=f"p={result.pvalues[res['terms'].iloc[i]]:.2g}",
                showarrow=False,
                xshift=10
            )
        fig.update_layout(
            plot_bgcolor="white",  # White background
            height=500,  # Adjust height as needed
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
            )
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

