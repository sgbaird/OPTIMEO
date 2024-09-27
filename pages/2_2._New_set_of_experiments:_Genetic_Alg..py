import streamlit as st
from ressources.functions import *
import ressources.ga as ga
from ressources.functions import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="New experimental plan using Genetic Algorithm", 
                   page_icon="📈", layout="wide", menu_items=about_items)

style = read_markdown_file("ressources/style.css")
st.markdown(style, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# New experimental plan using Genetic Algorithm
""")


data = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

tabs = st.tabs(["Data Loading", "New Experimental Plan"])

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
        max_response = right.number_input(f"Targetted Maximum value of {response[0] if len(response) > 0 else 'response'}:", value=np.max(data[response]) if len(response) > 0 else 1, 
                    help=f"""The expected maximum and minimum values of the {response[0] if len(response) > 0 else 'response'} variable is used to normalize the data between 0 and 1.""")
        min_response = right.number_input(f"Minimum value of {response[0] if len(response) > 0 else 'response'}:", value=0, 
                    help=f"""The expected maximum and minimum values of the {response[0] if len(response) > 0 else 'response'} variable is used to normalize the data between 0 and 1.""")
        # data[response] needs to be normalized between 0 and 1
        data[response] = (data[response] - min_response) / (max_response - min_response)
        if len(response) > 0:
            response = response[0]
        data, encoders, dtypes = encode_data(data, factors)


with tabs[1]:
    if data is not None and len(factors) > 0 and len(response) > 0:
        left, right = st.columns([3,1])
        mutation_slider = right.slider("Mutation shrink factor (10^x)", 
                                    key="mutation_slider",
                                    min_value=-1., max_value=1., value=0., step=0.01,
                                    help="""### Purpose of Mutation Shrink Factor
                                    
Early in the algorithm, larger mutations help to explore a wide range of potential solutions by introducing significant variation. As the algorithm nears convergence, smaller mutations ensure that the solutions are fine-tuned without dramatically altering promising solutions.

### How it Works

The mutation rate or mutation magnitude is progressively decreased as a function of the number of generations or the current performance of the population.
Here, we chose to reduce the mutation rate by a constant factor.
""")
        new_pop, variables = ga.main(input_data=data.values,
                                    var_names=list(data),
                                    mutation_shrink_factor=10**mutation_slider)
        df_new = pd.DataFrame(new_pop, columns=variables)
        df_new[response] = ""
        df_new['run_order'] = np.arange(1, len(df_new)+1)
        cols = df_new.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df_new = df_new[cols]
        # reverse the encoding
        df_new = decode_data(df_new, factors, dtypes, encoders)
        timestamp = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        outfile = writeout(df_new)
        right.download_button(
            label     = f"Download new Experimental Design with {len(df_new)} runs",
            data      = outfile,
            file_name = f'newDOE_{timestamp}.csv',
            mime      = 'text/csv',
            key       = 'download-csv'
        )
        left.dataframe(df_new, hide_index=True)
        # plot the old design and the new one on top in red
        show_new = right.checkbox("Show new design in red on the scatter plot", key="show_scatter", value=True)
        cols = st.columns(len(factors)-1)
        if len(df_new.values)>0:
            plt.rcParams.update({'font.size': 22})
            for i,faci in enumerate(factors):
                for j,facj in enumerate(factors):
                    if j>i:
                        fig, ax = plt.subplots()
                        ax.scatter(data[facj], data[faci], s=100)
                        if show_new:
                            ax.scatter(df_new[facj], df_new[faci], s=100, color='red')
                        ax.set_ylabel(faci)
                        ax.set_xlabel(facj)
                        fig.tight_layout()
                        cols[j-1].pyplot(fig)