# run with "streamlit run Home.py"

import streamlit as st
from resources.functions import *
from resources.functions import about_items

st.set_page_config(
    page_title="OPTIMEO",
    page_icon="resources/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=about_items
)

style = read_markdown_file("resources/style.css")
st.markdown(style, unsafe_allow_html=True)

cols=st.columns([4,1])
cols[0].title("OPTIMEO")
cols[0].subheader("Optimization Platform for Tuning, Inference, Modeling, Exploration, and Orchestration")
cols[1].write('')
cols[1].write('')
cols[1].image("resources/logo.png", width=200)
st.markdown("""
<a href="https://doi.org/10.5281/zenodo.15308437"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15308437.svg" alt="DOI"></a>
[![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/colinbousige/OPTIMEO)

---

- [About this app](#about-this-app)
- [Usage](#usage)
  - [With the web app](#with-the-web-app)
  - [With the Python package](#with-the-python-package)
- [Support](#support)
- [How to cite](#how-to-cite)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## About this app

[OPTIMEO](https://optimeo.streamlit.app/) is a package doubled by a web application that helps you optimize your experimental process by generating a Design of Experiment (DoE), generating new experiments using Bayesian Optimization, and analyzing the results of your experiments using Machine Learning models.

This app was developed within the frame of an academic research project, MOFSONG, funded by the French National Research Agency (N° ANR-24-CE08-7639). See the related paper reference in [How to cite](#how-to-cite).

---

## Usage

### With the web app

You can use the app directly on its [Streamlit.io web page](https://optimeo.streamlit.app/), or run it locally (see [Installation](https://colinbousige.github.io/OPTIMEO/optimeo.html#installation)).

Choose the page you want to use in the sidebar, and follow the instructions. Hover the mouse on the question marks to get more information about the parameters.

**1. Design of Experiment:**  
Generate a Design of Experiment (DoE) for the optimization of your process. Depending on the number of factors and levels, you can choose between different types of DoE, such as Sobol sequence, Full Factorial, Fractional Factorial, or Definitive Screening Design.

**2. New experiments using Bayesian Optimization:**  
From a previous set of experiments and their results, generate a new set of experiments to optimize your process. You can choose the sampler you want to use, depending on if you are on the early stages of the optimization and want to explore the phase space (then, choose the Sobol pseudo-random generator), or if you want to efficiently find the maximum of minimum in the response (then choose the Bayesian Optimization one).  

**3. Data analysis and modeling:**  
Analyze the results of your experiments and model the response of your process.

### With the Python package

You can also use the app as a Python package (see [Documentation](https://colinbousige.github.io/OPTIMEO/optimeo.html)).

---

## Support

This app was made by [Colin Bousige](mailto:colin.bousige@cnrs.fr). Contact me for support or to signal a bug, or leave a message on the [GitHub page of the app](https://github.com/colinbousige/OPTIMEO).

---

## How to cite

The source can be found [on Github](https://github.com/colinbousige/optimeo), please consider citing as:

```bibtex
@software{Bousige_optimeo,
    author = {Bousige, Colin},
    title = {{OPTIMEO}},
    url = {https://github.com/colinbousige/optimeo},
    doi = {10.5281/zenodo.15308437}
}
```

---

## Documentation

The package documentation is available on the [GitHub page of the app](https://colinbousige.github.io/OPTIMEO/optimeo.html).

---

## Acknowledgements

This work was supported by the French National Research Agency (N° ANR-24-CE08-7639).  
Also, this work was made possible thanks to the following open-source projects:

- [ax](https://ax.dev/)
- [BoTorch](https://botorch.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pyDOE3](https://github.com/relf/pyDOE3)
- [dexpy](https://statease.github.io/dexpy/)
- [doepy](https://doepy.readthedocs.io/en/latest/)
- [definitive-screening-design](https://pypi.org/project/definitive-screening-design/)

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


""", unsafe_allow_html=True)

