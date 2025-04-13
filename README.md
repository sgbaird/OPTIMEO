# DoE Designer, Optimizer, and Analyzer <img src="ressources/logo.png" alt="Logo" width="100"/>

---

## About this app

[DOE-DOA](https://doe-doa.streamlit.app/) is a web application that helps you design experiments, generate new experiments, and analyze the results of your experiments. It is designed to help you optimize your process by generating a Design of Experiment (DoE) and analyzing the results of your experiments. You can also generate new experiments to optimize your process, and analyze the results of your experiments.

This App was developed within the frame of an academic research project, MOFSONG, funded by the French National Research Agency (N° ANR-xxxxxx). See the related paper reference in [How to cite](#how-to-cite).

---

## Usage

Choose the page you want to use in the sidebar, and follow the instructions. Hover the mouse on the question marks to get more information about the parameters.

**1. Design of Experiment:**  
Generate a Design of Experiment (DoE) for the optimization of your process. Depending on the number of factors and levels, you can choose between different types of DoE, such as Full Factorial, Fractional Factorial, or Definitive Screening Design.

**2. New experiments using Bayesian Optimization:**  
From a previous set of experiments and their results, generate a new set of experiments to optimize your process. You can choose the sampler you want to use, depending on if you are on the early stages of the optimization and want to explore the phase space (then, choose the Sobol pseudo-random generator), or if you want to efficiently find the maximum of minimum in the response (then choose the Bayesian Optimization one).  

**3. Data analysis and modeling:**  
Analyze the results of your experiments and model the response of your process.

---

## Installation

You can use the app directly on its [Streamlit.io web page](https://doe-doa.streamlit.app/).

If you'd rather run this app on your local machine (which will most probably make it faster than running it on streamlit.io), you need to have Python installed. You can download it [here](https://www.python.org/downloads/).

Then, you can install the required packages by running the following command in your terminal:

```bash
git clone https://github.com/colinbousige/DOE-DOA.git
cd DOE-DOA
pip install -r requirements.txt
```

Finally, you can run the app by running the following command in your terminal:

```bash
streamlit run Home.py
```

---

## Support

This app was made by [Colin Bousige](mailto:colin.bousige@cnrs.fr). Contact me for support or to signal a bug, or leave a message on the [GitHub page of the app](https://github.com/colinbousige/DOE-DOA).

---

## How to cite

This work is related to the article "xxxx". Please cite this work if you publish using this code:

```bibtex
@article{xxx,
    title = {xxx},
    author = {xxx},
    journal = {xxx},
    volume = {xxx},
    year = {xxx},
    pages = {xxx}
    doi = {xxx}
}
```

The source can be found [on Github](https://github.com/colinbousige/DOE-DOA), please consider citing it too:

```bibtex
@software{Bousige_DOE-DOA,
    author = {Bousige, Colin},
    title = {{DOE-DOA}},
    url = {https://github.com/colinbousige/DOE-DOA},
    doi = {xxxx}
}
```

---

## Acknowledgements

This work was supported by the French National Research Agency (N° ANR-xxxxxx).  
Also, this work was made possible thanks to the following open-source projects:

- [ax](https://ax.dev/)
- [pyDOE3](https://github.com/relf/pyDOE3)
- [dexpy](https://statease.github.io/dexpy/)
- [doepy](https://doepy.readthedocs.io/en/latest/)
- [definitive-screening-design](https://pypi.org/project/definitive-screening-design/)

---

## License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>
