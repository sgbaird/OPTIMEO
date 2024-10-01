# DoE Designer, Optimizer, and Analyzer

---

## About this app

DOE-DOA is a web application that helps you design experiments, generate new experiments, and analyze the results of your experiments. It is designed to help you optimize your process by generating a Design of Experiment (DoE) and analyzing the results of your experiments. You can also generate new experiments to optimize your process, and analyze the results of your experiments.

This App was developed within the frame of an academic research project, MOFSONG, funded by the French National Research Agency (N° ANR-xxxxxx). See the related paper reference in [How to cite](#how-to-cite).

---

## Usage

Choose the page you want to use in the sidebar, and follow the instructions.

- **1. Design of Experiment:** Generate a Design of Experiment (DoE) for the optimization of your process. Depending on the number of factors and levels, you can choose between different types of DoE, such as Full Factorial, Fractional Factorial, or Definitive Screening Design. You can also generate a Latin Hypercube Sampling (LHS) for a Monte-Carlo simulation.
- **2. New experiments:** From a previous set of experiments and their results, generate a new set of experiments to optimize your process. You can choose the sampler you want to use, depending on if you are on the early stages of the optimization and want to explore the phase space (then, chose the genetic algorithm sampler), or if you want to efficiently find the maximum of minimum in the response (then choose the TPE one).
- **3. Data analysis and modeling:** Analyze the results of your experiments and model the response of your process. You can use different types of regression models, such as linear, polynomial, or Random Forest. You can also use the trained model to predict the response of your process for new experiments.

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

- [pyDOE3](https://github.com/relf/pyDOE3)
- [dexpy](https://statease.github.io/dexpy/)
- [doepy](https://doepy.readthedocs.io/en/latest/)
- [definitive-screening-design](https://pypi.org/project/definitive-screening-design/)

---

## License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>
