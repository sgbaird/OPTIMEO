---
title: 'OPTIMEO: Optimization Platform for Tuning, Inference, Modeling, Exploration, and Orchestration'
tags:
  - Python
  - Bayesian Optimization
  - Design of Experiment
  - Machine Learning
  - Active Learning
authors:
  - name: Colin Bousige
    orcid: 0000-0002-0490-2277
    affiliation: "1"
    email: colin.bousige@cnrs.fr
affiliations:
 - name: Universite Claude Bernard Lyon 1, CNRS, LMI UMR 5615, Villeurbanne, F-69100, France
   index: 1
   ror: 03cfem402
date: 2025-04-30
bibliography: paper.bib
---

# Summary

We present `OPTIMEO`, an Optimization Platform for Tuning, Inference, Modeling, Exploration, and Orchestration.
`OPTIMEO` is a python package doubled by a web application that helps the user optimize their experimental process by generating a design of experiment (DoE), generating new experiments using Bayesian optimization, and analyzing/predicting the results of their experiments using machine learning (ML) models.
The package and the web application are designed to be user-friendly and accessible to researchers and students alike, providing a powerful tool for optimizing experimental processes in various research fields.
The Bayesian optimization part is based on the `ax-platform` package [@bakshy_ae_2018], which in turn is based on `BoTorch` [@balandat_botorch_2020].
It therefore allows for numerical and categorical variables, as well as multi-objective optimization.
To use the web application, no knowledge of python is required: the user can run it on the streamlit.io web page, [https://optimeo.streamlit.app/](https://optimeo.streamlit.app/), or run it locally on their machine for better performance.
The user can simply (1) select the desired options to generate a DoE; or (2) upload their data and perform Bayesian optimization to look for optimal parameters to minimize/maximize their experimental response; or (3) upload their data and plot it, assess possible correlations between variables, as well as run various regression models using `scikit-learn` [@pedregosa_scikitlearn_2011] to make predictions and analyze the importance of each feature on their experimental process.
Extensive yet accessible descriptions of the different options are provided in the web application to help the user understand what they are doing.
Using the package in python is also possible and it offers more versatility, like the possibility to make an optimization loop (in case experiments and their characterizations are done by a robot, for example) or to provide more parameters to the ML models -- the heavy lifting is done under the hood by the package (e.g. categorical variables encoding and decoding, data formatting, normalizations, workflows, etc.).

# Statement of need

Experimental processes are often complex and time-consuming, requiring careful planning and execution to achieve reliable results.
In many cases, researchers must conduct multiple experiments to optimize their processes, which can be both costly and time-consuming.
The traditional approach to experimental design often relies on trial and error, resulting in inefficient use of resources and time.
In recent years, there has been a growing interest in using advanced computational techniques to optimize experimental processes using active learning techniques such as genetic algorithms or Bayesian optimization.
For example, `Sycofinder` [@moosavi_capturing_2019;@talirz_ltalirz_2019], an application with web-based User Interface (UI) was developed to optimize the synthesis of metal-organic frameworks (MOFs) using a genetic algorithm, and Bayesian optimization was used to find optimal synthesis parameters -- the optimization target can be either the yield and quality of the synthesized material as well as its physical properties or performances [@lambard_optimization_2022;@matsuda_datadriven_2022;@osada_adaptive_2020].
While these techniques have now proven their effectiveness in various fields, they often require a deep understanding of the underlying algorithms and programming skills to be implemented effectively.
This can be a barrier for many researchers, especially those without a strong background in computer science or data analysis -- which is often the case for experimentalists.
To address these challenges, we present `OPTIMEO`, a powerful tool that streamlines the experimental process by providing a comprehensive platform for design of experiment, Bayesian optimization, and machine learning analysis.

# State of the Field

Bayesian optimization is known for its data efficiency, meaning it can find optimal solutions with fewer evaluations compared to other methods.
However, it can become computationally expensive as the number of features and evaluations increases, leading to longer computation times [@lan_time_2022].
Genetic algorithms, on the other hand, are generally faster in terms of computation time per evaluation but may require more evaluations to converge to an optimal solution [@lan_time_2022].

However, the trade-off between data efficiency, computation time, and experimentation time is a key consideration when choosing between these two optimization methods.
The `OPTIMEO` package is aimed at helping scientists of any field to reach the optimum parameters of their process using the minimum amount of ressources and effort.
Therefore, it is based on Bayesian optimization for its data efficiency: when each experiment might take one or more day to run and characterize, to minimize the number of experiments to run before reaching optimal parameterization, it is much preferable to use Bayesian optimization compared to a genetic algorithm, even if it necessitates a few more minutes to run the algorithm to find which new experiments to run.

There are several free and open source software that provide similar functionality to `OPTIMEO`, such as `AutoOED` [@tian_autooed_2021], `BOXVIA` [@ishii_boxvia_2022], and `MADGUI` [@bajan_madgui_2025].
All three rely on Bayesian optimization to minimize the number of experimental evaluations, and are available via executables or source code.
However, none of them provide the ability to build an experimental design, which, from interacting with our experimentalist colleagues, is a feature they often need and highly appreciate -- it gives a good idea of where to start an optimization process from scratch.
Also, while we agree that it's probably very suggestive, we feel that they lack the usability and accessibility that we aim to provide with our UI.
For example, the upper and lower bounds of the features need to be entered manually in these packages -- which is a tedious task.
These values are automatically set (while editable) in `OPTIMEO` based on the data provided by the user.
Finally, these programs are not available as standalone Python packages for more advanced use.
Such advanced usage might be required if the user wants to run the optimization loop in a robotic high-throughput lab, for example -- which is why this package was developed in the first place.

All in all, while other software is available to perform some of the tasks of `OPTIMEO`, this package offers a unique concatenation of features that makes it a powerful and versatile tool for optimizing experimental processes.

# Acknowledgements

This work was supported by the French National Research Agency (N° ANR-24-CE08-7639).
It would also not have been possible without the development of the open source packages `ax-platform` [@bakshy_ae_2018], `BoTorch` [@balandat_botorch_2020], `scikit-learn` [@pedregosa_scikitlearn_2011], `pyDOE3` [@tisimst_relf_2024], `dexpy` [@_statease_2025], `doepy` [@dirkaudaz_dirkaudaz_2018], and `definitive-screening-design` [@ongari_danieleongari_2024].

# References