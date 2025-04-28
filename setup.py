from setuptools import setup, find_packages

setup(
    name='optimeo',
    version='0.1.0',
    packages=find_packages(include=['optimeo', 'optimeo.*']),
    install_requires=[
        "streamlit",
        "ax-platform",
        "seaborn",
        "definitive_screening_design",
        "dexpy",
        "doepy",
        "matplotlib",
        "numpy",
        "pandas",
        "plotly",
        "pyDOE3",
        "pyjanitor",
        "scikit_learn",
        "statsmodels",
        "xlsxwriter"
    ],
    author='Colin Bousige',
    author_email='colin.bousige@cnrs.fr',
    description='OPTIMA: A Python package for optimization and design of experiments',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/colinbousige/OPTIMEO',  # Replace with your project's URL
    scripts=['bin/optimeo'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial 4.0 International License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)