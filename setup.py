from setuptools import setup, find_packages

setup(
    name='optima',
    version='0.1.0',
    packages=find_packages(include=['optima', 'optima.*']),
    install_requires=[
        "streamlit==1.44.1",
        "ax-platform==0.5.0",
        "seaborn",
        "definitive_screening_design==0.5.0",
        "dexpy==0.12",
        "doepy==0.0.1",
        "matplotlib==3.10.1",
        "numpy",
        "pandas",
        "plotly",
        "pyDOE3==1.0.4",
        "pyjanitor==0.31.0",
        "scikit_learn==1.6.1",
        "statsmodels",
        "xlsxwriter==3.2.2"
    ],
    entry_points={
        'console_scripts': [
            # If you have any command-line scripts, list them here, e.g.,
            # 'optima-cli=optima.cli:main',
        ],
    },
    author='Colin Bousige',
    author_email='colin.bousige@cnrs.fr',
    description='OPTIMA: A Python package for optimization and design of experiments',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/colinbousige/OPTIMA',  # Replace with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial 4.0 International License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)