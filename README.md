
<p align="center">
  <img src="reports/readme_figs/banner.png" width="950" title="Project Banner">
</p>

[![DOI](https://zenodo.org/badge/259046250.svg)](https://zenodo.org/badge/latestdoi/259046250) 
[![Coverage Status](https://coveralls.io/repos/github/aalling93/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size/badge.svg?branch=master)](https://coveralls.io/github/aalling93/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size?branch=master) 
[![Build Status](https://app.travis-ci.com/aalling93/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size.svg?branch=master)](https://app.travis-ci.com/aalling/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size) 
![Repo Size](https://img.shields.io/github/repo-size/aalling93/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size) 
[![Known Vulnerabilities](https://snyk.io/test/github/aalling93/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size//badge.svg)](https://snyk.io/test/github/aalling93/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size/)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

Kristian Aalling Sørensen

kaaso@space.dtu.dk

# Brief Description
 <a class="anchor" id="Introduction"></a>

In Deep Quantile Regression Synthetic Aperture Radar Ship Size, the objective is to determine ship paramters from Synthetic Aperture Radar (SAR) images, i.e., ship length and ship width. However, the techinque could be fully extended to, e.g., direction, speed, or to classification purposes such as ship type. 

Ship detection and charecerisation is an important part of increasing maritime security and surveillance and when securing maitime sovereignty. For decision makers (local navy, other stakeholder), it is important to know where and how many Dark Ships (Ships that do not transmit self-transmitted information such as AIS data) are present in their waters. Furhtermore, knowing as much information as possible for a found Dark Ships is important when tasking other missions. Missions such as Satellite Tasking, sending out the coast-guard ect. In short, it is important to know:
- is there a dark ships
- how big is the ship (length/width). You act very differently for different ships
- what type of ship is it
- what is it doing.

In the Arctic, you can not use Multi Spectral Images, as you see on Google Earth. In part due to clouds, and partly due to the lack of illumination. You would therefore a SAR instrument. The SAR provides its own illumination and the returned signals (amplitude and phase) is dependant on both the geometrical and underlying physical charectgerstics. However, SAR iamges are subjucated to a great deal of Speckle/Clutter/ other stuff.  The same target can consequently look very different only different images.

In this application, we therefore try to estiamte the 99 % Confidence Interval for an estiamte ship length. This would provide valuable information on the length of the ship. 
For instance, if we have a length estiamtion of 200 m with a Confidence Interval of [190-210] we are quite certain it is a good estiamte.
Conversly, if we have a length estiamte of 200 m with a Confidense Interval of [30-500], we are not very certain. 

Specifically, deterministic models might over estiamte the length to be, e.g., 800 m long which is impossible. If we had an automatic way of saying "this is a bad estiamte", we could just neglegt it somewhat. 


In The Figure below, we see t2o estimates from our model. One of which is clear and the other is not. We see  how the 95CI varies greatly.



<p align="center">
  <img src="reports/readme_figs/ex_good.png" width="450" title="Example of good CI99">
  <img src="reports/readme_figs/ex_bad.png" width="450" title="Example of poor CI99">
</p>





# Table Of Contents
-  [Introduction](#Introduction)
-  [Requirements](#Requirements)
-  [Install and Run](#Install-and-Run)
-  [Overview of results](#Overview-of-results)
-  [Model](#Model)
-  [Results](#Results)
-  [Code structure](#Code-structure)
-  [Usage](#Usage)
-  [Acknowledgments](#Acknowledgments)

# Requirements
 <a class="anchor" id="Requirements"></a>

- [pandas](https://github.com/pandas) (Data wrangling)
- [tensorflow](https://github.com/tensorflow) (The models)
- [tensorflow_probability](https://www.tensorflow.org/probability) (For probability stuff)
- [sklearn](https://scikit-learn.org/stable/) (Used specifically for Cross Validation of the model)
- [scipy](https://scipy.org) (stats when analysing the model)






# Install and Run
 <a class="anchor" id="Install-and-Run"></a>


 1. 
  install Requirements

 2. 
Clone the Repo
 ```
 git clone https://github.com/aalling93/Deep-Quantile-Regression-Synthetic-Aperture-Radar-Ship-Size.git
 ```

  3. 
 Profit

# Overview of results
 <a class="anchor" id="Overview-of-results"></a>



# Model
 <a class="anchor" id="Model"></a>

Coming soon. 
# Results
 <a class="anchor" id="Results"></a>

Coming soon. 
# Code structure
 <a class="anchor" id="Code-structure"></a>

Coming soon. 
------------
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks         
    │                      
    │                       
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

<p><small>(Project structure based on cookiecutter)</small></p>

--------



# Usage
 <a class="anchor" id="Usage"></a>

 1. Import Model
 --------------------
 ```python
from src.models.model import Model
from src.models._util import seed_everything
from src.analyse_model.analyse import Analyse_model

 ```

 2. Load Model
 -----------------
 ```python 
with Model() as M:
    M.data_load(dataset["images"],dataset["metadata"],dataset["targets"])
    M.model_load(which_model='quantile')
 ```


  3. Analyse model results
 -----------------

  ```python 
with Analyse_model(M.model.model) as anal:
    anal.get_data(
        [
            dataset["testing_images"],
            dataset["testing_metadata"],
            dataset["testing_targets"],
        ]
    )
    anal.get_scaling(scaling_vars)
    anal.inverse_values()
    anal.all_estimates_inverse()
 ```



# Acknowledgments
 <a class="anchor" id="Acknowledgments"></a>
Donno yet. 

 # Licence
 [![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)

