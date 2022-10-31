# Higgs boson prediction
### By Daniel Cavaleri, Bilel El-Ghallali and Gaelle Pillon
## Machine learning project

The purpose of this project is to predict the nature
of a set of collision events. The outcome is either 
a signal (Higgs boson) or a background (something else).
***
### How to use the model :

### Main files
  * __implementations.py__ contains multiple useful functions 
  to make the predictions. More precisely, it contains everything we need for data
  processing and different machine learning algorithms.
  We implemented all 6 functions from Table 1 (project description) and an additional
  a Newton method discussed during the lectures.
  * __run.py__ is the main script that was used to generate our predictions.
  * __helpers.py__ regroups basic functions for .csv importation and generation.
  * __data.zip__ .csv files regrouping different data sets used for the prediction and the submission.
### The prediction
  To get a final prediction of 73.7%, we used Ridge Regression
  method with a lambda = 10^-9.
  We applied some data processing beforehand, performed a polynomial feature augmentation without cross-interactions
  and split the data.
  You can achieve the same results by running the script in __run.py__.

  