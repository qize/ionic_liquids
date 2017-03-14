import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

FIG_SIZE = (4,4)

def parity_plot(y_pred,y_act):
    """
    Creates a parity plot

    Input
    -----
    y_pred : predicted values from the model 
    y_act : 'true' (actual) values

    Output
    ------
    fig : matplotlib figure

    """

<<<<<<< HEAD
    fig = plt.figure(figsize=FIG_SIZE)
    plt.scatter(y_act,y_pred)
    plt.plot([y_act.min(),y_act.max()],[y_act.min(),y_act.max()],lw=4,color='r')
=======
    fig = plt.figure(figsize=(4,4))
    plt.scatter(y_act,y_pred, alpha=0.5)
    plt.plot([y_act.min(),y_act.max()],[y_act.min(),y_act.max()],lw=4,'r')
>>>>>>> master
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    return fig 


def train_test_error(e_train,e_test,model_params):
    """
    Creates a plot of training vs. test error

    Input
    -----
    e_train : numpy array of training errors
    e_test : numpy array of test errors
    model_params : independent parameters of model (eg. alpha in LASSO)

    Returns
    -------
    fig : matplotlib figure
    """

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(model_params,e_train,label='Training Set')
    plt.plot(model_params,e_train,label='Test Set')
    plt.xlabel('Model Parameter')
    plt.ylabel('MSE of model')
    plt.legend()

    return fig


