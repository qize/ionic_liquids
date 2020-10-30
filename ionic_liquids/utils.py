# default python modules
import os
from datetime import datetime
# external packages
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from rdkit import Chem
# internal modules
from methods import methods
from mordred import Calculator, descriptors

def use_mordred(mols, descs=None):
    if descs is None:
        calc = Calculator(descriptors, ignore_3D=True)
        df = pd.DataFrame(calc.pandas(mols,quiet=True).fill_missing().dropna(axis='columns'),dtype=np.float64)
        return df
    else:
        calc = Calculator(descriptors, ignore_3D=True)
        calc.descriptors = [d for d in calc.descriptors if str(d) in descs]
        df = pd.DataFrame(calc.pandas(mols,quiet=True).fill_missing().dropna(axis='columns'),dtype=np.float64)
        return df
#        print(descs)
#        return df[descs]

def train_model(model, data_file, test_percent, save=True):
    """
    Choose the regression model

    Input
    ------
    model: string, the model to use
    data_file: dataframe, cleaned csv data
    test_percent: float, the percentage of data held for testing

    Returns
    ------
    obj: objective, the regressor
    X: dataframe, normlized input feature
    y: targeted electrical conductivity

    """
    df, y_error = read_data(data_file)
    X, y = molecular_descriptors(df)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=(test_percent/100))
    X_train, X_mean, X_std = normalization(X_train)

    model = model.replace(' ', '_')
    # print("training model is ",model)
    if (model.lower() == 'lasso'):
        obj = methods.do_lasso(X_train, y_train)
    elif (model.lower() == 'mlp_regressor'):
        obj = methods.do_MLP_regressor(X_train, y_train)
    elif (model.lower() == 'svr'):
        obj = methods.do_svr(X_train, y_train)
    else:
        raise ValueError('Invalid model type!')

    return obj, X_train, y_train, X_mean, X_std


def normalization(data, means=None, stdevs=None):
    """
    Normalizes the data using the means and standard
    deviations given, calculating them otherwise.
    Returns the means and standard deviations of columns.

    Inputs
    ------
    data : Pandas DataFrame
    means : optional numpy argument of column means
    stdevs : optional numpy argument of column st. devs

    Returns
    ------
    normed : the normalized DataFrame
    means : the numpy row vector of column means
    stdevs : the numpy row vector of column st. devs

    """
    cols = data.columns
    data = data.values

    if (means is None) or (stdevs is None):
        means = np.mean(data, axis=0)
        stdevs = np.std(data, axis=0, ddof=1)
    else:
        means = np.array(means)
        stdevs = np.array(stdevs)

    # handle special case of one row
    if (len(data.shape) == 1) or (data.shape[0] == 1):
        for i in range(len(data)):
            data[i] = (data[i] - means[i]) / stdevs[i]
    else:
        for i in range(data.shape[1]):
            data[:,i] = (data[:,i] - means[i]*np.ones(data.shape[0])) / stdevs[i]

    normed = pd.DataFrame(data, columns=cols)

    return normed, means, stdevs


def predict_model(A_smile, B_smile, obj, t, p, m, X_mean, X_stdev, flag=None):
    """
    Generates the predicted model data for a mixture
    of compounds A and B at temperature t and pressure p.

    Inputs
    -----
    A_smile : SMILES string for compound A
    B_smile : SMILES string for compound B
    obj : model object
    t : float of temperature
    p : float of pressure
    m : float of mol_fraction
    X_mean : means of columns for normalization
    X_stdev : stdevs fo columns for normalization
    flag : string to designate which variable is on x-axis

    Returns
    ------
    x_vals : x-values chosen by flag
    y_pred : predicted conductivity (y_values)

    """
    N = 100 # number of points

    y_pred = np.empty(N+1)
    if (flag == 'm'):
        x_conc = np.linspace(0, 1, N+1)
    elif (flag == 't'):
        x_conc = np.linspace(100, 400, N+1)
    elif (flag == 'p'):
        x_conc = np.linspace(5, 400, N+1)
    else:
        raise ValueError("unexpected flag")
    for i in range(len(x_conc)):
        if (flag == 'm'):
            my_df = pd.DataFrame({'A': A_smile, 'B': B_smile, 'MOLFRC_A': x_conc[i], \
                                'P': p, 'T': t, 'EC_value': 0}, index=[0])
        elif (flag == 't'):
            my_df = pd.DataFrame({'A': A_smile, 'B': B_smile, 'MOLFRC_A': m, \
                                 'P': p, 'T': x_conc[i], 'EC_value': 0}, index=[0])
        elif (flag == 'p'):
            my_df = pd.DataFrame({'A': A_smile, 'B': B_smile, 'MOLFRC_A': m, \
                                 'P': x_conc[i], 'T': t, 'EC_value': 0}, index=[0])
        X, trash = molecular_descriptors(my_df)
        X, trash, trash = normalization(X, X_mean, X_stdev)
        y_pred[i] = obj.predict(X)


    return x_conc, y_pred


def molecular_descriptors(data,descs):
    """
    Use RDKit to prepare the molecular descriptor

    Inputs
    ------
    data: dataframe, cleaned csv data

    Returns
    ------
    prenorm_X: normalized input features
    Y: experimental electrical conductivity

    """

    Y = data['Tm']
    mols = list(map(Chem.MolFromSmiles,data['SMILES'].values))
    X = use_mordred(mols,descs)

    return X, Y


def read_data(filename):
    """
    Reads data in from given file to Pandas DataFrame

    Inputs
    -------
    filename : string of path to file

    Returns
    ------
    df : Pandas DataFrame
    y_error : vector containing experimental errors

    """
    cols = filename.split('.')
    name = cols[0]
    filetype = cols[1]
    if (filetype == 'csv'):
        df = pd.read_csv(filename)
    elif (filetype in ['xls', 'xlsx']):
        df = pd.read_excel(filename)
    else:
        raise ValueError('Filetype not supported')

    # clean the data if necessary
    df = df.drop(df[df['SMILES'] == 'XXX'].index).reset_index(drop=True)
    y_error = np.copy(df['dev'])
    df = df.drop('dev', 1)
    df = df.drop('name', 1)

    return df, y_error


def save_model(obj, X_mean, X_stdev, X=None, y=None, dirname='default'):
    """
    Save the trained regressor model to the file

    Input
    ------
    obj: model object
    X_mean : mean for each column of training X
    X_stdev : stdev for each column of training X
    X : Predictor matrix
    y : Response vector
    dirname : the directory to save contents

    Returns
    ------
    None
    """
    if (dirname == 'default'):
        timestamp = str(datetime.now())[:19]
        dirname = 'model_'+timestamp.replace(' ', '_')
    else:
        pass
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = dirname + '/model.pkl'
    joblib.dump(obj, filename)

    joblib.dump(X_mean, dirname+'/X_mean.pkl')
    joblib.dump(X_stdev, dirname+'/X_stdev.pkl')

    if (X is not None):
        filename = dirname + '/X_data.pkl'
        joblib.dump(X, filename)
    else:
        pass

    if (y is not None):
        filename = dirname + '/y_data.pkl'
        joblib.dump(y, filename)
    else:
        pass

    return


def read_model(file_dir):
    """
    Read the trained regressor to
    avoid repeating training.

    Input
    ------
    file_dir : the directory containing all model info

    Returns
    ------
    obj: model object
    X_mean : mean of columns in training X
    X_stdev : stdev of columns in training X
    X : predictor matrix (if it exists) otherwise None
    y : response vector (if it exists) otherwise None

    """
    filename = file_dir + '/model.pkl'
    obj = joblib.load(filename)
    X_mean = joblib.load(file_dir+'/X_mean.pkl')
    X_stdev = joblib.load(file_dir+'/X_stdev.pkl')

    try:
        X = joblib.load(file_dir + '/X_data.pkl')
    except:
        X = None
    try:
        y = joblib.load(file_dir + '/y_data.pkl')
    except:
        y = None

    return obj, X_mean, X_stdev, X, y
