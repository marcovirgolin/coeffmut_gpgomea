'''
THIS CODE IS AN ALMOST-EXACT COPY OF https://github.com/laic-ufmg/gp-benchmarks/blob/master/gp_benchmarks_meta_features.ipynb
ALL CREDIT GOES TO THE AUTHORS OF THAT PAPER
'''

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import re
import timeit

from math import sqrt, sin, cos, log, pi, e

from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import os

def get_data(name, url, rnd=None, pd_sep=',', pd_header=None, pd_skiprows=None, dataset=None):
    print("Reading the \"" + name + "\" dataset")
    if dataset == "BOH":
        from sklearn.datasets import load_boston
        boston = load_boston()
        df = pd.DataFrame(boston['data'])
        df = pd.concat([df, pd.Series(boston['target'])], axis=1)
    elif dataset == "CCP":
        # Get the file object from an url
        r = requests.get(url)
        # Create a ZipFile object from it
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # Read from a xlsx file inside the zip file
        df = pd.read_excel(z.open('CCPP/Folds5x2_pp.xlsx'))
    elif dataset == "CST":
        df = pd.read_excel(url)
    elif dataset == "ENC":
        df = pd.read_excel(url)
        # Drop Y1
        df.drop("Y1", axis=1, inplace=True)
    elif dataset == "ENH":
        df = pd.read_excel(url)
        # Drop Y2
        df.drop("Y2", axis=1, inplace=True)
    elif dataset == "YAC":
        # Get the data as text
        text = requests.get(url).text
        # Split in rows (remove the last line)
        e_re = re.compile("\s*\n\s*")
        rows = e_re.split(text)[:-1]
        e_re = re.compile(" +")
        # Split cells per row
        df = pd.DataFrame([e_re.split(row) for row in rows])
    else:
        df = pd.read_csv(url, header=pd_header, sep=pd_sep, skiprows=pd_skiprows)
        if dataset == "ABA":
            # Get dummy variables for the first column
            df_dummies = pd.get_dummies(df.iloc[:,0])
            # Drop the first column
            df.drop(df.columns[0], axis=1, inplace=True)
            # Concatenate the dummy variables with the data
            df = pd.concat([df_dummies, df], axis=1)
            df = df.sample(500, random_state=rnd, axis=0)
        elif dataset == "CPU":
            # Drop the first two columns
            df.drop(df.columns[[0,1]], axis=1, inplace=True)
        elif dataset == "FFR":
            df.drop(["month", "day"], axis=1, inplace=True)
        elif dataset == "OZO":
            # Imputation (replance NaN's by the mean of the column)
            df.fillna(df.mean(), inplace=True)
    return df.apply(np.float64)

seed = 4567
rnd = np.random.RandomState(seed)
# Loading real-world datasets
data_real = {"abalone": get_data('Abalone', 
                                 'https://archive.ics.uci.edu/ml/'
                                 'machine-learning-databases/abalone/abalone.data',
                                 rnd=rnd, dataset='ABA'),
            "airfoil": get_data('Airfoil',
                                'https://archive.ics.uci.edu/ml/'
                                'machine-learning-databases/00291/airfoil_self_noise.dat',
                                pd_sep="\t"),
            "boston": get_data('Boston', "", dataset="BOH"),
            "combined-cycle": get_data('Combined-cycle', 
                                       'https://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/00294/CCPP.zip',
                                       dataset="CCP"),
            "computer-hardware": get_data('Computer-hardware', 
                                          'https://archive.ics.uci.edu/ml/'
                                          'machine-learning-databases/cpu-performance/machine.data',
                                          dataset="CPU"),
            "concrete-strength": get_data('Concrete-strength', 
                                          'https://archive.ics.uci.edu/ml/'
                                          'machine-learning-databases/concrete/compressive/Concrete_Data.xls',
                                          dataset="CST"),
            "energy-cooling": get_data('Energy-cooling', 
                                       'http://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/00242/ENB2012_data.xlsx',
                                       dataset="ENC"),
            "energy-heating": get_data('Energy-heating', 
                                       'http://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/00242/ENB2012_data.xlsx',
                                       dataset="ENH"),
            "forest-fire": get_data('Forest-fire', 
                                    'https://archive.ics.uci.edu/ml/'
                                    'machine-learning-databases/forest-fires/forestfires.csv',
                                    pd_header='infer', dataset="FFR"),
            #"ozone": get_data('Ozone', './data/ozone.data', dataset="OZO"),
            "wine-quality-red": get_data('Wine-quality-red', 
                                         'https://archive.ics.uci.edu/ml/'
                                         'machine-learning-databases/wine-quality/winequality-red.csv',
                                         pd_header='infer', pd_sep=';'),
            "wine-quality-white": get_data('Wine-quality-white', 
                                           'https://archive.ics.uci.edu/ml/'
                                           'machine-learning-databases/wine-quality/winequality-white.csv', 
                                           pd_header='infer', pd_sep=';'),
            "yacht": get_data('Yacht', 'https://archive.ics.uci.edu/ml/'
                              'machine-learning-databases/00243/yacht_hydrodynamics.data', dataset="YAC")
            }

import joblib 
joblib.dump(data_real, "data_rw.joblib", compress=3)