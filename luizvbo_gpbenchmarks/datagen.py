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



def synthetic_gen(name, rnd, function, training_gen, test_gen=None):
    """Generates training and test sets for synthetic datasets.
    """
    print("Generating the \"" + name + "\" dataset")
    training_set = []
    test_set = []
    for i in range(training_gen.n):
        inst = training_gen.generate(rnd)
        training_set.append(inst + [function(*inst)])
    if test_gen:
        for i in range(test_gen.n):
            inst = test_gen.generate(rnd)
            test_set.append(inst + [function(*inst)])
    else:
        test_set = training_set.copy()
    return {"training": pd.DataFrame(training_set),
            "test": pd.DataFrame(test_set)}

class U:
    def __init__(self, ini, end, n):
        self.ini = ini
        self.end = end
        self.n = n
    
    def generate(self, rnd):
        return [rnd.uniform(ini, end) for ini, end in zip(self.ini, self.end)]

class E:
    def __init__(self, ini, end, step):
        self.ini = ini
        self.end = end
        self.step = step
        
        mesh = np.meshgrid(*[np.arange(ini, end+step, step) 
                           for ini, end, step in zip(self.ini, self.end, self.step)])
        self.points = [dim.reshape(1,-1)[0] for dim in mesh]
        self.index = 0
        self.n = len(self.points[0])
    
    def generate(self, rnd):
        inst = [self.points[i][self.index] for i in range(len(self.points))]
        self.index += 1
        return inst

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

seed = 1234
rnd = np.random.RandomState(seed)
# Loading synthetic datasets
data_synt = {"meier-3": synthetic_gen("Meier-3", rnd, 
                                      lambda x_1,x_2: (x_1**2*x_2**2)/(x_1+x_2), 
                                      U([-1, -1], [1, 1], 50), U([-1, -1], [1, 1], 50)),
            "meier-4": synthetic_gen("Meier-4", rnd, 
                                     lambda x_1,x_2: x_1**5/x_2**3, 
                                     U([-1, -1], [1, 1], 50), U([-1, -1], [1, 1], 50)),
            "nonic": synthetic_gen("Nonic", rnd,
                                   lambda x_1: sum([x_1**i for i in range(1,10)]), 
                                   E([-1], [1], [2/19]), U([-1], [1], 20)),
            "sine": synthetic_gen("Sine", rnd,
                                  lambda x_1: sin(x_1), 
                                  E([0], [6.2], [0.1])),
            "burks": synthetic_gen("Burks", rnd,
                                   lambda x_1: 4*x_1**4 + 3*x_1**3 + 2*x_1**2 + x_1, 
                                   U([-1], [1], 20)),
            "r1": synthetic_gen("R1", rnd,
                                lambda x_1: (x_1+1)**3/(x_1**2-x_1+1), 
                                E([-1], [1], [2/19]), U([-1], [1], 20)),
            "r2": synthetic_gen("R2", rnd,
                                lambda x_1: (x_1**5-3*x_1**3+1)/(x_1**2+1), 
                                E([-1], [1], [2/19]), U([-1], [1], 20)),
            "r3": synthetic_gen("R3", rnd,
                                lambda x_1: (x_1**6+x_1**5)/(x_1**4+x_1**3+x_1**2+x_1+1), 
                                E([-1], [1], [2/19]), U([-1], [1], 20)),
            "poly-10": synthetic_gen("Poly-10", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10: 
                                         x_1*x_2+x_3*x_4+x_5*x_6+x_1*x_7*x_9+x_3*x_6*x_10,
                                     U([0]*10, [1]*10, 330), U([0]*10, [1]*10, 170)),
            "koza-2": synthetic_gen("Koza-2", rnd,
                                    lambda x_1: x_1**5-2*x_1**3+x_1, 
                                    U([-1], [1], 20), U([-1], [1], 20)),
            "koza-3": synthetic_gen("Koza-3", rnd,
                                    lambda x_1: x_1**6-2*x_1**4+x_1**2,
                                    U([-1], [1], 20), U([-1], [1], 20)),
            "korns-1": synthetic_gen("Korns-1", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 1.57+24.3*x_4,
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "korns-4": synthetic_gen("Korns-4", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         -2.3+0.13*sin(x_3),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "korns-7": synthetic_gen("Korns-7", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         213.80940889*(1-e**(-0.54723748542*x_1)),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "korns-11": synthetic_gen("Korns-11", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         6.87+11*cos(7.23*x_1**3),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "korns-12": synthetic_gen("Korns-12", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         2-2.1*cos(9.8*x_1)*sin(1.3*x_5),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "vladislavleva-1": synthetic_gen("Vladislavleva-1", rnd,
                                             lambda x_1, x_2: 
                                                 e**(-(x_1-1)**2)/(1.2+(x_2-2.5)**2),
                                             U([0.3]*2, [4]*2, 100), E([-0.2]*2, [4.2]*2, [0.1]*2)),
            "vladislavleva-2": synthetic_gen("Vladislavleva-2", rnd,
                                             lambda x_1: 
                                                 e**(-x_1)*x_1**3*(cos(x_1)*sin(x_1))*(cos(x_1)*sin(x_1)**2-1),
                                             E([0.05], [10], [0.1]), E([-0.5], [10.5], [0.05])),
            "vladislavleva-3": synthetic_gen("Vladislavleva-3", rnd,
                                             lambda x_1, x_2: 
                                                 e**(-x_1)*x_1**3*(cos(x_1)*sin(x_1))*(cos(x_1)*sin(x_1)**2-1)*(x_2-5),
                                             E([0.05]*2, [10, 10.05], [0.1, 2]), E([-0.5]*2, [10.5]*2, [0.05, 0.5])),
            "vladislavleva-4": synthetic_gen("Vladislavleva-4", rnd,
                                             lambda x_1, x_2, x_3, x_4, x_5: 
                                                 10/(5+(x_1-3)**2+(x_2-3)**2+(x_3-3)**2+(x_4-3)**2+(x_5-3)**2),
                                             U([0.05]*5, [6.05]*5, 1024), U([-0.25]*5, [6.35]*5, 5000)),
            "vladislavleva-5": synthetic_gen("Vladislavleva-5", rnd,
                                             lambda x_1, x_2, x_3: 
                                                 30*(x_1-1)*(x_3-1)/((x_1-10)*x_2**2),
                                             U([0.05, 1, 0.05], [2]*3, 300), 
                                             E([-0.05, 0.95, -0.05], [2.1, 2.05, 2.1], [0.15, 0.1, 0.15])),
            "vladislavleva-6": synthetic_gen("Vladislavleva-6", rnd,
                                             lambda x_1, x_2: 6*sin(x_1)*cos(x_2),
                                             U([0.1]*2, [5.9]*5, 30), 
                                             E([-0.05]*2, [6.05]*2, [0.02]*2)),
            "vladislavleva-7": synthetic_gen("Vladislavleva-7", rnd,
                                             lambda x_1, x_2: 
                                                 (x_1-3)*(x_2-3)+2*sin((x_1-4)*(x_2-4)),
                                             U([0.05]*2, [6.05]*2, 300), U([-0.25]*2, [6.35]*2, 1000)),
            "vladislavleva-8": synthetic_gen("Vladislavleva-8", rnd,
                                             lambda x_1, x_2: 
                                                 ((x_1-3)**4+(x_2-3)**3-(x_2-3))/((x_2-2)**4+10),
                                             U([0.05]*2, [6.05]*2, 50), E([-0.25]*2, [6.35]*2, [0.2]*2)),
            "pagie-1": synthetic_gen("Pagie-1", rnd,
                                     lambda x_1, x_2: 1/(1+x_1**(-4))+1/(1+x_2**(-4)),
                                     E([-5]*2, [5]*2, [0.4]*2)),
            "keijzer-1": synthetic_gen("Keijzer-1", rnd,
                                       lambda x_1: 
                                           0.3*x_1*sin(2*pi*x_1),
                                       E([-1], [1], [0.1]),
                                       E([-1], [1], [0.001])),
            "keijzer-2": synthetic_gen("Keijzer-2", rnd,
                                       lambda x_1: 
                                           0.3*x_1*sin(2*pi*x_1),
                                       E([-2], [2], [0.1]),
                                       E([-2], [2], [0.001])),
            "keijzer-3": synthetic_gen("Keijzer-3", rnd,
                                       lambda x_1: 
                                           0.3*x_1*sin(2*pi*x_1),
                                       E([-3], [3], [0.1]),
                                       E([-3], [3], [0.001])),
            "keijzer-4": synthetic_gen("Keijzer-4", rnd,
                                       lambda x_1: 
                                           x_1**3*e**(-x_1)*cos(x_1)*sin(x_1)*(sin(x_1)**2*cos(x_1)-1),
                                       E([0], [10], [0.05]),
                                       E([0.05], [10.05], [0.05])),
            "keijzer-5": synthetic_gen("Keijzer-5", rnd,
                                       lambda x_1, x_2, x_3: 30*x_1*x_3/((x_1-10)*x_2**2),
                                       U([-1, 1, -1], [1,2,1], 1000),
                                       U([-1, 1, -1], [1,2,1], 10000)),
            "keijzer-6": synthetic_gen("Keijzer-6", rnd,
                                       lambda x_1: sum([1/i for i in range(1, x_1+1)]),
                                       E([1], [50], [1]),
                                       E([1], [120], [1])),
            "keijzer-7": synthetic_gen("Keijzer-7", rnd,
                                       lambda x_1: log(x_1),
                                       E([1], [100], [1]),
                                       E([1], [100], [0.1])),
            "keijzer-8": synthetic_gen("Keijzer-8", rnd,
                                       lambda x_1: sqrt(x_1),
                                       E([0], [100], [1]),
                                       E([0], [100], [0.1])),
            "keijzer-9": synthetic_gen("Keijzer-9", rnd,
                                       lambda x_1: log(x_1+sqrt(x_1**2+1)),
                                       E([0], [100], [1]),
                                       E([0], [100], [0.1])),
            "keijzer-10": synthetic_gen("Keijzer-10", rnd,
                                       lambda x_1, x_2: x_1**x_2,
                                       U([0]*2, [1]*2, 100),
                                       E([0]*2, [1]*2, [0.01]*2)),
            "keijzer-11": synthetic_gen("Keijzer-11", rnd,
                                       lambda x_1, x_2: x_1*x_2+sin((x_1-1)*(x_2-1)),
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "keijzer-12": synthetic_gen("Keijzer-12", rnd,
                                       lambda x_1, x_2: x_1**4-x_1**3+(x_2**2/2)-x_2,
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "keijzer-13": synthetic_gen("Keijzer-13", rnd,
                                       lambda x_1, x_2: 6*sin(x_1)*cos(x_2),
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "keijzer-14": synthetic_gen("Keijzer-14", rnd,
                                       lambda x_1, x_2: 8/(2+x_1**2+x_2**2),
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "keijzer-15": synthetic_gen("Keijzer-15", rnd,
                                       lambda x_1, x_2: (x_1**3/5)+(x_2**3/2)-x_2-x_1,
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "nguyen-1": synthetic_gen("Nguyen-1", rnd,
                                       lambda x_1: x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "nguyen-2": synthetic_gen("Nguyen-2", rnd,
                                       lambda x_1: x_1**4+x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "nguyen-3": synthetic_gen("Nguyen-3", rnd,
                                       lambda x_1: x_1**5+x_1**4+x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "nguyen-4": synthetic_gen("Nguyen-4", rnd,
                                       lambda x_1: x_1**6+x_1**5+x_1**4+x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "nguyen-5": synthetic_gen("Nguyen-5", rnd,
                                       lambda x_1: sin(x_1**2)*cos(x_1)-1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "nguyen-6": synthetic_gen("Nguyen-6", rnd,
                                       lambda x_1: sin(x_1)+sin(x_1+x_1**2),
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "nguyen-7": synthetic_gen("Nguyen-7", rnd,
                                       lambda x_1: log(x_1+1)+log(x_1**2+1),
                                       U([0], [2], 20),
                                       U([0], [2], 20)),
            "nguyen-8": synthetic_gen("Nguyen-8", rnd,
                                       lambda x_1: sqrt(x_1),
                                       U([0], [4], 20),
                                       U([0], [4], 20)),
            "nguyen-9": synthetic_gen("Nguyen-9", rnd,
                                       lambda x_1, x_2: sin(x_1)+sin(x_2**2),
                                       U([-1]*2, [1]*2, 100),
                                       U([-1]*2, [1]*2, 100)),
            "nguyen-10": synthetic_gen("Nguyen-10", rnd,
                                       lambda x_1, x_2: 2*sin(x_1)*cos(x_2),
                                       U([-1]*2, [1]*2, 100),
                                       U([-1]*2, [1]*2, 100))
            }

import joblib 
joblib.dump(data_synt, "data.joblib", compress=3)







