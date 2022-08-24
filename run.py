from pyGPGOMEA import GPGOMEARegressor as GPG
from sklearn.datasets import load_boston
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sympy import simplify, preorder_traversal
import time, os
import pandas as pd
from io import StringIO
import sys, joblib

OVERWRITE=False
n_jobs = int(sys.argv[1])

is_rw = False
if len(sys.argv) > 3:
  is_rw = True

if is_rw:
  data = joblib.load("luizvbo_gpbenchmarks/data_rw.joblib")
else:
  data = joblib.load("luizvbo_gpbenchmarks/data.joblib")
dataset_names = list(data.keys())

less_than_two_constants_datasets = [
  "keijzer-4","keijzer-6","keijzer-7","keijzer-8","keijzer-9","keijzer-10","keijzer-12","keijzer-13",
  "koza-2","koza-3",
  "meier-3","meier-4",
  "nguyen-1","nguyen-2","nguyen-3","nguyen-4","nguyen-5","nguyen-6","nguyen-8","nguyen-9","nguyen-10",
  "nonic","poly-10",
  "r3",
  "sine",
  "vladislavleva-2",
  "vladislavleva-6"
]

if not is_rw:
  for dataset in less_than_two_constants_datasets:
    assert(dataset in dataset_names)
  dataset_names = [x for x in dataset_names if x not in less_than_two_constants_datasets]
print("tot num datasets", len(dataset_names))

def sympy_model_size(sympy_model):
  c=0
  for _ in preorder_traversal(sympy_model):
    c += 1
  return c

def load_dataset(dataset_name):
  ds = data[dataset_name]
  Xy_train = ds["training"].to_numpy()
  X_train = Xy_train[:,:-1]
  y_train = Xy_train[:,-1]
  Xy_test = ds["test"].to_numpy()
  X_test = Xy_test[:,:-1]
  y_test = Xy_test[:,-1]
  return X_train, X_test, y_train, y_test
  

def run(i, dataset, coeffmut=False, gomcoeffmutstrat="within", ims=False, linearscaling=True, standardize=False, quick_test=False ):

  uid = str(i)+"_"+str(coeffmut).replace("False","0").replace("_","").replace(".","")
  uid += str(gomcoeffmutstrat)+str(ims).replace("False","0").replace("_","")
  uid += str(int(linearscaling))+str(int(standardize))

  if not OVERWRITE and os.path.exists("results/{}/result_{}.csv".format(dataset,uid)):
    print("Skipping pre-existing result")
    return
  print("Running", dataset, uid)

  X_train, X_test, y_train, y_test = load_dataset(dataset)

  if standardize:
    ss_X = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(y_train)
    y_test = ss_y.transform(y_test)

  start_time = time.time()
  gpg = GPG(
    gomea=True,
    functions="+_-_*_p/_sqrt_exp_plog_sin_cos",
    time=-1, generations=-1, evaluations=int(1e4) if quick_test else int(1e6),
    initmaxtreeheight=4,
    ims=ims,
    popsize=64 if ims != False else 1024,
    batchsize=min(len(X_train),256),
    coeffmut=coeffmut,
    gomcoeffmutstrat=gomcoeffmutstrat,
    linearscaling=linearscaling,
    parallel=False, 
    silent=True,
    seed=i
  )

  gpg.fit(X_train,y_train)
  end_time = time.time() - start_time


  if standardize:
    train_rmse = np.sqrt(np.mean(np.square(ss_y.var_*(y_train - gpg.predict(X_train)))))
    test_rmse = np.sqrt(np.mean(np.square(ss_y.var_*(y_test - gpg.predict(X_test)))))
  else:
    train_rmse = np.sqrt(np.mean(np.square(y_train - gpg.predict(X_train))))
    test_rmse = np.sqrt(np.mean(np.square(y_test - gpg.predict(X_test))))
  train_r2 = 1 - np.square(train_rmse) / np.var(y_train)
  test_r2 = 1 - np.square(test_rmse) / np.var(y_train)

  model = gpg.get_model().replace("p/","/").replace("plog","log")
  simpl_model = simplify(model)
  size = gpg.get_n_nodes()
  simpl_size = sympy_model_size(simpl_model)

  result = {
    "dataset" : dataset,
    "seed" : i,
    "train_rmse" : train_rmse,
    "test_rmse" : test_rmse,
    "train_r2" : train_r2,
    "test_r2" : test_r2,
    "model" : model,
    "simpl_model" : simpl_model,
    "size" : size,
    "simpl_size" : simpl_size,
    "runtime" : end_time,
    "coeff_mut" : coeffmut,
    "gom_coeff_mut_strat" : gomcoeffmutstrat,
    "ims" : ims,
    "linear_scaling" : linearscaling,
    "data_standardization" : standardize,
    "quick_test" : quick_test,
  }
  for k in result:
    result[k] = [result[k]]

  if not os.path.exists("results/{}".format(dataset)):
    os.makedirs("results/{}".format(dataset), exist_ok=True)

  result_df = pd.DataFrame.from_dict(result)
  result_df.to_csv("results/{}/result_{}.csv".format(dataset,uid), index=False)

  log = gpg.get_progress_log().replace("\t",",")
  log_df = pd.read_csv(StringIO(log))
  log_df.to_csv("results/{}/log_{}.csv".format(dataset,uid), index=False)




def build_coeff_mut_possibilities():

  strategy = [False, "es", "naive"]
  prob_of_applying = [0.5, 1.0]
  naive_temp = [0.1, 0.9]
  naive_decay = [0.1, 0.9]
  naive_noimpgen_decay = [-1, 5]

  combos = list()
  for s in strategy:
    if s == False:
      combos.append(s)
    elif s == "es":
      for p in prob_of_applying:
        combo = str(p)+"_es"
        combos.append(combo)
    else:
      for p in prob_of_applying:
        for t in naive_temp:
          for d in naive_decay:
            for g in naive_noimpgen_decay:
              combo = str(p)+"_"+str(t)+"_"+str(d)+"_"+str(g)
              combos.append(combo)

  return combos


coeff_mut_possibilities = build_coeff_mut_possibilities()
gom_coeff_mut_strat = ["within","interleaved","afteronce","afterfossize",False]
#inittype_possibilities = [False, "heuristic"]
ims_possibilities = [False]#, '6_1']



##dataset_names = [d for d in dataset_names if len(os.listdir("results/"+d)) < 2736]
#print(dataset_names)


params = list()
#for run_i in range(10):
run_i = int(sys.argv[2])
for dataset in dataset_names:
  for cm in coeff_mut_possibilities:
    for gcms in gom_coeff_mut_strat:
      for ims in ims_possibilities:

        # remove invalid param combos
        # 1) if cm probability is 0, then gcms must be False (& vice versa)
        if cm == False and gcms != False or gcms == False and cm != False:
          continue
        # 2) if cm noimp_gen_decay is -1, then it does not matter what decay is used (skip one of the two)
        if type(cm) == str and cm.endswith("-1") and cm.split("_")[2] != "0.1":
          continue

        param = {"i":run_i, "dataset":dataset, "coeffmut":cm, "gomcoeffmutstrat":gcms, "ims":ims, "quick_test":False}
        params.append(param)

print("considering {} runs".format(len(params)))

from joblib import Parallel, delayed

if n_jobs > 1:
  Parallel(n_jobs=n_jobs)(delayed(run)(**p) for p in params)
else:
  for p in params:
    run(**p)

