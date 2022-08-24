import os
import pandas as pd
import joblib



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
non_gom_coeff_mut_possibilities = [True, False]
inittype_possibilities = [False, "heuristic"]
ims_possibilities = [False, '6_1']


dataset_names = os.listdir("results")
print(dataset_names)


params = list()
for run_i in range(10):
  for dataset in dataset_names:
    for cm in coeff_mut_possibilities:
      for ngcm in non_gom_coeff_mut_possibilities:
        for init_type in inittype_possibilities:
          for ims in ims_possibilities:
            param = {"i":run_i, "dataset":dataset, "coeffmut":cm, "nongomcoeffmut":ngcm, "ims":ims, "inittype":init_type, "quick_test":False}
            params.append(param)

print("considering {} runs".format(len(params)))


def get_runfile_name(i, dataset, coeffmut=False, nongomcoeffmut=False, ims=False, inittype=False, linearscaling=False, standardize=False, quick_test=False ):

  uid = str(i)+"_"+str(coeffmut).replace("False","0").replace("_","").replace(".","")
  uid += str(int(nongomcoeffmut))+str(ims).replace("False","0").replace("_","")+str(inittype).replace("False","0").replace("heuristic","h")
  uid += str(int(linearscaling))+str(int(standardize))

  return "result_{}.csv".format(uid)

# create list of compatible names
print("Creating list of compatible names")
runfile_names = list()
for p in params:
  runfile_names.append(get_runfile_name(**p))
logfile_names = [x.replace("result_","log_") for x in runfile_names]
allfile_names = runfile_names+logfile_names


# scan logs and remove stuff
print("Scanning and removing")
from os.path import join
for d in dataset_names:
  print("  processing",d)
  files = os.listdir(join("results",d))
  for f in files:
    if f not in allfile_names:
      print(f)
  


