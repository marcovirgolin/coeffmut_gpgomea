import os
import pandas as pd
import joblib


datasets = os.listdir("results")
complete_results_df = None
complete_logs_df = None
n_jobs=8

def get_dfs_from_result_file(f, d):
  """
  Parameters
  ----------
  f : str
    file name
  d : str
    dataset name
  """
  uid = f.split("_")[2].replace(".csv","")
  seed = f.split("_")[1]

  resp_log_f = "log_"+seed+"_"+uid+".csv"
  
  rdf = pd.read_csv(os.path.join("results",d,f))
  ldf = pd.read_csv(os.path.join("results",d,resp_log_f))
  
  # add all info in log even if redundant, for easy access
  for col in rdf.columns:
    ldf[col] = rdf[col]

  return rdf, ldf

rdfs = list()
ldfs = list()
for d in datasets:
  print("Processing",d)
  files = [f for f in os.listdir(os.path.join("results",d)) if f.startswith("result_")]
  curr_rdfsldfs = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_dfs_from_result_file)(f,d) for f in files)
  # assemble
  rdfs = rdfs + [x[0] for x in curr_rdfsldfs]
  ldfs = ldfs + [x[1] for x in curr_rdfsldfs]

# create pd dataframes
print("Concatenating")
final_rdf = pd.concat(rdfs)
final_ldf = pd.concat(ldfs)
final_rdf.reset_index(inplace=True, drop=True)
final_ldf.reset_index(inplace=True, drop=True)

print("Dumping")
joblib.dump(final_rdf, "final_rdf.joblib", compress=5)
joblib.dump(final_ldf, "final_ldf.joblib", compress=5)