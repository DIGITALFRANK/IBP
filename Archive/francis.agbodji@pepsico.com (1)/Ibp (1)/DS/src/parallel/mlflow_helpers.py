# Databricks notebook source
def grab_latest_mlflow_runid(experiment_id, hyperparam_level):
  " Returns the run_id for the latest mlflow run within a specified segmentation"
  df = mlflow.search_runs(experiment_ids=experiment_id)
  df = df[df["params.HYPERPARAM_LEVEL"]==str(hyperparam_level)]
  df = df.sort_values(by=['end_time'], ascending = False)
  df = df.head(1)
  return(df[["run_id"]]["run_id"][0])

def downlad_mlflow_artifact(download_dir, run_id, artifact_name):
  " Returs an artifact from a particular run_id "
  if not os.path.exists(download_dir):
      os.mkdir(download_dir)
  local_path = client.download_artifacts(run_id, artifact_name, download_dir)
  
  artifact_to_return = None
  #Return yml
  if "yml" in artifact_name:
    with open(local_path) as file:
      # The FullLoader parameter handles the conversion from YAML
      # scalar values to Python the dictionary format
      artifact_to_return = yaml.load(file, Loader=yaml.FullLoader)

  return(artifact_to_return)