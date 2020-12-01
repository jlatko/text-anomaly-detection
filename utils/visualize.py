import json
import os
import numpy as np
from datetime import datetime
import torch
import shutil
import pickle
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def get_runs_and_cfgs():
  runs  = []
  cfgs = []
  for run in glob.glob('data/models/*/*/*/*/*'): 
    runs.append(run)
    with open(f"{run}/tags.txt", 'r') as fp:
      cfgs.append(fp.readlines())

  return runs, cfgs

def print_cfg(cfg):
  print(cfg[0])
  print(cfg[1])
  KL_kwargs =  json.loads(cfg[2])
  model_kwargs = json.loads(cfg[3])
  print("Kl_kwargs:")
  for key in KL_kwargs:
    print("  ", key, ": ", KL_kwargs[key])
  print("model_kwargs:")
  for key in model_kwargs:
    print("  ", key, ": ", model_kwargs[key])

def print_cfgs(cfgs):
  for idx, cfg in enumerate(cfgs):
    print("\n######\n")
    print("CONFIG ", idx, "\n\n")
    print_cfg(cfg)

def print_runs(runs):
  for idx, run in enumerate(runs):
    print("\n######\n")
    print("RUN ", idx, "\n\n")
    print(run)

def plot_run(run):
  with open(f"{run}/val_losses_desc.json", 'r') as fp:
    val_loss = json.load(fp)
  with open(f"{run}/train_losses_desc.json", 'r') as fp:
    train_loss = json.load(fp)

  val_df =  pd.DataFrame(data=val_loss).transpose()#.drop('name', axis=1)
  train_df =  pd.DataFrame(data=train_loss).transpose()#.drop('name', axis=1)

  anomaly = []
  with (open("data/models/RNN/runs/12-01/09:50/1/anomaly_scores.pickle", "rb")) as openfile:
    while True:
      try:
        anomaly.append(pickle.load(openfile))
      except EOFError:
        break

  anomaly_recon = anomaly[0][list(anomaly[0].keys())[0]]['recon']
  anomaly_kl = anomaly[0][list(anomaly[0].keys())[0]]['kl']
  anomaly_df = pd.DataFrame(list(zip(anomaly_recon, anomaly_kl)), 
               columns =['recon', 'kl'])
  
  plot_loss(val_df)
  plot_loss(train_df)
  plot_anomaly(anomaly_df)

def plot_loss(df):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df['epoch'], y=df['recon_loss'],
                      mode='lines+markers',
                      name='recon_loss'))
  fig.add_trace(go.Scatter(x=df['epoch'], y=df['kl_loss'],
                      mode='lines+markers',
                      name='kl_loss'))
  fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'],
                      mode='lines+markers', 
                      name='loss'))
  
  fig.update_layout(
       title={
        'text': df['name'][0],
        'y':0.9,
        'x':0.45},
         width=800, height=300,
        #  xaxis_title="Epoch",
         margin=dict(l=10, r=10, t=60, b=0))
  fig.show()

def plot_anomaly(df):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=list(range(0,len(df['recon']))), y=df['recon'],
                      mode='lines',
                      name='recon_loss'))
  fig.add_trace(go.Scatter(x=list(range(0,len(df['kl']))) , y=df['kl'],
                      mode='lines',
                      name='kl_loss'))
  
  fig.update_layout(
       title={
        'text': 'anomaly',
        'y':0.9,
        'x':0.45},
         width=800, height=300,
         margin=dict(l=10, r=10, t=60, b=0))
  fig.show()