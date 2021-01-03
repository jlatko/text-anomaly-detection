# Anomaly detection in text using variational autoencoders
Project done by Jan Latko, Artur Przybysz and Jonatan Cichawa 
for a Deep Learning DTU course under the supervision from Corti.

## Running
To run an RNN-VAE experiment use `experiment.py` file with appropriate configuration (experiments use [Sacred](https://sacred.readthedocs.io/en/stable/quickstart.html)). 
To evaluate a saved model against different dataset use `load_model.py`.  

To run a word frequency baseline download those [english word frequencies](https://www.kaggle.com/rtatman/english-word-frequency) and run `baseline.py`.

`lang_model.py` runs a RNN-LM experiment.

`analyze_results.ipynb` was used to load logs and metrics from experiments and analyze them.

Needed data and embeddings should download automatically.

---
Parts of the code were based on https://github.com/wiseodd/controlled-text-generation.