# Eluding Secure Aggregation in Federated Learning via Model Inconsistency
https://arxiv.org/abs/2111.07380

The folder 'EludingSecureAggregation' contains the code to replicate our results as well as the data collected and reported in the paper.

**Requirements**:

* python3 / jupyter
  * TensorFlow2
  * tensorflow_datasets 
  * numpy
  * matplotlib
  * tqdm

The notebook 'GradientSuppression_POC.ipynb' offers an end-to-end example of gradient suppression + gradient inversion.

The notebook 'CanaryGradientInteractive_POC.ipynb' allows to test the canary gradient attack in an interactive fashion. A non-interactive version of the code is available in 'canary_attack_main.py'. This can be used by providing a configuration file as input. For instance:

```
canary_attack_main.py settings.c10_c100 0
```

The main hyper-parameters for 'canary_attack_main.py' are in 'settings/\_\_init\_\_.py'

The result will be saved in the 'results' folder and it can be read using the notebook 'plot_data.ipynb'. The script 'run_all.sh' can be used to run all the tests.

In order to run the test with the tinyImagenet dataset, it is necessary to manually download it: http://cs231n.stanford.edu/tiny-imagenet-200.zip and pre-process it.
