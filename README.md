# Eluding Secure Aggregation in Federated Learning via Model Inconsistency
Accepted at CCS'22 -> https://arxiv.org/abs/2111.07380

The folder 'EludingSecureAggregation' contains the code to replicate our results.

**Requirements**:

* python3 / jupyter
  * TensorFlow2
  * tensorflow_datasets 
  * numpy
  * matplotlib
  * tqdm

The notebook 'GradientSuppression_POC.ipynb' offers an end-to-end example of gradient suppression + gradient inversion.

**⚠️ Note that ⚠️: in the code we use an unsophisticated gradient inversion attack for simplicity. However, as discussed in the paper, that can be replaced with more recent attacks and/or malicious gradient inversion attacks (e.g., trap weights) by providing the right payload to the target user.**

The notebook 'CanaryGradientInteractive_POC.ipynb' allows to test the canary gradient attack in an interactive fashion. A non-interactive version of the code is available in 'canary_attack_main.py'. This can be used by providing a configuration file as input. For instance:

```
canary_attack_main.py settings.c10_c100 0
```

The main hyper-parameters for 'canary_attack_main.py' are in 'settings/\_\_init\_\_.py'

The result will be saved in the 'results' folder and it can be read using the notebook 'plot_data.ipynb'. The script 'run_all.sh' can be used to run all the tests.

In order to run the test with the tinyImagenet dataset, it is necessary to manually download it: http://cs231n.stanford.edu/tiny-imagenet-200.zip and pre-process it.

How to cite the paper:
```
@inproceedings{10.1145/3548606.3560557,
author = {Pasquini, Dario and Francati, Danilo and Ateniese, Giuseppe},
title = {Eluding Secure Aggregation in Federated Learning via Model Inconsistency},
year = {2022},
isbn = {9781450394505},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3548606.3560557},
doi = {10.1145/3548606.3560557},
booktitle = {Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security},
pages = {2429–2443},
numpages = {15},
keywords = {federated learning, secure aggregation, model inconsistency},
location = {Los Angeles, CA, USA},
series = {CCS '22}
}
```
