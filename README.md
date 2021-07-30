# An Evaluation of Disentangled Representation Learning for Texts

This repository contains code for the experiments listed in the paper "An Evaluation of Disentangled Representation Learning for Texts", accepted to the Findings of ACL 2021. 

Some components borrow heavily from the following repositories:
- https://github.com/mingdachen/disentangle-semantics-syntax
- https://github.com/shentianxiao/text-autoencoders

## Components

1. Config Files:
- The `config/` folder contains the config files used to train our models. The parameters can be modified as desired. 

2. Data:
- The `data/` folder contains datasets used for experiments on PersonageNLG and the Bible dataset. The GYAFC corpus is not distributable by us (see [https://github.com/raosudha89/GYAFC-corpus] for how to obtain access.)

3. Models:
- The `models` folder contains code for the model variations enumerated in the paper. Run `trainer.py` with the appropriate arguments to train and save models. 
- The conditional decoder baseline in `baseline.py`.

4. Evalutation:
- the `evaluation` folder contains python functions to evaluate along the three dimensions of Retrieval, Classification and Style Transfer respectively. These are not directly executable, but provide utility functions that can be used once models have been trained and saved.

5. Scripts:
- The `scripts` folder provides sample bash scripts that can be used to run the training module on a server. 

## Contact

Please email `vkpriya@cs.toronto.edu` with any questions. 

## Citation
TBD

