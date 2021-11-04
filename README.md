<!-- #region -->
# Graph-less Neural Network (GLNN)

Code for [Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation](https://arxiv.org/pdf/2110.08727.pdf)

## Overview
### Distillation framework
<p align="center">
  <br />
  <img src="imgs/glnn.png" width="800">
  <br />
</p>


### Accuracy vs. inference time on the `ogbg-product` dataset

<p align="center">
  <br />
  <img src="imgs/trade_off.png" width="800">
  <br />
</p>


## Getting Started

### Requirements
- Python version >= 3.6
- PyTorch version >= 1.7.1
- DGL

Optional
- OGB version >= 1.2.6 for the OGB datasets
- Additional requirements in requirements.txt



### Dataset
CPF_data: download the '.npz' files from https://github.com/BUPT-GAMMA/CPF/tree/master/data/npz and put them under `CPF_data/data/`

OGB_data: download the OGB dataset and put them under `OGB_data/data/`
(More details: https://ogb.stanford.edu/)


### Usage

See `CPF_data/scripts.sh` and `OGB_data/scripts.sh`

<!-- 
## Results

There are some results on GCN teacher model, with different datasets and student varients. More results can be seen in our paper.

| Datasets    | GCN (Teacher) | CPF-ind (Student) | CPF-tra (Student) | improvement |
| ----------- | ------------- | ----------------- | ----------------- | ----------- |
| Cora        | 0.8244        | **0.8576**        | 0.8567            | 4.0%        |
| Citeseer    | 0.7110        | 0.7619            | **0.7652**        | 7.6%        |
| Pubmed      | 0.7804        | 0.8080            | **0.8104**        | 3.8%        |
| A-Computers | 0.8318        | **0.8443**        | **0.8443**        | 1.5%        |
| A-Photo     | 0.9072        | **0.9317**        | 0.9248            | 2.7%        |
 -->
<!-- ## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{yang2021extract,
  title={Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework},
  author={Cheng Yang and Jiawei Liu and Chuan Shi},
  booktitle={Proceedings of The Web Conference 2021 (WWW ’21)},
  publisher={ACM},
  year={2021}
}
``` -->

## Contact Us

Please open an issue or contact shichang@cs.ucla.edu if you have any questions.
<!-- #endregion -->
