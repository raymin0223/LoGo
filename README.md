
# [Official] Re-thinking Federated Active Learning based on Inter-class Diversity

---
This repository is the official implementation of "Re-thinking Federated Active Learning based on Inter-class Diversity" paper presented in CVPR 2023. [[Paper](https://arxiv.org/abs/2303.12317), [Video](https://www.youtube.com/watch?v=gAoKIAE-a9o)]

---

## Abstract
Although federated learning has made awe-inspiring advances, most studies have assumed that the client's data are fully labeled.
However, in a real-world scenario, every client may have a significant amount of unlabeled instances.
Among the various approaches to utilizing unlabeled data, a federated active learning framework has emerged as a promising solution. 
In the decentralized setting, there are two types of available query selector models, namely global and local-only models, but little literature discusses their performance dominance and its causes.
In this work, we first demonstrate that the superiority of two selector models depends on the global and local inter-class diversity.
Furthermore, we observe that the global and local-only models are the keys to resolving the imbalance of each side.
Based on our findings, we propose LoGo, a FAL sampling strategy robust to varying local heterogeneity levels and global imbalance ratio, that integrates both models by two steps of active selection scheme.
LoGo consistently outperforms six active learning strategies in the total number of 38 experimental settings.



## Installation
Please check below requirements and install packages from `requirements.txt`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Usage
The following commands are examples of running the code.

```bash
# Query Selector: Global, AL Strategy: Entropy, Dataset: CIFAR-10
python main.py --seed 1  \
--al_method entropy \
--model cnn4conv \
--dataset cifar10 \
--partition dir_balance \
--dd_beta 0.1 \
--num_users 10 \
--frac 1.0 \
--num_classes 10 
--rounds 100 \
--local_ep 5 \
--reset random \
--query_model_mode global \
--query_ratio 0.05
```

```bash
# Query Selector: Global, AL Strategy: LoGo, Dataset: CIFAR-10
python main.py --seed 1  \
--al_method logo \
--model cnn4conv \
--dataset cifar10 \
--partition dir_balance \
--dd_beta 0.1 \
--num_users 10 \
--frac 1.0 \
--num_classes 10 
--rounds 100 \
--local_ep 5 \
--reset random \
--query_ratio 0.05
```

### Parameters for learning
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. default = `cnn4conv`. |
| `dataset`      | Dataset to use. Options:  `cifar10`, `svhn`, `pathmnist`, `organmnist`, `dermamnist`. default = `cifar10`. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `momentum` | SGD momentum, default = `0.9`. |
| `weight_decay` | SGD momentum, default = `0.00001`. |


### Parameters for federated learning
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `rounds` | The total number of communication roudns, default = `100`. |
| `local_bs` | Local batch size, default = `64`. |
| `local_ep` | Number of local update epochs, default = `5`. |
| `num_users` | Number of users, Default = `10`. |
| `frac` | The fraction of participating cleints, default = `1.0`. |
| `dd_alpha` | The concentration parameter alpha for Dirichlet distribution, default = `0.1`. |

### Parameters for active learning
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `query_ratio` | The ratio of data examples per one query = `0.5`. |
| `query_model_mode` | The query selector model. Options: `globa`, `local_only`. default = `global`. |
| `al_method` | The active learning strategy. Options: `random`, `entropy`, `coreset`, `badge`, `gcnal`, `alfa_mix`, `logo`.|



## Experimental Result
![exp_bar](https://user-images.githubusercontent.com/12638561/202088590-48b421ec-11a2-4319-a106-3cda808d255f.png)
![exp_results](https://user-images.githubusercontent.com/12638561/202088388-716f6693-de59-4d77-a3b5-0b809379091c.png)

## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{kim2023re,
  title={Re-thinking Federated Active Learning based on Inter-class Diversity},
  author={Kim, SangMook and Bae, Sangmin and Song, Hwanjun and Yun, Se-Young},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3944--3953},
  year={2023}
}
```

## Contact
Feel free to contact us if you have any questions:)

- SangMook Kim: sangmook.kim@kaist.ac.kr
- Sangmin Bae: bsmn0223@kaist.ac.kr
