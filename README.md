# flsim_dqn

### CSCE689 Deep Reinforcement Learning Course Project (Fall 2022)

![alt text](https://github.com/tian1327/flsim_dqn/blob/master/flsim_dqn_logo.png "dqn logo")

[[Slides](https://github.com/tian1327/flsim_dqn/blob/master/Project_Overview.pdf)]
[[Project Report](https://github.com/tian1327/flsim_dqn/blob/master/CSCE689_DRL_Project_Report.pdf)]
[[5-min YouTube Presentation](https://youtu.be/ZNkAfigHkN0)]

In this project, we implemented the `D-DQN model` for device selection during Federated Learning process and reproduced some results in the following paper.

[Hao Wang](https://www.haow.ca), Zakhary Kaplan, [Di Niu](https://sites.ualberta.ca/~dniu/Homepage/Home.html), [Baochun Li](http://iqua.ece.toronto.edu/bli/index.html). "Optimizing Federated Learning on Non-IID Data with Reinforcement Learning," in the Proceedings of IEEE INFOCOM, Beijing, China, April 27-30, 2020.



#### To install env:

```shell
conda env create -f environment_torch_cpu.yml
```

The Double DQN server for learning how to select devices are implemented in `/server/dqn.py`.

#### Evaluation done in report:
1. Reproduce Fig 1 (done by Tian)
   * `python run.py --config=configs/MNIST/mnist_fedavg_iid.json`
   * `python run.py --config=configs/MNIST/mnist_fedavg_noniid.json`
   * `python run.py --config=configs/MNIST/mnist_kcenter_noniid.json`
   * `python run.py --config=configs/MNIST/mnist_kmeans_noniid.json`
   * until model achieves 99% test accuracy
   * `python plots/plot_fig_1.py`
  
2. Reproduce Figure 3, PCA on clients weights (by Tian and YuTing)
   * 100 clients with 2 PCA components
   * 20 clients with 2 PCA components
   * `python plots/plot_fig_3.py`
  
3. Reproduce Fig 5(a), D-DQN trained on MNIST dataset (by Tian and YuTing)
   * select 10 out of 100, each client has 600 data
     * `python run.py --config=dqn_noniid_10_100.json`
   * select 4 out of 20, each client has 3000 data
     * `python run.py --config=dqn_noniid_4_20.json`
   * Plot Total_reward vs. Training Episodes (YuTing)
     
     
4. Compare using the target reward function vs. new proposed difference function (Tian, NiuCheng)
   * select 10 out of 100, each client has 600 data, using the new reward function
     * `python run.py --config=dqn_noniid_10_100_difference.json`
     * `python run.py --config=dqn_noniid_4_20_difference.json`
   * Plot Total_reward vs. Training Episodes (YuTing)    
     * check `plots/` folder
  
5. Reproduce Fig 6(c) on MNIST datasets with non-IID degree of 0.8 (Tian)
   * For each of two settings, compare DQN_infer vs. FedAvg (Random selection) vs. K-Center vs. K-means
   * Plot Testing Accuracy vs. Communication Rounds (YuTing)
   * check `plots/` folder   


Reference: [FL-Lottery](https://github.com/iQua/fl-lottery/tree/360d9c2d54c12e2631ac123a4dd5ac9184d913f0)


***

## About

Welcome to **FLSim**, a PyTorch based federated learning simulation framework, created for experimental research in a paper accepted by [IEEE INFOCOM 2020](https://infocom2020.ieee-infocom.org):

[Hao Wang](https://www.haow.ca), Zakhary Kaplan, [Di Niu](https://sites.ualberta.ca/~dniu/Homepage/Home.html), [Baochun Li](http://iqua.ece.toronto.edu/bli/index.html). "Optimizing Federated Learning on Non-IID Data with Reinforcement Learning," in the Proceedings of IEEE INFOCOM, Beijing, China, April 27-30, 2020.



## Installation

To install **FLSim**, all that needs to be done is clone this repository to the desired directory.

### Dependencies

**FLSim** uses [Anaconda](https://www.anaconda.com/distribution/) to manage Python and it's dependencies, listed in [`environment.yml`](environment.yml). To install the `fl-py37` Python environment, set up Anaconda (or Miniconda), then download the environment dependencies with:

```shell
conda env create -f environment.yml
```

## Usage

Before using the repository, make sure to activate the `fl-py37` environment with:

```shell
conda activate fl-py37
```

### Simulation

To start a simulation, run [`run.py`](run.py) from the repository's root directory:

```shell
python run.py
  --config=config.json
  --log=INFO
```

##### `run.py` flags

* `--config` (`-c`): path to the configuration file to be used.
* `--log` (`-l`): level of logging info to be written to console, defaults to `INFO`.

##### `config.json` files

**FLSim** uses a JSON file to manage the configuration parameters for a federated learning simulation. Provided in the repository is a generic template and three preconfigured simulation files for the CIFAR-10, FashionMNIST, and MNIST datasets.

For a detailed list of configuration options, see the [wiki page](https://github.com/iQua/flsim/wiki/Configuration).

If you have any questions, please feel free to contact Hao Wang (haowang@ece.utoronto.ca)
