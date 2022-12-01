# FLSim

## For CSCE689-DRL Project

```shell
conda env create -f environment.yml
```

##### To do:

Evaluation:
1. Reproduce Fig 1
   * `python run.py --config=configs/MNIST/mnist_fedavg_iid.json`
   * `python run.py --config=configs/MNIST/mnist_fedavg_noniid.json`
   * `python run.py --config=configs/MNIST/mnist_kcenter_noniid.json`
   * `python run.py --config=configs/MNIST/mnist_kmeans_noniid.json`
   * until model achieves 99% test accuracy
   * `python plot_fig_1.py`
  
2. Fig 5 on 3 datasets (DQN vs. DDQN vs. Actor-critic?)
   
3. Fig 6 on non-IID MNIST datasets with different levels (DQN vs. DDQN vs. Actor-critic?)
   
4. Table 1

Reference: [FL-Lottery](https://github.com/iQua/fl-lottery/tree/360d9c2d54c12e2631ac123a4dd5ac9184d913f0)

* state PCA -+
* select 1
* select 10 4
* reward using accuracy +
* Done -
* Inference server using saved train server 


2. Implement DDQN with sampling 1 device during training, check performance
3. sample top k during training
4. Check sampling 1 + random 9 devices
5. Check consecutivly sampling 10 times (10 rounds), then aggregrate once
6. Try small action space, like selecting 2 devices out of 10 devices



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
