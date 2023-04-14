## Self-Expressive Network

This repository presents the final project of EECS6322 for  [Learning a Self-Expressive Networl for Subspace Clustering]()

### Usage

Run the scripts by using following command

``` python
# dataset: MNIST, EMNIST, FashionMNIST, CIFAR10
python SENet_main.py --dataset=MNIST
```

### Parameter setting
Here is some parameters that can be changed to get experiment results.
``` python
parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--batch_eval', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--gamma', type=float, default=200.0)
    parser.add_argument('--hid_dims', type=int, default=[1024, 1024, 1024])
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--mean_subtract', dest='mean_subtraction', action='store_true')
    parser.set_defaults(mean_subtraction=False)
    parser.add_argument('--num_subspaces', type=int, default=10)
    parser.add_argument('--out_dims', type=int, default=1024)
    parser.add_argument('--spectral_dim', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_iters', type=int, default=100000)
    parser.add_argument('--top_k', type=int, default=1000)
```

### Dataset
The processed experimental datasets can be download [here](https://drive.google.com/file/d/19U9TDzoQjppWSDf9zQXQhmubQVqZFJmY/view?usp=sharing). Put the datasets folder in the same work path to execute the code.
