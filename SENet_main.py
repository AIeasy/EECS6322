"""
Author: Daoming Wan
Date: 2023-04-10
"""
import argparse
from SENet_utils import same_seeds
from SENet_utils import SENet
from SENet_utils import regularization
from SENet_utils import evaluate
import pickle
import torch
import numpy as np
from tqdm import tqdm
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    # Set special arguments for each dataset
    if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
        args.mean_subtract = False
        args.lr_min = 0.0
    elif args.dataset == 'EMNIST':
        args.gamma = 150.0
        args.spectral_dim = 26
        args.mean_subtract = True
        args.batch_eval = 10611
        args.num_subspaces = 26
        args.lmbd = 1.0
    elif args.dataset == 'CIFAR10':
        args.spectral_dim = 10
        args.mean_subtract = False
        # args.total_iters = 500
    else:
        raise Exception("CIFAR10, MNIST, FashionMNIST and EMNIST can be used here.")

    same_seeds(args.seed)

    # Load data
    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
        with open(f'datasets/{args.dataset}/{args.dataset}_scattering_train_data.pkl', 'rb') as f:
            train_samples = pickle.load(f)
        with open(f'datasets/{args.dataset}/{args.dataset}_scattering_train_label.pkl', 'rb') as f:
            train_labels = pickle.load(f)
        with open(f'datasets/{args.dataset}/{args.dataset}_scattering_test_data.pkl', 'rb') as f:
            test_samples = pickle.load(f)
        with open(f'datasets/{args.dataset}/{args.dataset}_scattering_test_label.pkl', 'rb') as f:
            test_labels = pickle.load(f)
        data_samples = np.concatenate([train_samples, test_samples], axis=0)
        data_labels = np.concatenate([train_labels, test_labels], axis=0)
    elif args.dataset in ["CIFAR10"]:
        data_samples = np.load('datasets/CIFAR10-MCR2/cifar10-features.npy')
        data_labels = np.load('datasets/CIFAR10-MCR2/cifar10-labels.npy')
    else:
        raise Exception("Only MNIST, FashionMNIST and EMNIST are currently supported.")

    if args.mean_subtract:
        print("Mean Subtraction")
        data_samples = data_samples - np.mean(data_samples, axis=0, keepdims=True)      # mean subtraction

    data_labels = data_labels - np.min(data_labels)                                     # keep label from 1 to num_space - 1

    global_it = 0


    # Create folderfolder
    folder = f"{args.dataset}_result"
    if not os.path.exists(folder):
        os.mkdir(folder)

    f = open(f"{folder}/{args.dataset}_result.csv", "w")
    writer = csv.writer(f)
    writer.writerow(["sample_size", "ACC", "NMI", "ARI"])


    # Sample data
    for sample_size in [200, 500, 1000, 2000, 5000, 10000, 20000]:
        sample_idx = np.random.choice(data_samples.shape[0], sample_size, replace=False)
        samples = data_samples[sample_idx]
        labels = data_labels[sample_idx]
        ambient_dim = samples.shape[1]

        # L2 Normalize data
        im = torch.from_numpy(samples).float()
        im = im / (torch.norm(im, p=2, dim=1, keepdim=True))

        iters_per_epoch = int(np.ceil(samples.shape[0] / args.batch_size))
        steps_per_iter = int(np.ceil(samples.shape[0] / sample_size))
        epochs = int(np.ceil(args.total_iters / iters_per_epoch))

        # Initialize model
        model = SENet(ambient_dim, args.out_dims, args.hid_dims)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr_min)
        iters = 0

        bar = tqdm(range(epochs), ncols=120)
        for epoch in bar:
            bar.set_description(f"Epoch {epoch}")
            # Shuffle data
            perm = torch.randperm(im.shape[0])
            for i in range(iters_per_epoch):
                model.train()

                # Get batch
                batch_idx = perm[i * args.batch_size: (i + 1) * args.batch_size]
                batch = im[batch_idx]
                batch = batch.cuda()
                # Algorithm 2
                reconstruction = torch.zeros_like(batch).cuda()
                reg = torch.zeros([1]).cuda()
                query_out = model.query_net(batch)
                key_out = model.key_net(batch)
                coeff_matrix = model.coeff_output(query_out, key_out)
                reconstruction = reconstruction + coeff_matrix.mm(batch)
                reg = reg + regularization(coeff_matrix, args.lmbd)

                diag_coeff = model.threshold((query_out * key_out).sum(dim=1, keepdim=True)) * model.alpha
                reconstruction = reconstruction - diag_coeff * batch
                reg = reg - regularization(diag_coeff, args.lmbd)

                L_rec = torch.sum(torch.pow(batch - reconstruction, 2))
                loss = (0.5 * args.gamma * L_rec + reg) / args.batch_size

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
                optimizer.step()

                iters += 1
                global_it += 1

            scheduler.step()
            bar.set_postfix(loss= f"{loss.item():3.2f}",
                            L_rec=f"{L_rec.item() / args.batch_size :3.2f}",
                            reg=f"{reg.item() / args.batch_size:3.2f}")

        full_data = torch.from_numpy(data_samples).float()
        full_data = full_data / (torch.norm(full_data, p=2, dim=1, keepdim=True))
        acc, nmi, ari = evaluate(model, data=full_data, labels=data_labels, num_subspaces=args.num_subspaces,
                                 spectral_dim=args.spectral_dim, top_k=args.top_k,batch_size=args.batch_eval,)
        print(f"sample_size-{sample_size:d}: ACC-{acc:.6f}, NMI-{nmi:.6f}, ARI-{ari:.6f}")
        result_list = [sample_size, acc, nmi, ari]
        writer.writerow(result_list)
        f.flush()

        torch.cuda.empty_cache()

    f.close()
