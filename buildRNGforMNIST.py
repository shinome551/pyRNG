#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse

import numpy as np
import torch
from torchvision.datasets import MNIST

from cy import calcDistanceTriuMatrix, buildRelativeNeighborhoodGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=100)
    args = parser.parse_args()
    trainset = MNIST(root='./data', train=True, download=True)

    train_data = trainset.data.flatten(1) / 255.0
    train_targets = np.array(trainset.targets)

    n = args.num

    start = time.time()

    ## only cpu
    dist_mat_triu = np.memmap('dist_mat_triu.dat', mode='w+', dtype=np.float32, shape=(n * (n - 1) // 2, ))
    calcDistanceTriuMatrix(train_data[:n].numpy(), dist_mat_triu)
    del dist_mat_triu
    ## use GPU and write file and read as np.memmap
    #dist_p = torch.pdist(train_data[:n].to('cuda:0'))
    #s = torch.FloatStorage.from_file('dist_mat_triu.dat', shared=True, size=len(dist_p))
    #t = torch.FloatTensor(s).copy_(dist_p)
    #del s

    end = time.time()
    print(f'caluclated dist mat triu / time: {end - start:.3f}s')

    start = time.time()
    dist_mat_triu = np.memmap('dist_mat_triu.dat', mode='r+', dtype=np.float32)
    edges = buildRelativeNeighborhoodGraph(dist_mat_triu, n)
    end = time.time()
    print(f'built RNG / time: {end - start:.3f}s')

    edges = np.array(edges)
    edges_condensed = edges[train_targets[edges[:,0]] != train_targets[edges[:,1]]]
    nodes_condensed = np.unique(edges_condensed)
    print(f'number of condensed nodes: {len(nodes_condensed)}')

    np.savetxt('item/edges.txt', edges, fmt='%u')
    np.savetxt('item/edges_condensed.txt', edges_condensed, fmt='%u')
    np.savetxt('item/nodes_condensed.txt', nodes_condensed, fmt='%u')

    os.remove('dist_mat_triu.dat')


if __name__ == '__main__':
    main()