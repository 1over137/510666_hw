#!/usr/bin/env python3
import numpy as np
import random
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
args = parser.parse_args()
def unif(n_samples):
    return np.random.rand(n_samples)

def box_muller(n_samples):
    MU = 0
    SIGMA = 1.0
    u1 = 1-np.random.rand(n_samples) # guarantee not to be 0 since 1.0 is excluded
    u2 = np.random.rand(n_samples)

    mag = SIGMA * np.sqrt(-2.0 * np.log(u1))
    z0 = mag*np.cos(2*np.pi*u2) + MU
    # forget about z1
    return z0

def gaussian_pdf(x):
    return 1 / np.sqrt(2*np.pi) * np.exp(-x**2/2)

def draw_bins(n_samples,ax,start_msidx, randfunc=np.random.rand, draw_gaussian=False):
    # start_msidx: starting most significant index
    n = randfunc(n_samples)

    for i,n_bins in enumerate([10,20,50,100]):
        ax[start_msidx][i].hist(n, n_bins, density=True)
        if draw_gaussian:
             x = np.linspace(-4,4,200)
             ax[start_msidx][i].plot(x, gaussian_pdf(x))
        ax[start_msidx][i].set_title(f"{n_samples} samples, {n_bins} bins")

def part1():
    fig,ax = plt.subplots(2,4, figsize=(16,8))
    draw_bins(1000,ax,0)
    draw_bins(1000000,ax,1)
    plt.savefig("rnumbers-part1.png")
    plt.show()
def part2():
    fig,ax = plt.subplots(2,4, figsize=(16,8))
    draw_bins(1000,ax,0, box_muller, draw_gaussian=True)
    draw_bins(1000000,ax,1, box_muller, draw_gaussian=True)
    plt.savefig("rnumbers-part2.png")
    plt.show()


if __name__ == '__main__':
    if args.part == 1:
        part1()
    elif args.part == 2:
        part2()
    else:
        print('Error: bad part number')
