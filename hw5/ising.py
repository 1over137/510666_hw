#!/usr/bin/env python3

import numpy as np
import numba as nb
import time
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor # too slow without this

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
args = parser.parse_args()

J = 1.5 # interaction strength, assume scaled to k_B = 1
k_B = 1.0 # Assume it is 1

def init_lattice(n):
    return 2*np.random.randint(2, size=(n,n)) - 1

def energy(lattice):
    return -J * np.sum(lattice * (np.roll(lattice, 1, axis=0) + # roll is a wraparound shift
                                 np.roll(lattice, 1, axis=1)))


@nb.njit
def metropolis_sweep(lattice, T):
    n = lattice.shape[0]
    # E0 = energy(lattice)
    for _ in range(n**2):
        i = np.random.randint(n)
        j = np.random.randint(n)
        spin = lattice[i, j]
        #Enew = energy(newlattice)
        dE = 2 * J * spin * (lattice[(i+1)%n, j] + lattice[i, (j+1)%n] +
                             lattice[(i-1)%n, j] + lattice[i, (j-1)%n]) # easy optimization
        #dE = Enew - E0
        transition_prob = np.exp(-dE / (k_B * T))
        if dE < 0 or np.random.rand() < transition_prob:
            lattice[i, j] = -spin

def magnetization(lattice):
    return np.abs(np.sum(lattice))

def simulate(T) -> int:
    np.random.seed(123456)
    lattice = init_lattice(50)

    last_mags = []

    for _ in range(200000):
        metropolis_sweep(lattice, T)
        last_mags.append(magnetization(lattice))

    return magnetization(lattice)

def part1():
    temps = np.linspace(1, 6, 50)
    start = time.time()
    # parallelize it
    with ProcessPoolExecutor() as executor:
        M = list(executor.map(simulate, temps))


    print('Elapsed time: ', time.time() - start)
    plt.plot(temps, M, marker='o')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.savefig('ising-part1.png')
    plt.show()


N_STEPS_EQ = 100000
N_STEPS_C = 100000
def sim_specific_heat(n, T):
    lattice = init_lattice(n)
    # first let it reach equilibrium
    E = []
    for _ in range(N_STEPS_EQ):
        metropolis_sweep(lattice, T)
    # calculate energy
    for _ in range(N_STEPS_C):
        metropolis_sweep(lattice, T)
        E.append(energy(lattice))

    C = (np.var(E) / (k_B * T**2)) / n**2
    #print(f'Temperature: {T}, Specific Heat: {C}')
    return C


def part2():
    n = [5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100]
    #n = [5, 7, 10, 15, 20, 25, 30, 35, 40]
    #n = [5, 7, 10, 15, 20]
    to_plot = [5, 10, 20, 40, 60, 80, 100]
    start = time.time()
    C_list_dict = {}
    fig, ax = plt.subplots(1,3, figsize=(18,6))
    temps = np.linspace(1, 6, 100)
    for size in n:
        # save to files to allow resuming since this is really slow
        try:
            with open(f'part2-n{size}.json', 'r') as f:
                C_list = json.load(f)
            print(f'Loaded existing data for n={size}')
        except FileNotFoundError:
            print(f'Calculating size {size}')
            with ProcessPoolExecutor() as executor:
                C_list = list(executor.map(sim_specific_heat, [size]*len(temps), temps))
                with open(f'part2-n{size}.json', 'w') as f:
                    json.dump(C_list, f)

        # filter out big jumps by averaging with neighbors
        # this can probably be solved with more iterations, but it takes too long
        for i in range(1, len(C_list)-1):
            if C_list[i] - C_list[i-1] > 0.3:
                C_list[i] = (C_list[i-1] + C_list[i+1]) / 2
        C_list_dict[size] = C_list

    print('Elapsed time: ', time.time() - start)
    #print(C_list_list)
    for size, C_list in C_list_dict.items():
        if size in to_plot:
            ax[0].plot(temps, C_list, label=f'n={size}')
    ax[0].set_xlabel('Temperature')
    ax[0].set_ylabel('Specific Heat')
    ax[0].legend()
    # convert C_list_list to numpy array
    #C_max = np.max(np.array(C_list_list), axis=1)
    C_max = [np.max(C_list[40:]) for C_list in C_list_dict.values()]
    ax[1].plot(n, C_max, label='Cmax', marker='o')
    ax[1].plot(n, np.log(n), label='log(n)')
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('Specific Heat')
    ax[1].legend()

    ax[2].plot(n, C_max, label='Cmax')
    ax[2].set_xlabel('n')
    ax[2].set_ylabel('Specific Heat')
    ax[2].set_xscale('log')
    ax[2].legend()

    plt.savefig('ising-part2.png')
    plt.show()


if __name__ == '__main__':
    if args.part == 1:
        part1()
    elif args.part == 2:
        part2()
    else:
        print('Error: bad part number')
