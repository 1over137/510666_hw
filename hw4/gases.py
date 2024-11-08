#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import colors
cmap = colors.ListedColormap(['white', 'blue', 'red'])
parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
args = parser.parse_args()
def init_grid():
    grid = np.zeros((40,60))
    grid[:,:20] = 1 # blue
    grid[:,-20:] = 2 # red
    return grid

def diffuse(grid):
    # pick random position
    x = np.random.randint(0,60)
    y = np.random.randint(0,40)
    newx = x
    newy = y
    # pick random direction
    direction = np.random.randint(0,4) # 0: right, 1: up, 2: left, 3: down

    if direction == 0 and x < 59:
        newx = x+1
    elif direction == 1 and y < 39:
        newy = y+1
    elif direction == 2 and x > 0:
        newx = x-1
    elif direction == 3 and y > 0:
        newy = y-1

    if grid[y,x] != 0  and grid[newy,newx] == 0:
        # swap
        tmp = grid[y,x]
        grid[y,x] = grid[newy,newx]
        grid[newy,newx] = tmp

N_STEPS = 500000
def part2():
    grid = init_grid()
    datapoints = []
    for _ in range(N_STEPS+1):
        diffuse(grid)
        if _ % (N_STEPS//5) == 0: # this will actually run 6 times
            datapoints.append(grid.copy())


    fig, ax = plt.subplots(4,3, figsize=(20,20))
    for i in range(6):
        ax[i//3*2,i%3].set_title(f'Step {i*(N_STEPS//5)}')
        ax[i//3*2,i%3].imshow(datapoints[i], cmap=cmap)
        ax[i//3*2,i%3].axis('off')

        # linear density
        density_a = np.count_nonzero(datapoints[i]==1, axis=0)
        density_b = np.count_nonzero(datapoints[i]==2, axis=0)


        ax[i//3*2+1,i%3].plot(density_a)
        ax[i//3*2+1,i%3].plot(density_b)
    plt.savefig("gases-part2.png")
    plt.show()

def part3():
    N_RUNS = 100
    number_a = np.zeros((N_RUNS,6,60))
    number_b = np.zeros((N_RUNS,6,60))
    # same thing but average over 100 runs
    for i in range(N_RUNS):
        grid = init_grid()
        datapoints = []
        for n in range(N_STEPS+1):
            diffuse(grid)
            if n % (N_STEPS//5) == 0:
                idx = n//(N_STEPS//5)
                datapoints.append(grid.copy())
                number_a[i,idx] = np.count_nonzero(grid == 1, axis=0)
                number_b[i,idx] = np.count_nonzero(grid == 2, axis=0)

    fig,ax = plt.subplots(2,3, figsize=(15,10))

    for i in range(6):

        # linear density
        avg_number_a = np.mean(number_a[:,i], axis=0)
        avg_number_b = np.mean(number_b[:,i], axis=0)
        ax[i//3,i%3].plot(avg_number_a, label='$n_A(x)$')
        ax[i//3,i%3].plot(avg_number_b, label='$n_B(x)$')
        ax[i//3,i%3].set_title(f'Average across {N_RUNS} runs: Step {i*(N_STEPS//5)}')
        ax[i//3,i%3].legend()

    plt.savefig("gases-part3.png")
    plt.show()

if __name__ == '__main__':
    if args.part == 2:
        part2()
    elif args.part == 3:
        part3()
    else:
        print('Error: bad part number')
