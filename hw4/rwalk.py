#!/usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
args = parser.parse_args()
def part1():
    walks = np.ndarray((10000,100,2))
    for walkidx in range(10000):
        x = 0
        y = 0
        for n in range(100):
            walks[walkidx,n,0] = x
            walks[walkidx,n,1] = y
            direction = random.randrange(4)
            # number the directions from x axis going CCW
            if direction == 0: # right
                x+=1
            if direction == 1: # up
                y+=1
            if direction == 2: # left
                x-=1
            if direction == 3: # down
                y-=1
    # plot the average of each step across all walks
    fig,ax = plt.subplots(2)
    x = np.mean(walks[:,:,0], axis=0)
    x_squared = np.mean(walks[:,:,0]**2, axis=0)

    ax[0].plot(x)
    ax[1].plot(x_squared)
    ax[0].set_title("$<x>$")
    ax[1].set_title("$<x^2>$")
    ax[0].set_xlabel("Step")
    ax[1].set_xlabel("Step")
    plt.savefig("rwalk-part1.png")
    plt.show()



if __name__ == '__main__':
    if args.part == 1:
        part1()
    else:
        print('Error: bad part number')
