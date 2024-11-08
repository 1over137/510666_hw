#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
args = parser.parse_args()
D=2
xlimit=100
sigma_init=1

dt=0.2
dx=1
def part2():
    rho = np.zeros(xlimit*2)
    rho[xlimit-2:xlimit+2] = 1/4 # box size 4
    fig,ax = plt.subplots(2,3)
    for t in range(600):
        rho_new = np.copy(rho)

        for i in range(1,xlimit*2-1):
            # finite difference
            # from here: https://sites.me.ucsb.edu/~moehlis/APC591/tutorials/tutorial5/node3.html
            rho_new[i] = rho[i] + D*(rho[i+1] + rho[i-1] - 2*rho[i])*dt/dx**2

        rho = rho_new

        if t % 100 == 0:
            idx = t//100
            ax[idx//3,idx%3].set_title(f't={t*dt}')
            ax[idx//3,idx%3].plot(rho)

            if idx == 0:
                continue # skip the first one
            # fit a gaussian in here
            sigma = np.sqrt(2*D*t*dt)
            x = np.arange(xlimit*2)
            y = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-xlimit)**2/(2*sigma**2))
            ax[idx//3,idx%3].plot(x,y)


    plt.savefig('diffusion-part2.png')
    plt.show()

if __name__ == '__main__':
    if args.part == 2:
        part2()
    else:
        print('Error: bad part number')
