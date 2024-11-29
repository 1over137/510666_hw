import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
args = parser.parse_args()

R = 10 # radius
a = 0.6 # dist

def jacobi_relax(X,Y,V, rho, tol):
    V_new = V.copy()
    for i in tqdm(range(1000000)): #max iterations
        # use poisson's equation. New value is the average of the neighbors minus our laplacian (-rho)
        V_new[1:-1, 1:-1] = (
            V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2] + rho[1:-1, 1:-1]
        ) / 4

        # boundary cond
        r = np.sqrt(X**2 + Y**2)
        V_new[r >= R] = 0

        
        rel_diff = np.abs((V_new - V)).max()
        if rel_diff < tol:
            V = V_new.copy()
            break
        V = V_new.copy()
    
    return V_new, i # iter count

def sor(X,Y,V, rho, tol, alpha):
    V_new = V.copy()
    N = V.shape[0]
    for niter in tqdm(range(1000000)): #max iterations
        V_old = V_new.copy()
        # use poisson's equation. New value is the average of the neighbors plus our laplacian (rho)
        # we update in place so use V_new's values when possible
        for i in range(1, N-1):
            for j in range(1, N-1):
                V_new[i, j] = (
                    (1 - alpha) * V_old[i, j] +
                    alpha * (V_new[i+1, j] + V_new[i-1, j] + V_new[i, j+1] + V_new[i, j-1] + rho[i, j])/4 
                )


        # boundary cond
        r = np.sqrt(X**2 + Y**2)
        V_new[r >= R] = 0

        rel_diff = np.abs((V_new - V)).max()
        if rel_diff < tol:
            V = V_new.copy()
            break
        V = V_new.copy()
    
    return V_new, niter # iter count


def part1():
    N = 200 # points on each axis

    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    dx = x[1] - x[0]
    X,Y = np.meshgrid(x,y)

    # initial conditions
    V = np.zeros((N, N))
    rho = np.zeros((N, N)) # charge denstiy

    # dipole
    rho[N//2 + int(a/2/dx), N//2] = 1 / dx**2
    rho[N//2 - int(a/2/dx), N//2] = -1 / dx**2

    Vx, i = jacobi_relax(X,Y,V, rho, 1e-6)

    plt.figure(figsize=(10, 10))
    plt.contour(X, Y, Vx, 20)
    #plt.colorbar(label="Potential (V)")
    plt.xlabel("x")
    plt.xlim(-3,3)
    plt.ylim(-3,3) #focus on the import part
    plt.ylabel("y")
    plt.title(f"Equipotential lines for dipole")
    plt.grid(True)
    plt.savefig("poisson-part1.png")
    plt.show()

def part2():
    N = 200 # points on each axis

    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    dx = x[1] - x[0]
    X,Y = np.meshgrid(x,y)

    # initial conditions
    V = np.zeros((N, N))
    rho = np.zeros((N, N)) # charge denstiy

    # dipole
    rho[N//2 + int(a/2/dx), N//2] = 1 / dx**2
    rho[N//2 - int(a/2/dx), N//2] = -1 / dx**2

    niter = {}
    for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        Vx, i = jacobi_relax(X,Y,V, rho, tol)
        print(f"Converged after {i} iterations with tolerance {tol}")
        niter[tol] = i

    plt.figure(figsize=(6, 6))
    plt.plot(list(niter.keys()), list(niter.values()))
    plt.xlabel("Tolerance")
    plt.ylabel("Iterations")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.title("Iterations vs Tolerance")
    plt.savefig("poisson-part2.png")
    plt.show()
    
def part3():
    niter_sor = {}
    niter_jacobi = {}
    # we hold the tolerance steady, vary N
    for N in [50, 100, 150, 200, 250, 300]: 
        x = np.linspace(-R, R, N)
        y = np.linspace(-R, R, N)
        dx = x[1] - x[0]
        X,Y = np.meshgrid(x,y)

        # initial conditions
        V = np.zeros((N, N))
        rho = np.zeros((N, N)) # charge denstiy

        # dipole
        rho[N//2 + int(a/2/dx), N//2] = 1 / dx**2
        rho[N//2 - int(a/2/dx), N//2] = -1 / dx**2

        Vres, i = sor(X,Y,V, rho, 1e-5, 1.7) 
        Vres2, i2 = jacobi_relax(X,Y,V, rho, 1e-5) 
        niter_sor[N] = i
        niter_jacobi[N] = i2

        print(f"SOR: Converged after {i} iterations with N={N}")
        print(f"Jacobi: Converged after {i2} iterations with N={N}")
    
    plt.figure(figsize=(6, 6))
    plt.plot(list(niter_sor.keys()), list(niter_sor.values()), label="SOR, $\\alpha$=1.7")
    plt.plot(list(niter_jacobi.keys()), list(niter_jacobi.values()), label="Jacobi")
    plt.xlabel("N")
    plt.ylabel("Iterations")
    plt.title("Iterations vs Grid size (N)")
    plt.legend()
    plt.savefig("poisson-part3.png")
    plt.show()
   
    

if __name__ == '__main__':
    if args.part == 1:
        part1()
    elif args.part == 2:
        part2()
    elif args.part == 3:
        part3()
    else:
        print('Error: bad part number')


