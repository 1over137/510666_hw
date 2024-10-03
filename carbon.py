import numpy as np
import matplotlib.pyplot as plt
import argparse

tau = 5700 / np.log(2)
N0_kg = 1e-12 # kg

parser = argparse.ArgumentParser(description="Carbon-14 decay activity")
parser.add_argument("--width", type=float, metavar="w", dest="time_step", help="Step size for numerical differentiation", default=45)
args = parser.parse_args()
STEP_SIZE = args.time_step

N0 = N0_kg * 1000 / 14 * 6.022e23 # atoms

def N(t):
    return N0 * np.exp(-t/tau)

def R(t):
    return N0/tau * np.exp(-t/tau) # derivative of N(t)



for STEP_SIZE in [10, 100, 1000]:
    prev_n = N0
    t_values = []
    r_values = []
    for t in np.arange(0, 20_000, STEP_SIZE)[1:]:
        r = ( prev_n - N(t) ) / STEP_SIZE
        prev_n = N(t)
        t_values.append(t)
        r_values.append(r)
    plt.plot(t_values, r_values, label=f"Step size = {STEP_SIZE} years")
t_values = np.arange(0, 20_000, 10)
plt.plot(t_values, R(t_values), label="Exact solution")
plt.xlabel("Time (years)")
plt.ylabel("Activity (decays per year)")
plt.legend()
plt.show()