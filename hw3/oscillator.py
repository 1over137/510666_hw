import numpy as np 
import matplotlib.pyplot as plt
import argparse

# parse --part=2 3 4
parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int)
args = parser.parse_args()




g = 9.8
l = 9.8
gamma = 0.25
T_N_POINTS = 2000

def dwdt_part2(theta, omega, t, Omega_D, alpha_D):
    return -(g/l) * theta - 2*gamma*omega + alpha_D * np.sin(Omega_D * t)

def euler_cromer(Omega_D, dwdt, t, theta0, alpha_D):
    dt = t[1] - t[0]
    theta = np.zeros_like(t)
    omega = np.zeros_like(t) # theta'
    for i in range(1, len(t)): # skip zeroth
        omega[i] = omega[i-1] + dt * dwdt(theta[i-1], omega[i-1], t[i-1], Omega_D, alpha_D)
        theta[i] = theta[i-1] + dt * omega[i]

    return theta, omega

def rk4(Omega_D, dwdt, t, theta0, alpha_D):
    dt = t[1] - t[0]
    theta = np.zeros_like(t)
    omega = np.zeros_like(t) # theta'
    theta[0] = theta0
    for i in range(1, len(t)):
        k1_theta = omega[i-1]
        k1_omega = dwdt(theta[i-1], omega[i-1], t[i-1], Omega_D, alpha_D)

        # use k1 to estimate midpoint slope
        k2_theta = omega[i-1] + dt/2 * k1_omega
        k2_omega = dwdt(theta[i-1] + dt/2 * k1_theta, omega[i-1] + dt/2 * k1_omega, t[i-1] + dt/2, Omega_D, alpha_D)

        # use k2 to estimate midpoint slope
        k3_theta = omega[i-1] + dt/2 * k2_omega
        k3_omega = dwdt(theta[i-1] + dt/2 * k2_theta, omega[i-1] + dt/2 * k2_omega, t[i-1] + dt/2, Omega_D, alpha_D)

        # use k3 to estimate end slope
        k4_theta = omega[i-1] + dt * k3_omega
        k4_omega = dwdt(theta[i-1] + dt * k3_theta, omega[i-1] + dt * k3_omega, t[i-1] + dt, Omega_D, alpha_D)

        # update

        theta[i] = theta[i-1] + dt/6 * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        omega[i] = omega[i-1] + dt/6 * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    
    return theta, omega


def part2():
    OMEGA_N_POINTS = 500
    t = np.linspace(0, 50, T_N_POINTS) 
    amplitude = np.zeros(OMEGA_N_POINTS)
    phase = np.zeros(OMEGA_N_POINTS)
    last_phase = 0
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    for i, Omega_D in enumerate(omegas:=np.linspace(0, 2, OMEGA_N_POINTS)):
        theta, omega = rk4(Omega_D, dwdt_part2, t, theta0=0, alpha_D=0.2)
        amplitude[i] = np.max(theta[T_N_POINTS//3:]) # skip the first 1/3 points, only consider steady state

        # find the phase w.r.t. the driving force, we know driving force is sin with phase 0
        last_peak = np.argmax(theta[T_N_POINTS//2:]) + T_N_POINTS//2 # use only second half for this
        phase[i] = (t[last_peak] * Omega_D) % (2*np.pi) - np.pi/2 # phase difference
        if phase[i] < 0:
            phase[i] += 2*np.pi

        last_phase = phase[i]
        #print(f'$\Omega_D$ = {Omega_D}, amplitude = {amplitude[i]}, phase = {phase[i]}')

    omega_d_max = omegas[np.argmax(amplitude)]

    print(f'Max: $\Omega_D$ = {omega_d_max}, amplitude = {np.max(amplitude)}')
    ax[0][0].plot(omegas[int(0.1*OMEGA_N_POINTS):], amplitude[int(0.1*OMEGA_N_POINTS):]) # ignore artifacts at very low omega_D
    ax[0][0].annotate(f'Max: $\Omega_D$ = {omega_d_max:.3f}, amplitude = {np.max(amplitude):.3f} ', xy=(omega_d_max, np.max(amplitude)))
    ax[0][1].plot(omegas[int(0.1*OMEGA_N_POINTS):], phase[int(0.1*OMEGA_N_POINTS):]) # ignore artifacts at very low omega_D

    ax[0][0].set_title('RK4: Amplitude')
    ax[0][1].set_title('RK4: Phase')
    ax[0][0].set_xlabel(r'$\Omega_D$ (rad/s)')
    ax[0][1].set_xlabel(r'$\Omega_D$ (rad/s)')
    ax[0][0].set_ylabel('Amplitude (radians)')
    ax[0][1].set_ylabel('Phase (radians)')

    # use euler cromer
    for i, Omega_D in enumerate(omegas:=np.linspace(0, 2, OMEGA_N_POINTS)):
        theta, omega = euler_cromer(Omega_D, dwdt_part2, t, theta0=0, alpha_D=0.2)
        amplitude[i] = np.max(theta[T_N_POINTS//3:]) # skip the first 1/3 points, only consider steady state

        # find the phase w.r.t. the driving force, we know driving force is sin with phase 0
        last_peak = np.argmax(theta[T_N_POINTS//2:]) + T_N_POINTS//2 # use only second half for this
        phase[i] = (t[last_peak] * Omega_D) % (2*np.pi) - np.pi/2 # phase difference
        if phase[i] < 0:
            phase[i] += 2*np.pi

        last_phase = phase[i]
        #print(f'$\Omega_D$ = {Omega_D}, amplitude = {amplitude[i]}, phase = {phase[i]}')

    omega_d_max = omegas[np.argmax(amplitude)]

    print(f'Max: $\Omega_D$ = {omega_d_max}, amplitude = {np.max(amplitude)}')
    ax[1][0].plot(omegas[int(0.1*OMEGA_N_POINTS):], amplitude[int(0.1*OMEGA_N_POINTS):]) # ignore artifacts at very low omega_D
    ax[1][0].annotate(f'Max: $\Omega_D$ = {omega_d_max:.3f}, amplitude = {np.max(amplitude):.3f} ', xy=(omega_d_max, np.max(amplitude)))
    ax[1][1].plot(omegas[int(0.1*OMEGA_N_POINTS):], phase[int(0.1*OMEGA_N_POINTS):]) # ignore artifacts at very low omega_D

    ax[1][0].set_title('Euler-Cromer: Amplitude')
    ax[1][1].set_title('Euler-Cromer: Phase')
    ax[1][0].set_xlabel(r'$\Omega_D$ (rad/s)')
    ax[1][1].set_xlabel(r'$\Omega_D$ (rad/s)')
    ax[1][0].set_ylabel('Amplitude (radians)')
    ax[1][1].set_ylabel('Phase (radians)')

    plt.savefig("oscillator-part2.png")
    plt.show()
    # calculate FWHM
    half_max = np.max(amplitude) / 2
    left = np.where(amplitude > half_max)[0][0]
    right = np.where(amplitude > half_max)[0][-1]
    fwhm = omegas[right] - omegas[left]

    print('left', omegas[left], 'right', omegas[right])
    print(f'FWHM of resonance curve: {fwhm} rad/s')



def part3():
    omega_d = 0.95
    alpha_D = 0.2
    theta0 = 0.1
    t = np.linspace(0, 60, T_N_POINTS)
    # plot potential, kinetic, total
    theta, omega = rk4(omega_d, dwdt_part2, t, theta0, alpha_D)
    print(theta, omega)

    potential = g * l * (1 - np.cos(theta)) # gl (1-cos theta)
    kinetic = 0.5 * (l*omega)**2 # (1/2 v^2)
    print(theta)

    plt.plot(t, potential, label='Potential')
    plt.plot(t, kinetic, label='Kinetic')
    plt.plot(t, theta, label="Theta")
    #plt.plot(t, potential + kinetic, label='Total')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy per mass (J/kg)')
    plt.legend()
    plt.savefig("oscillator-part3.png")
    plt.show()

def dwdt_part4(theta, omega, t, Omega_D, alpha_D):
    # non linear
    return -(g/l) * np.sin(theta) - 2*gamma*omega + alpha_D * np.sin(Omega_D * t)

def part4():
    Omega_D = 0.95
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    t = np.linspace(0, 20, T_N_POINTS//2)
    thetas = []
    theta0 = 0

    theta, omega = rk4(Omega_D, dwdt_part4, t, theta0, 0.2)
    ax[0][0].plot(t, theta, label="$\\theta$")
    ax[0][0].plot(t, omega, label="$\\omega$")
    ax[0][0].legend()
    theta, _ = rk4(Omega_D, dwdt_part4, t, theta0, 1.2)
    ax[0][1].plot(t, theta, label="$\\theta$")
    ax[0][1].plot(t, omega, label="$\\omega$")
    ax[0][1].legend()


    theta, _ = rk4(Omega_D, dwdt_part2, t, theta0, 0.2)
    ax[1][0].plot(t, theta, label="$\\theta$")
    ax[1][0].plot(t, omega, label="$\\omega$")
    ax[1][0].legend()
    theta, _ = rk4(Omega_D, dwdt_part2, t, theta0, 1.2)
    ax[1][1].plot(t, theta, label="$\\theta$")
    ax[1][1].plot(t, omega, label="$\\omega$")
    ax[1][1].legend()
    

    ax[0][0].set_title("Non-linear, $\\alpha_D$ = 0.2 rad/s")
    ax[0][1].set_title("Non-linear, $\\alpha_D$ = 1.2 rad/s")
    ax[1][0].set_title("Linear, $\\alpha_D$ = 0.2 rad/s")
    ax[1][1].set_title("Linear, $\\alpha_D$ = 1.2 rad/s")
    plt.savefig("oscillator-part4.png")
    plt.show()

def part5():
    Omega_D = 0.666
    #fig, ax = plt.subplots(1,3)
    theta0 = 0.5
    d_theta0 = 0.001
    t = np.linspace(0, 100, T_N_POINTS*2)
    for idx, alpha_D in enumerate([0.2,0.5,1.2]):
        theta_a, _ = rk4(Omega_D, dwdt_part4, t, theta0, alpha_D)
        theta_b, _ = rk4(Omega_D, dwdt_part4, t, theta0 + d_theta0, alpha_D)

        d_theta = np.abs(theta_b - theta_a)
        d_theta[d_theta < 1e-9] = 1e-9
        lyapunov_exp = (np.log(d_theta[1:] / d_theta0) / t[1:])[-1] # start from t=1 to not divide by 0
        plt.plot(t, d_theta, label=f"$\\alpha_D$ = {alpha_D} rad/s")
        print(f'Lyapunov exponent for alpha_D = {alpha_D}: {lyapunov_exp}')
    

    plt.legend()    
    plt.savefig("oscillator-part5.png")
    
    plt.show()

if __name__ == '__main__':
    if args.part == 2:
        part2()
    elif args.part == 3:
        part3()
    elif args.part == 4:
        part4()
    elif args.part == 5:
        part5()
    else:
        print('Error: bad part number')

