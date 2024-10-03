import argparse
import numpy as np
import matplotlib.pyplot as plt


# constants

INITIAL_VELOCITY = 70 # m/s
DEBUG = False
MASS = 0.046 # kg
TIME_STEP = 0.1

parser = argparse.ArgumentParser(description="Golf ball trajectory")
parser.add_argument("--plot", type=float, metavar="theta", dest="theta", help="Angle (degrees) of golf ball launch", default=45)
args = parser.parse_args()
angle = args.theta

# ideal

data = {}


def drag_accel_smooth(v): # acceleration due to drag
    return - (1/2) * 1.29 * 0.0014 * np.pow(v, 2) / MASS

def drag_accel_dimpled(v): # acceleration due to drag
    if (v <= 14):
        return - (1/2) * 1.29 * 0.0014 * np.pow(v, 2) / MASS
    else:
        return - (7.0 / v) * 1.29 * 0.0014 * np.pow(v, 2) / MASS
    
def magnus_accel(v): # acceleration due to magnus force
    return 0.25 * v # perpendicular to v

rad = np.deg2rad(angle)

# ideal case
vx = INITIAL_VELOCITY * np.cos(rad)
vy = INITIAL_VELOCITY * np.sin(rad)

t_values = np.arange(0, 100, TIME_STEP)
x_values = []
y_values = []
x = 0
y = 0
for t in t_values:
    v = np.sqrt(vx**2 + vy**2)
    vy += - 9.81 * TIME_STEP # apply acceleration from gravity
    x += vx * TIME_STEP
    y += vy * TIME_STEP
    x_values.append(x)
    y_values.append(y)
    if (DEBUG):
        print(f"t = {t}, x = {x}, y = {y}, vx = {vx}, vy = {vy}")


x_values = np.array(x_values)
y_values = np.array(y_values)

# filter out negative y values
x_values = x_values[y_values >= 0]
y_values = y_values[y_values >= 0]
plt.plot(x_values, y_values, label="Ideal case")

# drag, smooth ball
vx = INITIAL_VELOCITY * np.cos(rad)
vy = INITIAL_VELOCITY * np.sin(rad)

t_values = np.arange(0, 100, TIME_STEP)
x_values = []
y_values = []
x = 0
y = 0
# accelerations
for t in t_values:
    v = np.sqrt(vx**2 + vy**2)
    drag_a = drag_accel_smooth(v)
    vx += drag_a * (vx / v) * TIME_STEP
    vy += drag_a * (vy / v) * TIME_STEP - 9.81 * TIME_STEP
    x += vx * TIME_STEP
    y += vy * TIME_STEP
    x_values.append(x)
    y_values.append(y)
    if (DEBUG):
        print(f"t = {t}, x = {x}, y = {y}, vx = {vx}, vy = {vy}")


x_values = np.array(x_values)
y_values = np.array(y_values)

# filter out negative y values
t_values = t_values[y_values >= 0]
x_values = x_values[y_values >= 0]
y_values = y_values[y_values >= 0]
plt.plot(x_values, y_values, label=f"Smooth ball, drag")

# drag, dimpled ball
vx = INITIAL_VELOCITY * np.cos(rad)
vy = INITIAL_VELOCITY * np.sin(rad)

t_values = np.arange(0, 100, TIME_STEP)
x_values = []
y_values = []
x = 0
y = 0
# accelerations
for t in t_values:
    v = np.sqrt(vx**2 + vy**2)
    drag_a = drag_accel_dimpled(v)
    vx += drag_a * (vx / v) * TIME_STEP
    vy += drag_a * (vy / v) * TIME_STEP - 9.81 * TIME_STEP
    x += vx * TIME_STEP
    y += vy * TIME_STEP
    x_values.append(x)
    y_values.append(y)

    if (DEBUG):
        print(f"t = {t}, x = {x}, y = {y}, vx = {vx}, vy = {vy}")
    
x_values = np.array(x_values)
y_values = np.array(y_values)

# filter out negative y values
t_values = t_values[y_values >= 0]
x_values = x_values[y_values >= 0]
y_values = y_values[y_values >= 0]
plt.plot(x_values, y_values, label=f"Dimpled ball, drag")


vx = INITIAL_VELOCITY * np.cos(rad)
vy = INITIAL_VELOCITY * np.sin(rad)

t_values = np.arange(0, 100, TIME_STEP)
x_values = []
y_values = []
x = 0
y = 0
# accelerations
for t in t_values:
    v = np.sqrt(vx**2 + vy**2)
    drag_a = drag_accel_dimpled(v)
    magnus_a = magnus_accel(v)
    vx += drag_a * (vx / v) * TIME_STEP - magnus_a * (vy / v) * TIME_STEP
    vy += drag_a * (vy / v) * TIME_STEP - 9.81 * TIME_STEP + magnus_a * (vx / v) * TIME_STEP
    x += vx * TIME_STEP
    y += vy * TIME_STEP
    x_values.append(x)
    y_values.append(y)

    if (DEBUG):
        print(f"t = {t}, x = {x}, y = {y}, vx = {vx}, vy = {vy}")
    
x_values = np.array(x_values)
y_values = np.array(y_values)

# filter out negative y values
t_values = t_values[y_values >= 0]
x_values = x_values[y_values >= 0]
y_values = y_values[y_values >= 0]
plt.plot(x_values, y_values, label=f"Dimpled ball, drag, magnus force")
plt.legend()
plt.title(f"Golf ball trajectory: Angle = {angle}\u00b0")


plt.show()
