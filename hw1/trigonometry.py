#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

parser = argparse.ArgumentParser(prog='trigonometry.py', 
                                 description='Plot functions')

parser.add_argument('--function', type=str,
                    help='function: cos, sin, sinc or any combination thereof',
                    dest='function')
parser.add_argument('--write', type=str, dest='write')
parser.add_argument('--read_from_file', type=str, dest='read')
parser.add_argument('--print', type=str, dest='print')

options = parser.parse_args()

x = np.linspace(-10, 10, int(20/0.05)) # get points every 0.05 between -10 and 10
y_cols = []
if options.function:
    for function in options.function.split(","):
        if function == "sin":
            y = np.sin(x)
        elif function == "cos":
            y = np.cos(x)
        elif function == "sinc":
            y = np.sinc(x)
        else:
            raise NotImplementedError("Unsupported function: " + function)
        y_cols.append(y)
        plt.plot(x, y, label=function + "(x)")


if options.read:
    with open(options.read, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        x_list = []
        y_list = []
        for row in reader:
            x_list.append(float(row[0]))
            y_list.append([float(i) for i in row[1:]]) # y_list is a list of lists

        y_cols = list(zip(*y_list)) # transpose y_list from list of rows to list of columns
        for idx, col in enumerate(y_cols):
            plt.plot(x_list, col, label=f'File: {options.read}, column {idx+2}') # because the first col is x

plt.title("Trigonometry plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

if options.write:
    with open(options.write[0], 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for x, *y_values in zip(x, *y_cols): # unpack ycols, zip, and take one for each row
            writer.writerow([x, *y_values])


if options.print:
    if options.print == "jpeg":
        options.print = "jpg"
    plt.savefig(f"output.{options.print}")

plt.show()
