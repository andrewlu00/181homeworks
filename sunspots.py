#####################
# CS 181, Spring 2019
# Homework 1, Problem 3
#
##################
import math
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts[:13])

X = []
for d in range(13):
    X.append([None]*26)

# TODO: basis functions
def basis0(x,i):
    return 1
def basis1(x,i):
    return x**i
def basis2(x,i):
    return math.exp(-(x-i)*(x-i)/25.0)
def basis3(x,i):
    return math.cos(x/i)

for j in range(13):
    X[j][0] = 1
    for i in range(25):
        X[j][i+1] = basis3(sunspot_counts[j],i+1)
X = np.array(X)



# Nothing fancy for outputs.
Y = np.array(republican_counts[:13])

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_spots = np.linspace(0, 160, 100)
Z = []
for d in range(100):
    Z.append([None]*26)
for a in range(100):
    Z[a][0] = 1
    for b in range(25):
        Z[a][b+1] = basis3(grid_spots[a],b+1)
Z = np.array(Z)

#grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(Z, w)


def sunleastsquares(size, f):
    total = 0
    for ind in range(13):
        current = sunspot_counts[ind]
        arr = []
        arr.append([1])
        for p in range(size):
            arr.append([f(current,p+1)])
        arr = np.array(arr)
        thispred = np.dot(arr.T,w)
        total+=(thispred - Y[ind])*(thispred - Y[ind])
    return 0.5*total

print("here is the loss")
print(sunleastsquares(25, basis3))

# TODO: plot and report sum of squared error for each basis

# Plot the data and the regression line.
plt.plot(sunspot_counts, Y, 'o', grid_spots, grid_Yhat, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()