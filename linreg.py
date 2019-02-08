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
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
# Regular: X = np.vstack((np.ones(years.shape), years)).T
X = []
for d in range(len(years)):
    X.append([None]*6)

# TODO: basis functions
def basis0(x,i):
    return 1
def basis1(x,i):
    return x**i
def basis2(x,i):
    return math.exp(-(x-i)*(x-i)/25.0)
def basis3(x,i):
    return math.cos(x/i)

for j in range(len(years)):
    X[j][0] = 1
    for i in range(5):
        X[j][i+1] = basis1(years[j],i+1)
X = np.array(X)



# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
Z = []
for d in range(200):
    Z.append([None]*6)
for a in range(200):
    Z[a][0] = 1
    for b in range(5):
        Z[a][b+1] = basis1(grid_years[a],b+1)
Z = np.array(Z)

#grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(Z, w)


def leastsquares(size, f):
    total = 0
    for ind in range(len(years)):
        current = years[ind]
        arr = []
        arr.append([1])
        for p in range(size):
            arr.append([f(current,p+1)])
        arr = np.array(arr)
        thispred = np.dot(arr.T,w)
        total+=(thispred - republican_counts[ind])*(thispred - republican_counts[ind])
    return 0.5*total

print("here is the loss")
print(leastsquares(5, basis1))

# TODO: plot and report sum of squared error for each basis

# Plot the data and the regression line.
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.show()