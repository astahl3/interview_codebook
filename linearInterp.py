'''
Simple linear interpolator that provides interpolated values using the set of
input points given by x (x points) and y (y points) where len(x) == len(y) == L

Interpolated values are only given within the range [min(x), max(x)]

'''

import numpy as np

class linInterpolator:
    def __init__(self, x, y):
        
        z = sorted(zip(x,y))
        x_sorted, y_sorted = zip(*z)
        self.x = np.array(x_sorted)
        self.y = np.array(y_sorted)
        self.m = (self.y[1:] - self.y[:-1]) / (self.x[1:] - self.x[:-1])
        self.b = self.y[1:] - self.m * self.x[1:]
        self.L = len(x)

    def __call__(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return np.nan
        else:
            k = np.searchsorted(self.x, x)
            if k == 0:
                return self.y[0]
            elif k == len(self.x):
                return self.y[-1]
            else:
                return self.m[k-1] * x + self.b[k-1]

# Example usage:
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

interp = linInterpolator(x, y)
print(interp(2.5))  # Output: 4.0
print(interp(4.5))  
print(interp(5))