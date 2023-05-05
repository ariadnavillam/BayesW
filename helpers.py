import numpy as np
import random


def find_missing_number(arr, bounds):
    min_val = min(bounds)
    max_val = max(bounds)
    
    # add random numbers to array
    while True:
        num = random.uniform(min_val, max_val)
        if num not in arr:
            return num

def find_positions_between_limits(array, bounds: tuple):
    
    bool_array = np.logical_and(( array[:, 0 ] >=   bounds[0]),  ( array[:,1] <=    bounds[1]))
    positions = np.where(bool_array)[0]
    return positions

## non-restrictive
# def find_positions_between_limits(array, bounds: tuple):
    
#     bool_array = np.logical_and(( bounds[0] <= array[:, 1 ]),  (  bounds[1] >=    array[:,0]))
#     positions = np.where(bool_array)[0]
#     return positions

if __name__ == "__main__":
    pass
