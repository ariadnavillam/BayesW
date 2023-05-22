import numpy as np
import random

def normalize_probabilities(probs):
    if probs.min() < 0:
        probs = probs - probs.min()
    return probs/probs.max()

def bounds_error(bounds):
    if bounds[0] == bounds[1]:
        return True
    else:
        return False
    
def find_missing_number(arr, bounds = None):
    '''
    Returns a number that is in the interval of an array. Bounds are optional.

    Input:
    - arr: array of numbers
    - bounds: define min and max values

    Output:
    - num: number sampled from a uniform from the min and max values
    '''
    if bounds == None or bounds_error(bounds):

        min_val = min(arr)
        max_val = max(arr)
    
    else:
        min_val = min(bounds)
        max_val = max(bounds)
    
    # add random numbers to array
    while True:
        num = random.uniform(min_val, max_val)
        if num not in arr:
            return num

def find_positions_between_limits(array, bounds):
    '''
    From an array of limits, find the ones that are included in certain interval 
    
    Input:
    - array: array of limits. The first column is the lower limit and the second the upper limit.
    - bounds: interval we want to search for in the limits.

    Output:
    - positions: indexes (rows) of array that are contained in the interval defined by the bounds
    '''
    
    bool_array = np.logical_and(( array[:, 0 ] >=   bounds[0]),  ( array[:,1] <=    bounds[1]))
    positions = np.where(bool_array)[0]
    return positions

## non-restrictive
# def find_positions_between_limits(array, bounds: tuple):
    
#     bool_array = np.logical_and(( bounds[0] <= array[:, 1 ]),  (  bounds[1] >=    array[:,0]))
#     positions = np.where(bool_array)[0]
#     return positions

def change_order(bounds):
    '''
    Changes the order of a tuple of bounds
    '''
    bounds = list(bounds)
    bounds[1], bounds[0] = bounds[0], bounds[1]
    return tuple(bounds)

if __name__ == "__main__":
    pass
