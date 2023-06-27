
import numpy as np

import random


def fill_missing_values(arr):

    cnter = 0
    mem = 0
    for i in range(len(arr)):
        if arr[i] is None:
            continue
        else:
            mem = arr[i]
            cnter += 1
    if cnter == len(arr):
        return arr
    if cnter == 1:
        for i in range(len(arr)):
            arr[i] = mem
        return arr


    # left to right
    for i in range(len(arr)):
        if arr[i] is None:
            dist = 0
            left_ind, right_ind = None, None
            left, right = None, None
            for j in range(i, len(arr)):
                if arr[j] is not None:
                    if left is None:
                        left = arr[j]
                        left_ind = j
                    else:
                        right = arr[j]
                        right_ind = j
                        break
                if left is not None:
                    dist += 1
            step_dist = (right - left) / dist
            arr[i] = left - (left_ind - i) * step_dist
        else:
            break

    # right to left
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] is None:
            dist = 0
            left_ind, right_ind = None, None
            left, right = None, None
            for j in range(len(arr) - 1, -1, -1):
                if arr[j] is not None:
                    if right is None:
                        right = arr[j]
                        right_ind = j
                    else:
                        left = arr[j]
                        left_ind = j
                        break
                if right is not None:
                    dist += 1
            step_dist = (right - left) / dist
            arr[i] = right + (i - right_ind) * step_dist
        else:
            break

    # middle
    for i in range(len(arr)):
        if arr[i] is None:
            left_dist, right_dist = 0, 0
            left_ind, right_ind = None, None
            left, right = None, None
            for j in range(i - 1, -1, -1):
                left_dist += 1
                if arr[j] is not None:
                    left = arr[j]
                    left_ind = j
                    break
            for j in range(i + 1, len(arr), 1):
                right_dist += 1
                if arr[j] is not None:
                    right = arr[j]
                    right_ind = j
                    break
            step_dist = (right - left) / (left_dist + right_dist)
            #print(right, left, left_dist, left_ind, right_dist, right_ind, step_dist)
            #assert left + (i - left_ind) * step_dist == right - (right_ind - i) * step_dist
            arr[i] = left + (i - left_ind) * step_dist

    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = 0

    return arr

def make_same_length(arr1, arr2):
    max_len = np.max([len(arr1), len(arr2)])
    list = np.arange(max_len, dtype=int)
    diff_len = np.abs(len(arr1) - len(arr2))
    insert_ind = random.sample(sorted(list), diff_len)
    insert_ind = np.sort(insert_ind)
    arr_mod = []
    cnter = 0
    another_cnter = 0
    if max_len == len(arr1):
        for i in range(max_len):
            if cnter == len(insert_ind) or i != insert_ind[cnter]:
                arr_mod.append(arr2[another_cnter])
                another_cnter += 1
            else:
                arr_mod.append(None)
                cnter += 1
        return arr_mod
    elif max_len == len(arr2):
        for i in range(max_len):
            if cnter == len(insert_ind) or i != insert_ind[cnter]:
                arr_mod.append(arr1[another_cnter])
                another_cnter += 1
            else:
                arr_mod.append(None)
                cnter += 1
        return arr_mod
    else:
        print('ERROR!')
        return None


if __name__ == '__main__':
    '''
    df = None
    indices = df.iloc[:, 0].to_numpy()
    indices = np.sort(indices)
    data = df.iloc[:, 1].to_numpy()
    data_mod = []
    ind = 0
    for i in range(72):
        if i in indices:
            data_mod.append(data[ind])
            ind += 1
        else:
            data_mod.append(None)
    '''
    data_mod = [None, None, 4, 5]
    arr = np.array(data_mod)
    a = fill_missing_values(arr)
    print(a)
