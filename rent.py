import csv

import numpy as np
from matplotlib import pyplot as plt

import linear


def my_float(num_string):
    if num_string == 'NA':
        return np.nan
    else:
        return np.float(num_string)


def load(data_file):
    with open(data_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            header = row
            break
        return header, np.array([list(map(my_float, row)) for row in reader])


def not_nan(arr):
    return arr[np.invert(np.isnan(arr))]


def remove_nan_lines(d):
    return d[np.all(np.invert(np.isnan(d)), axis=1)]


def put_average_in_nan(arr):
    arr[np.isnan(arr)] = np.average(not_nan(arr))


def put_average_in_nan_lines(d):
    cpy = np.copy(d)
    for j in range(d.shape[1]):
        put_average_in_nan(cpy.T[j])
    return cpy


def main():
    z = np.array([[1, 2, np.nan], [1, 2, np.nan], [1, 2, np.nan], [1, 2, 12, ], [12, 2, 12]])
    print(z)
    print(put_average_in_nan_lines(z))
    print(remove_nan_lines(z))
    data_file = "apartment_rent.csv"
    header, d = load(data_file)
    print(header)
    print(d.shape)
    for i, column in enumerate(header):
        plt.title(column)
        l = list(not_nan(d.T[i]))
        plt.hist(l, bins=30, cumulative=True)
        # plt.show()
    rent_column = header.index("rent")
    columns_to_ignore = [rent_column, header.index("town"), header.index("city"), header.index("age"),
                         header.index("city")]
    columns_to_use = sorted(set(range(len(header))) - set(columns_to_ignore))
    header_X = [header[i] for i in columns_to_use]
    for d_no_nan in [put_average_in_nan_lines(d), remove_nan_lines(d)]:
        y = d_no_nan.T[rent_column]
        X = d_no_nan.T[columns_to_use].T
        print("X", X.shape)
        print("y", y.shape)
        a, b = linear.sk_linear_regeression(X, y)
        print(a.shape)
        residuals = linear.residuals(X, y, a, b)
        plt.cla()
        plt.title("residuals")
        plt.hist(residuals)
        plt.show()
        print("a", list(zip(header_X, a)))
        print("R=", linear.R(X, y, a, b))


if __name__ == '__main__':
    main()
