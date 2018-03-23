import csv as csv_lib
import sys
from math import sqrt
from typing import List
from functools import reduce
from assignment2.lib.utils.pipe import pipe
from assignment2.lib.backpropagation_artificial_neural_network import BackpropagationArtificialNeuralNetwork


def convert_data_points_to_float(csv: List[List]) -> List[List]:
    """
    Converts all of the data points in a dataset to floats. Excludes the header row.
    :param csv: The dataset to convert
    :return: The converted dataset
    """
    csv_copy = csv.copy()
    for i in range(1, len(csv_copy)):  # skip header row
        row = csv_copy[i]
        for j in range(0, len(row)):
            val = row[j]
            csv_copy[i][j] = float(val)
    return csv_copy


def gaussian_normalization(csv: List[List]) -> List[List]:
    """
    Normalizes CSV data before ANN using Gaussian normalization
    :param csv: The CSV data to normalize
    :return: A normalized list of CSV data
    """

    csv_copy = csv.copy()
    data = remove_fields_for_normalization(csv_copy)

    means = get_means(data)
    std_deviations = get_standard_deviations(means, data)

    for i in range(1, len(csv_copy)):  # skip header row
        row = csv_copy[i]
        for j in range(0, len(row)):
            val = csv_copy[i][j]
            csv_copy[i][j] = normalize_data_point(val, means[j], std_deviations[j])

    return csv_copy


normalize_data = pipe([
    convert_data_points_to_float,
    gaussian_normalization
])


def normalize_data_point(val: float, mean: float, std_deviation: float) -> float:
    """
    Normalize a data point using Gaussian normalization
    :param val: The value to normalize
    :param mean: The mean calculated for the entire column
    :param std_deviation: The standard deviation calcluated for the entire column
    :return: The normalized data point value
    """
    return (val - mean) / std_deviation


def get_means(data: List[List]) -> List[float]:
    """
    Get the mean values for all columns in a dataset
    :param data: The dataset for which means will be calculated
    :return: A list of the means for the dataset
    """

    def add_row_to_sums(accum, row):
        for index in range(0, len(row)):
            accum[index] += row[index]
        return accum

    sums = reduce(
        add_row_to_sums,
        data,
        [0 for x in range(0, len(data[0]))]
    )

    return list(map(
        lambda val: val / len(data),
        sums
    ))


def get_standard_deviations(means: List[float], data: List[List]) -> List[float]:
    """
    Get the standard deviation values for all columns in a dataset
    :param means: The means calculated for the dataset
    :param data: The dataset for which standard deviations will be calculated
    :return: A list of the standard deviations for the dataset
    """

    def add_row_to_sums(accum, row):
        for index in range(0, len(row)):
            accum[index] += (row[index] - means[index]) ** 2
        return accum

    sums = reduce(
        add_row_to_sums,
        data,
        [0 for x in range(0, len(data[0]))]
    )

    return list(map(
        lambda val: sqrt(val / len(data)),
        sums
    ))


def remove_header(csv: List[List]) -> List[List]:
    """
    Remove the header row from a CSV
    :param csv: The CSV to alter
    :return: The altered CSV
    """
    return csv[1:]


"""
A function used to remove all fields not needed for data normalization
:param csv: The CSV which needs to have fields removed
:return: The CSV that is ready to be normalized
"""
remove_fields_for_normalization = pipe([
    remove_header
])

if __name__ == "__main__":
    args = sys.argv[1:]
    csv_filename = args[0]

    with open(csv_filename) as csv_file:
        csv_data = list(csv_lib.reader(csv_file, delimiter=','))
        normalized_data = normalize_data(csv_data)

        # for row in normalized_data:
        #     print(row)

        wine_features = {'input': normalized_data[0][:-1], 'expected_output': normalized_data[0][-1]}
        wine_data = list(map(lambda row: {'input': row[:-1], 'expected_output': row[-1]}, normalized_data[1:]))

        nn = BackpropagationArtificialNeuralNetwork(wine_features, wine_data)

        training_data = wine_data[0:560]
        testing_data = wine_data[560:]
        nn.train(training_data)
        nn.compute(testing_data)

