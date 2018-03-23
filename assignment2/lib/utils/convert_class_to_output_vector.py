def convert_class_to_output_vector(output_classes: list, output_class: any, epsilon: int = 0) -> list:
    """
    Converts an output class to a one-hot encoded output vector.

    Example:
    f([5, 7, 8], 5) -> [1, 0, 0]
    f([5, 7, 8], 8) -> [0, 0, 1]
    f([5, 7, 8], 8, 0.05) -> [0.05, 0.05, 0.95]

    :param epsilon: The constant to offset the preferred output value by. Defaults to 0 when not specified.
    When using sigmoid activation functions, you want to avoid a value of 0 as the function would typically
    approach 1 asymptotically.
    :param output_classes: A list of all of the output classes for the network
    :param output_class: The value of the output class to convert
    :return output_vector: A vector that is one-hot encoded to identify which class it represents
    """
    if output_class not in output_classes:
        raise IndexError('Supplied output class does is not a valid output class.')

    output_classes.sort()
    output_vector = list(map(lambda x: (1 - epsilon) if x == output_class else epsilon, output_classes))
    return output_vector
