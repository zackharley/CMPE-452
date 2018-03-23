from functools import reduce


def get_output_classes(data: list) -> list:
    """
    Create a list containing all of the unique output classes in a dataset

    Example:
    dataset = [
        [#,#,#,10],
        [#,#,#,4],
        [#,#,#,10],
        [#,#,#,3]
    ]
    f(dataset) -> [3, 4, 10]

    :param data: The dataset
    :return output_classes: The output classes for the dataset
    """
    output_classes = list(
        reduce(
            lambda accum, row: {*accum, row.get('expected_output')},
            data,
            set()
        )
    )
    output_classes.sort()
    return output_classes
