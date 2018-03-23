from functools import reduce


def get_data_frame_means(data_frame):
    num_rows, num_cols = data_frame.shape

    means = list(
        map(
            lambda sum: sum / num_rows,
            reduce(
                lambda accum, row: list(map(lambda pair: pair[0] + pair[1], zip(accum, row[1:]))),
                data_frame.itertuples(),
                [0 for x in range(num_cols)]
            )
        )
    )

    return means


def center_data_frame(data_frame, means):
    centered_data_frame = list(
        map(
            lambda row: list(
                map(
                    lambda pair: pair[0] - pair[1],
                    zip(row[1:], means)
                )
            ),
            data_frame.itertuples()
        )
    )
    return centered_data_frame
