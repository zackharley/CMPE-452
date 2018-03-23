from typing import Callable, List


def pipe(funcs: List[Callable]) -> Callable:
    for func in funcs:
        if not callable(func):
            raise (TypeError('Expected a function'))

    def helper(*args):
        result = args[0]
        for func in funcs:
            result = func(result)
        return result

    return helper
