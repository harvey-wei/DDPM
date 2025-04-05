from typing import Iterable, Iterator, Any


def get_infinite_generator(iterable: Iterable):
    '''
    Create an infinite generator from a finite iterable like a DataLoader
    This is different from the regular usage:
        for i in range(num_epochs):
            for image, label in DataLoader:
                ......
    The infinite loop is as follows:
        iter = get_data_iterator_v1(DataLoader)
        for i in range(num_iters):
            ...
            ...
            image, label = next(iter)

    why to use yeild to make get_data_iterator_v1 a generator?

    '''
    iterator = iter(iterable) # or iterable.__iter__()

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


if __name__ == "__main__":
    # Toy Example usage
    data = [1, 2, 3]
    infinite_gen = get_infinite_generator(data)

    for i in range(10):
        print(next(infinite_gen))
