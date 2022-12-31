class LazyArray:
    """Iterable array with generating function"""

    def __init__(self, gen, size) -> None:
        self.index = 0
        self.size = size
        self.gen = gen
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index <= self.size:
            return self.gen(self.index - 1)
        else:
            raise StopIteration

    def __len__(self):
        return self.size
