import numpy as np
from typing import Iterator, List
from torch.utils.data import Sampler


class SentenceLengthSampler(Sampler[int]):

    def __init__(self, sentences_length: List[int]) -> None:
        super(SentenceLengthSampler, self).__init__()
        self._sentences_length = np.array(sentences_length)

    def __len__(self) -> int:
        return len(self._sentences_length)

    def __iter__(self) -> Iterator[int]:
        yield from np.argsort(self._sentences_length).tolist()
