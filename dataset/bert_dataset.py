import jsonlines
from typing import List, Dict, Union
from torch.utils.data import Dataset
from utils import Vocabulary
from tqdm import tqdm
import logging
from config import LABEL_VOCAB_NAME


logger = logging.getLogger(__name__)


class BERTDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        super(BERTDataset, self).__init__()
        self._sentences: List[str] = []
        self._labels: List[List[str]] = []
        self._load_data(data_path)

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, index: int) -> Dict[str, Union[str, List[str]]]:
        return {"sentence": self._sentences[index], "label": self._labels[index]}

    @staticmethod
    def collate_fn(batch: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, Union[List[str], List[List[str]]]]:
        return {
            "sentences": [inst["sentence"] for inst in batch],
            "gold_labels": [inst["label"] for inst in batch]
        }

    def _load_data(self, data_path: str) -> None:
        with jsonlines.open(data_path, "r") as reader:
            for line in tqdm(reader):
                assert len(line["sentence"].split(" ")) == len(line["labels"].split(" ")), f"data error: {line}"
                self._sentences.append(line["sentence"])
                self._labels.append(line["labels"].split(" "))

    def get_sentences_length(self) -> List[int]:
        return [len(sentence.split(" ")) for sentence in self._sentences]

    def build_vocabulary(self) -> Vocabulary:
        logger.info("Building vocabulary...")
        vocab = Vocabulary()
        for labels in self._labels:
            for label in labels:
                vocab.add_token(LABEL_VOCAB_NAME, label)
        logger.info(f"Vocabulary of label: {vocab.get_vocab(LABEL_VOCAB_NAME)}")
        return vocab
