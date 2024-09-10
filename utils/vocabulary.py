from typing import Dict


class Vocabulary:
    def __init__(self) -> None:
        super(Vocabulary, self).__init__()
        self.token2id: Dict[str, Dict[str, int]] = {}
        self.id2token: Dict[str, Dict[int, str]] = {}

    def vocab_size(self, vocab: str) -> int:
        return len(self.token2id[vocab])

    def add_token(self, vocab: str, token: str) -> int:
        if vocab not in self.token2id:
            self.token2id[vocab] = {}
            self.id2token[vocab] = {}

        if token not in self.token2id[vocab]:
            self.token2id[vocab][token] = len(self.token2id[vocab])
            self.id2token[vocab][self.token2id[vocab][token]] = token

        return self.token2id[vocab][token]

    def get_token(self, vocab: str, token_id: int) -> str:
        return self.id2token[vocab][token_id]

    def get_token_id(self, vocab: str, token: str) -> int:
        return self.token2id[vocab][token]

    def get_vocab(self, vocab: str) -> Dict[str, int]:
        return self.token2id[vocab]

    def get_index_to_token_vocab(self, vocab: str) -> Dict[int, str]:
        return self.id2token[vocab]
