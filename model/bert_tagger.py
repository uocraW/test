import torch
import logging
from torch import nn
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from utils import Vocabulary, ExactF1
from config import IGNORE_INDEX, LABEL_VOCAB_NAME


logger = logging.getLogger(__name__)


class BERTTagger(nn.Module):

    def __init__(self, plm_dir: str, vocab: Vocabulary) -> None:
        super(BERTTagger, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(plm_dir)
        self.plm = AutoModel.from_pretrained(plm_dir)

        self.vocab = vocab

        self.classifier = nn.Linear(self.plm.config.hidden_size, vocab.vocab_size(LABEL_VOCAB_NAME))
        self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=IGNORE_INDEX)

        self.exact_f1 = ExactF1()

    def forward(self, sentences: List[str], gold_labels: List[List[str]] = None) -> Dict[str, torch.Tensor]:
        res = dict()

        sentences_ids, attention_masks, offsets, words_masks = self.__tokenize_sentences(sentences)
        if gold_labels is not None:
            labels_ids = self.__tokenize_labels(gold_labels)
        plm_outputs = self.plm(
            input_ids=sentences_ids, attention_mask=attention_masks,
            output_attentions=False, output_hidden_states=True, return_dict=True
        )

        # 将每个word对应的多个token中的第一个token表示提取出来，将其作为整个word的表示。
        last_hidden_state = plm_outputs.last_hidden_state
        batch_size, seq_len, hidden_size = last_hidden_state.size()
        word_repr = torch.zeros(batch_size, offsets.size(1), hidden_size, device=last_hidden_state.device)
        for i in range(batch_size):
            word_repr[i] = last_hidden_state[i, offsets[i], :]


        logits = self.classifier(word_repr)
        pred_labels_ids = torch.argmax(logits, dim=-1)
        pred_labels = self.__labels_from_id_to_str(pred_labels_ids, words_masks)

        if self.training:
            self.exact_f1(pred_labels, gold_labels)
            loss = self.loss(logits.permute(0, 2, 1), labels_ids)
            res["loss"] = loss
        else:
            if gold_labels is not None:
                self.exact_f1(pred_labels, gold_labels)
                loss = self.loss(logits.permute(0, 2, 1), labels_ids)
                res["loss"] = loss
                res["pred_labels"] = pred_labels
            else:
                res["pred_labels"] = pred_labels
        return res

    def __labels_from_id_to_str(self, pred_labels_ids: torch.Tensor, masks: torch.Tensor) -> List[List[str]]:
        pred_labels = []
        for pred_sent_labels_ids, sent_mask in zip(pred_labels_ids, masks):
            pred_sent_labels_ids = pred_sent_labels_ids[:sent_mask.sum()].tolist()
            pred_sent_labels = [self.vocab.get_token(LABEL_VOCAB_NAME, label_id) for label_id in pred_sent_labels_ids]
            pred_labels.append(pred_sent_labels)
        return pred_labels

    def __tokenize_sentences(
        self, sentences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, attention_masks, offsets, words_masks = [], [], [], []
        for sentence in sentences:
            words = sentence.split(" ")
            sent_ids, sent_atten_masks, sent_offsets, sent_words_mask = [], [], [], []
            for word in words:
                word_ids = self.tokenizer(f" {word}", add_special_tokens=False, return_attention_mask=False)["input_ids"]
                word_atten_mask = [1 for _ in range(len(word_ids))]
                word_offsets = len(sent_ids) + 1
                sent_ids.extend(word_ids)
                sent_atten_masks.extend(word_atten_mask)
                sent_offsets.append(word_offsets)
                sent_words_mask.append(1)
            input_ids.append([self.tokenizer.cls_token_id] + sent_ids + [self.tokenizer.sep_token_id])
            attention_masks.append([1] + sent_atten_masks + [1])
            offsets.append(sent_offsets)
            words_masks.append(sent_words_mask)

        token_num = max([len(sent) for sent in input_ids])
        input_ids = [sent_ids + [self.tokenizer.pad_token_id for _ in range(token_num - len(sent_ids))] for sent_ids in input_ids]
        attention_masks = [sent_atten_masks + [0 for _ in range(token_num - len(sent_atten_masks))] for sent_atten_masks in attention_masks]
        word_num = max([len(sent_offsets) for sent_offsets in offsets])
        offsets = [sent_offsets + [token_num-1 for _ in range(word_num - len(sent_offsets))] for sent_offsets in offsets]
        words_masks = [sent_words_mask + [0 for _ in range(word_num - len(sent_words_mask))] for sent_words_mask in words_masks]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        attention_masks = torch.tensor(attention_masks, dtype=torch.bool, device="cuda")
        offsets = torch.tensor(offsets, dtype=torch.long)
        words_masks = torch.tensor(words_masks, dtype=torch.bool, device="cuda")

        return input_ids, attention_masks, offsets, words_masks

    def __tokenize_labels(self, labels: List[List[str]]) -> torch.Tensor:
        labels_len = max([len(sent_labels) for sent_labels in labels])
        labels_list = [
            [self.vocab.get_token_id(LABEL_VOCAB_NAME, label) for label in sent_labels] + [IGNORE_INDEX for _ in range(labels_len - len(sent_labels))]
            for sent_labels in labels
        ]
        return torch.tensor(labels_list, dtype=torch.long, device="cuda")

    def save_model(self, output_dir: str) -> None:
        self.plm.save_pretrained(f"{output_dir}/bert")
        torch.save({"classifier": self.classifier.state_dict(), "vocab": self.vocab}, f"{output_dir}/model.pt")

    def load_model(self, input_dir: str) -> None:
        self.plm = AutoModel.from_pretrained(f"{input_dir}/bert")
        checkpoint = torch.load(f"{input_dir}/model.pt", map_location="cuda")
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.vocab = checkpoint["vocab"]

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_f1 = self.exact_f1.get_metric(reset)
        return exact_f1

    def get_optimizer_parameter_group(self, lr_pretr_param: float, lr_no_pretr_param: float) -> List[Dict]:
        pretrained_params = []
        no_pretrained_params = []
        for name, param in self.named_parameters():
            if name.startswith("plm"):
                pretrained_params.append(param)
            else:
                no_pretrained_params.append(param)

        return [
            {"params": pretrained_params, "lr": lr_pretr_param},
            {"params": no_pretrained_params, "lr": lr_no_pretr_param},
        ]

    def make_output_human_readable(self, prediction):
        "output prediction accroding to the specific requirement, see evaluater"
        pass
