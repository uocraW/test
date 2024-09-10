from argparse import ArgumentParser


def bert_lstm_tagger_parser_builder(parser: ArgumentParser) -> None:
    parser.add_argument("--plm_dir", default="../model/bert-base-uncased", type=str, required=True, dest="plm_dir")
    parser.add_argument("--lstm_layers", default=2, type=int, required=True, dest="lstm_layers")
