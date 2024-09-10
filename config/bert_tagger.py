from argparse import ArgumentParser


def bert_tagger_parser_builder(parser: ArgumentParser) -> None:
    parser.add_argument("--plm_dir", default="../model/bert-base-uncased", type=str, required=True, dest="plm_dir")
