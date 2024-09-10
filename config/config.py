from argparse import ArgumentParser, Namespace
from .bert_tagger import bert_tagger_parser_builder
from .bert_lstm_tagger import bert_lstm_tagger_parser_builder


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, dest="seed")
    parser.add_argument("--gpus", nargs="+", default=0, dest="gpus")
    # add dataset name if dev or test subcommands
    parser.add_argument("--output_dir", default="results/model_name", type=str, dest="output_dir")
    parser.add_argument("--debug", action="store_true", dest="debug")

    subparsers = parser.add_subparsers(title="subcommands", description="valid subcommands", dest="subcommand")
    train_subparser = subparsers.add_parser(name="train")
    train_parser_builder(train_subparser)
    dev_subparser = subparsers.add_parser(name="dev")
    dev_parser_builder(dev_subparser)
    test_subparser = subparsers.add_parser(name="test")
    test_parser_builder(test_subparser)

    args = parser.parse_args()

    return args


def train_parser_builder(parser: ArgumentParser) -> None:
    # add train specific argument
    parser.add_argument("train_data_path", type=str)
    parser.add_argument("dev_data_path", type=str)
    parser.add_argument("test_data_path", type=str)
    parser.add_argument("--mini_batch_size", default=4, type=int, dest="mini_batch_size")
    parser.add_argument("--accumulation_steps", default=4, type=int, dest="accumulation_steps")
    parser.add_argument("--evaluate_batch_size", default=8, type=int, dest="evaluate_batch_size")
    parser.add_argument("--lr", default=5e-5, type=float, dest="lr")
    parser.add_argument("--no_pret_lr", default=1e-3, type=float, dest="no_pret_lr")
    parser.add_argument("--warmup_steps", default=400, type=int, dest="warmup_steps")
    parser.add_argument("--clip_grad_norm", default=400., type=float, dest="clip_grad_norm")
    parser.add_argument("--epochs", default=80, type=int, dest="epochs")
    parser.add_argument("--patience", default=6, type=int, dest="patience")
    parser.add_argument("--metric", default="+BLEU-2", type=str, dest="metric")

    add_model_subparsers(parser)


def dev_parser_builder(parser: ArgumentParser) -> None:
    # add dev specific argument
    parser.add_argument("model_path", type=str)
    parser.add_argument("dev_data_path", type=str)

    # add model subparsers
    add_model_subparsers(parser)


def test_parser_builder(parser: ArgumentParser) -> None:
    # add dev specific argument
    parser.add_argument("model_path", type=str)
    parser.add_argument("test_data_path", type=str)

    # add model subparsers
    add_model_subparsers(parser)


def add_model_subparsers(parser: ArgumentParser) -> None:
    model_parsers = parser.add_subparsers(title="model names", description="valid model names", dest="model_name")

    bert_parser = model_parsers.add_parser(name="bert_tagger")
    bert_tagger_parser_builder(bert_parser)

    bert_lstm_parser = model_parsers.add_parser(name="bert_lstm_crf_tagger")
    bert_lstm_tagger_parser_builder(bert_lstm_parser)
