import jsonlines
from tqdm import tqdm


def conll03_preprocess():
    for read_name, write_name in zip(["train", "testa", "testb"], ["train", "dev", "test"]):
        with open(f"./data/conll03/origin/eng.{read_name}", "r", encoding="utf-8") as f_reader,\
        jsonlines.open(f"./data/conll03/{write_name}.jsonl", "w") as f_writer:
            words, labels = [], []
            for line in tqdm(f_reader):
                line = line.strip()
                if line.startswith("-DOCSTART-"):
                    continue
                if line == "":
                    if len(words) > 0:
                        f_writer.write({"sentence": " ".join(words), "labels": " ".join(labels)})
                    words, labels = [], []
                else:
                    line_content = line.split()
                    words.append(line_content[0])
                    labels.append(line_content[-1])
        assert len(words) == 0, f"last sentence is not empty in {read_name}: {words}"


if __name__ == '__main__':
    conll03_preprocess()
