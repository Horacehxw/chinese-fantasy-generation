from pathlib import Path
from transformers import BertTokenizerFast
from torchtext.data import Dataset, Field, Example, TabularDataset, BucketIterator
from src.prepare_data import split_sentence, clean_text
import json
import hanlp
import torch
import dill
import argparse

# parser = argparse.ArgumentParser(description="Generate datasets from data.json file")
# parser.add_argument("--use_cws", action="store_true", help="use chinese word segmentation in tokenization")
# arg = parser.parse_args()

data_dir = (Path(__file__) / "../../data").resolve()
book_list = {"唐家三少":["斗罗大陆", "斗罗大陆II绝世唐门", "酒神"],
             "天蚕土豆":["斗破苍穹", "武动乾坤", "大主宰", "魔兽剑圣异界纵横"],
             "猫腻":["庆余年", "间客", "将夜", "朱雀记", "择天记"]
             }

# store the labels just for visualization usage
BOOK = Field(sequential=False)
AUTHOR = Field(sequential=False)
# preprocessing & store data
fast_tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
fast_tokenizer.add_tokens(["…", "[EOS]", "[SOS]"])
# set [SEP] to stop words if neccessary
# FIXME: return sequence length before padding.
TEXT = Field(init_token="[SOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            stop_words=["[SEP]"],
            tokenize=fast_tokenizer.tokenize
           )

tokenizer = hanlp.load("PKU_NAME_MERGED_SIX_MONTHS_CONVSEG")
TEXT_cws = Field(init_token="[SOS]",
                eos_token="[EOS]",
                pad_token="[PAD]",
                unk_token="[UNK]",
                # stop_words=["[SEP]"],
                tokenize=tokenizer
               )

def generate_paragraph(text, max_count=10, comma=True, max_sent_len=30):
    """
    Generate paragraphs of 10 consecutive sentence.
    text: original text
    max_count: number of sentences per paragraph, default=10
    comma: whether to include [,，] in the separator
    max_sent_len: longest sentences allowed, default=30 (include 99% of all sentences in the dataset)
    """
    sent_count = 0
    # split text into list of sentences. (seqarated by .,!?。，！？, and ，, if comma=True)
    sentences = list(split_sentence(text, comma=comma))
    for idx_now, sentence in enumerate(sentences):
        if len(sentence) > max_sent_len: # or len(split_sentence(sentence, comma=True)) > 3:
            sent_count = 0
            continue
        if sent_count >= max_count-1:
            yield "[SEP]".join(sentences[idx_now+1-max_count: idx_now+1])
            sent_count = 0
        else:
            sent_count += 1

def build_datapoints():
    file = data_dir / "data.json"
    if file.exists():
        return
    datapoint_list = []
    for author in book_list:
        author_dir = data_dir / author
        for book in book_list[author]:
            print("Processing {} of {}".format(book, author))
            book_file = author_dir / "{}.clean.txt".format(book)
            text = book_file.read_text(encoding="utf-8")
            for paragraph in generate_paragraph(text, comma=False, max_count=1, max_sent_len=50):
                datapoint = {"author": author,
                             "book": book,
                             "text": paragraph
                             }
                datapoint_list.append(datapoint)
            print("Processed: {} of {}".format(book, author))
    # remove the paragraphs whose length is smaller than 60 (ignore the "[SEP]" marks)
    # datapoint_list_clip = [x for x in datapoint_list if len(x["text"])-len("[SEP]")*9 > 60]
    datapoint_list_clip = [x for x in datapoint_list if len(x["text"]) > 10]
    print("Generate {} data points in total".format(len(datapoint_list_clip)))
    with file.open(mode="w") as f:
        json.dump(datapoint_list_clip, f)

def load_datapoints():
    file = data_dir / "data.json"
    if not file.exists():
        build_datapoints()
    with file.open(mode="r") as f:
        datapoints = json.load(f)
    return datapoints

def build_dataset(cws=False):
    if cws:
        dataset_path = data_dir / "dataset" / "cws"
    else:
        dataset_path = data_dir / "dataset" / "char"
    if (dataset_path/"examples.pkl").exists():
        return
    dataset_path.mkdir(parents=True, exist_ok=True)
    datapoints = load_datapoints()
    if cws:
        TXT_field = TEXT_cws
    else:
        TXT_field = TEXT
    example_list = [datapoint2example(x, cws) for x in datapoints]
    dataset = Dataset(example_list,
                      fields={"author": AUTHOR,
                              "book": BOOK,
                              "text": TXT_field
                              }
                      )
    torch.save(dataset.examples, dataset_path / "examples.pkl", pickle_module=dill)


def load_dataset(cws=False):
    if cws:
        dataset_path = data_dir / "dataset" / "cws"
    else:
        dataset_path = data_dir / "dataset" / "char"
    if not (dataset_path/"examples.pkl").exists():
        build_dataset(cws)
    examples_load = torch.load(dataset_path / "examples.pkl", pickle_module=dill)
    if cws:
        TXT_field = TEXT_cws
    else:
        TXT_field = TEXT
    dataset = Dataset(examples_load,
                      fields={"author": AUTHOR,
                              "book": BOOK,
                              "text": TXT_field
                              }
                      )
    return dataset

def datapoint2example(datapoint, cws=False):
    if cws:
        TXT_field = TEXT_cws
    else:
        TXT_field = TEXT
    return Example.fromdict(datapoint,
                            fields={"author": ("author", AUTHOR),
                                  "book": ("book", BOOK),
                                  "text": ("text", TXT_field)
                                   }
                           )

def get_itrerators(opt):
    """
    Get dataset iterator and necessary fields information
    :param opt: opt from argparser
    :return:
    """
    import random
    random.seed(42)
    dataset = load_dataset(opt.use_cws)
    dataset.fields["text"].build_vocab(dataset)
    dataset.fields["author"].build_vocab(dataset)
    dataset.fields["book"].build_vocab(dataset)
    train, test = dataset.split(split_ratio=0.7)
    train_iter = BucketIterator(train, batch_size=opt.train_batch_size, shuffle=True)
    test_iter = BucketIterator(test, batch_size=opt.eval_batch_size, shuffle=False)
    return train_iter, test_iter, dataset.fields


if __name__ == "__main__":
    use_cws = False
    build_datapoints()
    build_dataset(use_cws)
    dataset= load_dataset(use_cws)
    print("The {}th text of dataset_char is:\n\n ----------------------"
          "-------------- \n\n{}\n\n---------------------------------\n".format(200, dataset[200].text))