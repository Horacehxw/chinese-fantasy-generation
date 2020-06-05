"""
从 www.xshuyaya.com 下载 txt 网络小说。
仅用于学习、研究使用，版权归原作者所有。
"""
import os
import os.path as osp
import requests
import zipfile
from pathlib import Path
import re
from hanlp.utils.lang.zh.char_table import CharTable
import json
import hanlp

book_list = {"唐家三少":["斗罗大陆", "斗罗大陆II绝世唐门", "酒神"],
             "天蚕土豆":["斗破苍穹", "武动乾坤", "大主宰", "魔兽剑圣异界纵横"],
             "猫腻":["庆余年", "间客", "将夜", "朱雀记", "择天记"]
             }

download_pattern = "http://down.xshuyaya.com/zip/{}.zip"

data_dir = (Path(__file__)/"../../data").resolve() # use absolute path


def download_data():
    """
    Download raw txt file in following order.
    data
        - 作家1
            -书名1.txt
            -书名2.txt
        - 作家2
        ...
    """
    for author in book_list:
        author_dir = data_dir / author
        if not author_dir.is_dir():
            author_dir.mkdir()
        for book in book_list[author]:
            if (author_dir / "{}.txt".format(book)).exists():
                continue
            print("Downloading {}.".format(book))
            r = requests.get(download_pattern.format(book))
            open(author_dir / "{}.zip".format(book), "wb").write(r.content)
            with zipfile.ZipFile(author_dir / "{}.zip".format(book), "r") as zip_book:
                for fn_name in zip_book.namelist():
                    extracted_path = Path(zip_book.extract(fn_name, path=author_dir))
                    extracted_path.rename(author_dir / fn_name.encode("cp437").decode("gbk"))

def clean_data(process=8):
    """
    clean the raw txt file, store cleaned data in the same dir with suffix .clean.txt
    :return:
    """
    from multiprocessing import Pool
    book_pair = [(author, book) for author in book_list for book in book_list[author]]
    # with Pool(processes=process) as p:
    #     p.map(clean_book, book_pair)
    for pair in book_pair:
        if not (data_dir/pair[0]/"{}.clean.txt".format(pair[1])).exists():
            clean_book(pair)

def clean_book(book_pair):
    """
    clean the text in each book, store as {book name}.clean.txt
    :param book_pair: (author, book)
    :return: None
    """
    author, book = book_pair
    print("Processing: {} of {}".format(book, author))
    book_file = data_dir/author/"{}.txt".format(book)
    encoding = "utf-8"
    text = book_file.read_text(encoding=encoding, errors='replace')
    new_text = clean_text(text)
    new_book_file = data_dir/author/"{}.clean.txt".format(book)
    new_book_file.write_text(new_text, encoding=encoding)
    print("Processing finished: {} of {}".format(book, author))

def clean_text(s):
    """
    1. normalize characters
    2. 按照规则去除特殊表达:
        1. html tags
        2. 章节名
        3. 括号 () 中所有内容
        4. 去除 url 链接
        5. 合并连续重复标点、字
    :param s: original text
    :return: cleaned text

    :reference:
        1. https://www.pythonf.cn/read/52608
        2. https://gist.github.com/gruber/8891611
    """
    new_text = CharTable.normalize_text(s)
    new_text = re.sub(r"<.*>|\(.*\)|第.*章.*| +", '', new_text)  # remove html  & 括号 & 章节名 & 空格
    url_regex = re.compile(
        r"""(?i)\b((?:https?://|www\d{0,3}[。,，.]|[a-z0-9.\-]+[.。，,][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))""")
    new_text = re.sub(url_regex, "", new_text)
    new_text = re.sub("shuyaya.com|wap", "", new_text)  # adhoc data cleaning
    new_text = re.sub(r"[ #$%&`\'/@★\[\\\]^_{|}~]+", '', new_text)  # remove special characters
    new_text = re.sub(",+", ",", new_text)
    new_text = re.sub("。。。+|\.\.\.+|…+", "…", new_text)  # 省略号 in HanLP
    new_text = re.sub("-+", "-", new_text)  # 合并-
    new_text = re.sub("—+", "-", new_text)  # 合并———
    new_text = re.sub("\?+", "?", new_text)
    new_text = re.sub("!+", "!", new_text)
    # new_text = re.sub(r"([^\.])(\.)([^\.])", r"\1\3", new_text) # remove single dots

    # chinese_regex = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
    # for chinese_character in re.findall(chinese_regex, new_text):
    #     new_text = re.sub("[{" + chinese_character + "}]{3,}", chinese_character * 3, new_text)  # 三个以上中文字符合并成三个
    return new_text

def cws_data():
    tokenizer = hanlp.load("PKU_NAME_MERGED_SIX_MONTHS_CONVSEG")
    pipeline = hanlp.pipeline() \
        .append(split_sentence, output_key="sentences") \
        .append(tokenizer, output_key="tokens")
    book_results = {}

    for author in book_list:
        author_dir = data_dir / author
        for book in book_list[author]:
            book_res_file = author_dir / "{}.json".format(book)
            if book_res_file.exists():
                continue
            print("Processing: {} of {}".format(book, author))
            book_file = author_dir / "{}.clean.txt".format(book)
            book_text = book_file.read_text(encoding="utf-8")
            book_res = pipeline(book_text)
            book_results[book] = book_res
            with book_res_file.open(mode="w") as f:
                json.dump(book_res, f)
            print("Processing finished: {} of {}".format(book, author))

def cws_to_text():
    for author in book_list:
        author_dir = data_dir / author
        for book in book_list[author]:
            book_res_file = author_dir / "{}.json".format(book)
            book_cws_file = author_dir / "{}.cws.txt".format(book)
            if book_cws_file.exists():
                continue
            with book_res_file.open(mode="r") as f:
                book_res = json.load(f)
            tokens = book_res["tokens"]
            text = "\n".join([" ".join(sentence) for sentence in tokens])
            book_cws_file.write_text(text, encoding="utf-8")


#######################################################################
# Modified version from HanLP
from hanlp.utils.english_tokenizer import tokenize_english
import  re

SEPARATOR = r'@'
RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)
AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + SEPARATOR + r'(\w)', re.UNICODE)
UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + SEPARATOR + r'(\w)', re.UNICODE)


def replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def split_sentence(text, best=True, comma=False):
    text = re.sub('([。！？!\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…+)([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？!\?][”’])([^，。！!？\?])', r'\1\n\2', text)
    if comma:
        text = re.sub('([，,])([^”’])', r"\1\n\2", text)
        text = re.sub('([，,][”’])([^,，])', r'\1\n\2', text)
    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not best:
            yield chunk
            continue
        processed = replace_with_separator(chunk, SEPARATOR, [AB_SENIOR, AB_ACRONYM])
        for sentence in RE_SENTENCE.finditer(processed):
            sentence = replace_with_separator(sentence.group(), r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])
            yield sentence


#######################################################################

if __name__ == "__main__":
    # download_data()
    # clean_data() # Note that we have done some hand cleaning, like remove the head & tail, etc.
    # cws_data()
    # cws_to_text()
    pass