import re
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import ru2
import pymorphy2
import numpy as np
import xml.etree.ElementTree as ET
from spacy.lang.ru import Russian


# from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS


def spacy_tokenizer(text, lemm: bool):
    """токенизатор на основе библиотеки spacy, учитывающий особенности русского языка
    лемматизация как параметр"""

    nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(text)

    # ВЗЯТЬ РУССКИЙ ТОКЕНИЗАТОР https://github.com/antongolubev5/spacy_russian_tokenizer
    # nlp = Russian()
    # russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    # nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')

    text = [token.lemma_ for token in doc] if lemm else text

    punc_list = set(' –!"@#$%^&*()*+_,.\:;<>=?[]{}|~`/«»—' + '0123456789')
    output = []

    for i in range(len(text)):
        text[i] = re.sub(" +", " ", text[i])
        text[i] = text[i].lower()
        if not (text[i] in punc_list):
            output.append(text[i])

    return output


def mkdir_labeled_texts(directory_path, corpus_name, new_dir_name):
    """создание папки, в которую будем добавлять размеченные тексты"""
    for month in os.listdir(os.path.join(directory_path, corpus_name)):
        for day in os.listdir(os.path.join(directory_path, corpus_name, month)):
            for utf in os.listdir(os.path.join(directory_path, corpus_name, month, day)):
                os.mkdir(os.path.join(directory_path, corpus_name, month, day, utf, new_dir_name))


def selection_entities(directory_path, corpus_name, entities_with_sentiments):
    """
    выделение сущностей из текстов.
    сущности = размеченные pos/neg слова из русентилекс
    каждый файл из папки разметить и перекинуть в labeles_texts с соотв названием файла + labeled
    :return:
    :param directory_path:
    :param corpus_name:
    :param entities_with_sentiments: словарь тональных слов
    """

    for month in os.listdir(os.path.join(directory_path, corpus_name)):
        for day in os.listdir(os.path.join(directory_path, corpus_name, month)):
            for utf in os.listdir(os.path.join(directory_path, corpus_name, month, day)):
                for text_file in os.listdir(os.path.join(directory_path, corpus_name, month, day, utf, 'items')):
                    tree = ET.parse(
                        os.path.join(os.path.join(directory_path, corpus_name, month, day, utf, 'items', text_file)))
                    text = tree.getroot()[0].text
                    f = open(os.path.join(directory_path, corpus_name, month, day, utf, 'labeled_items',
                                          text_file[:-4] + '_labeled.txt'), "w")
                    f.write(text + '\n')
                    f.write('\n')
                    text = spacy_tokenizer(text, True)
                    f.write('ТОНАЛЬНЫЕ СЛОВА: ' + '\n')
                    for word in text:
                        if word in entities_with_sentiments.keys():
                            f.write(word + ' : ' + entities_with_sentiments[word] + '\n')

                    f.write('\n')
                    f.write('НЕТОНАЛЬНЫЕ СЛОВА: ' + '\n')
                    for word in text:
                        if not (word in entities_with_sentiments.keys()):
                            f.write(word + '\n')
                    f.close


def main():
    """
    нужно из словаря тональных существительных вытащить те, которыы описывают людей
    и набрать выборку - если встретилось подобное слово - то контекст негативный
    """

    directory_path = '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis'
    corpus_name = 'Rambler_source_test3'

    nouns_pos = open(os.path.join(directory_path, 'nouns_pos'), 'r')
    nouns_neg = open(os.path.join(directory_path, 'nouns_neg'), 'r')
    adjs_pos = open(os.path.join(directory_path, 'adjs_pos'), 'r')
    adjs_neg = open(os.path.join(directory_path, 'adjs_neg'), 'r')

    entities_with_sentiments = {}

    # словарь слов из rusentilex: key=сущность, value=тональность
    for file in [nouns_pos, nouns_neg, adjs_pos, adjs_neg]:
        for line in file:
            line_info = line.strip().split(', ')
            word = line_info[0]
            if not (word in entities_with_sentiments.keys()):
                entities_with_sentiments[word] = line_info[3]

    for file in [nouns_pos, nouns_neg, adjs_pos, adjs_neg]:
        file.close()

    text = 'инженер-программист Владимир Путин устроил самый настоящий разнос кое-какому губернатору Московской области Борису Громову'
    tokens = spacy_tokenizer(text, True)
    print(tokens)
    print([token for token in tokens if token in list(entities_with_sentiments.keys())])

    # mkdir_labeled_texts(directory_path, corpus_name, 'labeled_items')
    # selection_entities(directory_path, corpus_name, entities_with_sentiments)


if __name__ == '__main__':
    main()
