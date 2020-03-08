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
# import spacy_russian_tokenizer


def spacy_tokenizer(text, lemm: bool):
    """токенизатор на основе библиотеки spacy, учитывающий особенности русского языка
    лемматизация как параметр"""

    nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(text)

    # ВЗЯТЬ РУССКИЙ ТОКЕНИЗАТОР https://github.com/antongolubev5/spacy_russian_tokenizer
    # nlp = Russian()
    # doc = nlp(text)
    # russian_tokenizer = spacy_russian_tokenizer.RussianTokenizer(nlp, spacy_russian_tokenizer.MERGE_PATTERNS)
    # nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    # doc = nlp(text)

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


def searching_entities_in_corpus(directory_path, corpus_name, entities_with_sentiments):
    """
    поиск тональных сущностей в текстах.
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


def creating_entities_vocab(directory_path, files: list):
    """
    выделение из словаря русентилекс тональных слов (positive/negative)
    :param directory_path: путь к файлам
    :param files: список файлов, из которых необходимо достать слова
    :return: словарь тональных слов
    """

    entities_with_sentiments = {}

    for i in range(len(files)):
        files[i] = open(os.path.join(directory_path, files[i]), 'r')

    for file in files:
        for line in file:
            line_info = line.strip().split(', ')
            word = line_info[0]
            if not (word in entities_with_sentiments.keys()):
                entities_with_sentiments[word] = line_info[3]
    file.close()

    return entities_with_sentiments


def searching_contexts_by_entities(directory_path, corpus_name):
    """
    по имеющимся сущностям набираем из корпуса выборку контекстов
    :param directory_path:
    :param corpus_name:
    :return:
    """

    contexts_for_entities = open(os.path.join(directory_path, 'contexts_for_labeled_entities'), 'w')
    entities_vocab = creating_entities_vocab(directory_path, ['nouns_person_neg'])

    # пробегаем по ВСЕМ текстам корпуса и выискиваем предложения, содержащие размеченные слова из словаря
    # пока просто принтим
    for month in os.listdir(os.path.join(directory_path, corpus_name)):
        for day in os.listdir(os.path.join(directory_path, corpus_name, month)):
            for utf in os.listdir(os.path.join(directory_path, corpus_name, month, day)):
                for item in os.listdir(os.path.join(directory_path, corpus_name, month, day, utf, 'items')):
                    tree = ET.parse(
                        os.path.join(os.path.join(directory_path, corpus_name, month, day, utf, 'items', item)))
                    text = tree.getroot()[0].text
                    print(text)

                for text in os.listdir(os.path.join(directory_path, corpus_name, month, day, utf, 'texts')):
                    f = open(os.path.join(directory_path, corpus_name, month, day, utf, 'texts', text), 'r')
                    list2vertical(text2sentences(f.read()))
                    f.close()


def searching_personal_entities(directory_path, file_from, file_to):
    """
    поиск сущностей, которыми можно охарактеризовать людей и запись их в другой файл
    :param directory_path: путь до файлов
    :param file_from: откуда брать сущности
    :param file_to: куда перекладывать сущности
    :return:
    """

    file_from = open(os.path.join(directory_path, file_from), 'r+')
    file_to = open(os.path.join(directory_path, file_to), 'w')

    for line in file_from:
        print(line.strip())
        if input() == 'y':
            file_to.write(line)


def text2sentences(text):
    """
    разделение текста на предложения
    """

    nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]

    return sentences


def list2vertical(lst):
    """
    вертикальная печать списка предложений
    """

    for element in lst:
        print(element)


def main():
    """
    нужно из словаря тональных существительных вытащить те, которые описывают людей
    и набрать выборку - если встретилось подобное слово - то контекст негативный
    """

    directory_path = '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis'
    corpus_name = 'Rambler_source_test3'

    f = open(
        '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source_test3/201101/20110101/20110101000749_utf/texts/22655843.txt',
        'r')
    text = f.read()
    print(text2sentences(text))

    # пробуем вытащить персональные сущ
    # searching_personal_entities(directory_path, 'nouns_neg', 'nouns_person_neg')

    # entities_with_sentiments = creating_entities_vocab(directory_path,
    #                                                    ['adjs_neg', 'adjs_neg', 'nouns_neg', 'nouns_pos'])

    # разметка и выделение
    # mkdir_labeled_texts(directory_path, corpus_name, 'labeled_items')
    # searching_entities_in_corpus(directory_path, corpus_name, entities_with_sentiments)

    searching_contexts_by_entities(directory_path, corpus_name)

    print(0)
    # text = 'инженер-программист Владимир Путин устроил самый настоящий разнос кое-какому губернатору Московской области Борису Громову'
    # tokens = spacy_tokenizer(text, True)
    # print(tokens)
    # print([token for token in tokens if token in list(entities_with_sentiments.keys())])


if __name__ == '__main__':
    main()
