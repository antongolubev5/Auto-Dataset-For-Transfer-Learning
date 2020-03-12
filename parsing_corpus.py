import re
import os
import spacy
import xml.etree.ElementTree as ET
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
import time
from pymystem3 import Mystem
from string import punctuation

# import ru2e

spacy.prefer_gpu()


def regTokenize(text):
    """
    вроде как быстрый токенизатор?
    :param text:
    :return:
    """
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    # return ' '.join(words)
    return words


def myany(text: list, vocab: list):
    """
    проверка на наличие слов из словаря в строке
    """

    for word in text:
        if word in vocab:
            return True
    return False


def mystem_tokenizer(text):
    """
    токенизатор на основе mystem
    :param text:
    :return:
    """

    mystem = Mystem()
    tokens = mystem.lemmatize(text.lower())
    punc_list = ' –!"@#$%^&*()*+_,.\:;<>=?[]{}|~`/«»—' + '0123456789'
    tokens = [token for token in tokens if token != " " and token.strip() not in set(punctuation + punc_list)]

    # return tokens
    return ' '.join(tokens)


def spacy_tokenizer(text, lemm: bool):
    """токенизатор на основе библиотеки spacy, учитывающий особенности русского языка
    spacy_russian_tokenizer --- токенизация
    spacy_ru2 --- лемматизация (как параметр)
    стоп слова?
    не всегда правильно работает, надо разбираться с ru2e
    """

    # nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    # # nlp = spacy.load('ru2', disable=['tagger', 'parser', 'NER'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    # doc = nlp(text)

    # ВЗЯТЬ РУССКИЙ ТОКЕНИЗАТОР https://github.com/antongolubev5/spacy_russian_tokenizer
    nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    doc = nlp(text)

    text = [token.lemma_ for token in doc] if lemm else text

    punc_list = set(' –!"@#$%^&*()*+_,.\:;<>=?[]{}|~`/«»—' + '0123456789')
    output = []

    for i in range(len(text)):
        text[i] = re.sub(" +", " ", text[i])
        text[i] = text[i].lower()
        if not (text[i] in punc_list):
            output.append(text[i])

    # return output
    return ' '.join(output)


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
                    f.close()


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


def searching_contexts_by_entities(directory_path, corpus_name, entities_vocab: dict, nlp):
    """
    по имеющимся сущностям набираем из корпуса выборку контекстов
    :param nlp: модель для разбиения текста на предложения
    :param entities_vocab:
    :param directory_path:
    :param corpus_name:
    :return:
    ================================================================
    сделать try except!!!! чтоб не ломалась программа из-за отсутствия файла
    """

    contexts_for_entities = open(os.path.join(directory_path, 'contexts_for_labeled_entities_1'), 'w')
    list_entities_vocab_keys = list(entities_vocab.keys())

    # пробегаем по всем текстам корпуса и выискиваем предложения, содержащие размеченные слова из словаря
    # for month in os.listdir(os.path.join(directory_path, corpus_name)):
    month = '201101'
    for day in os.listdir(os.path.join(directory_path, corpus_name, month)):
        for utf in os.listdir(os.path.join(directory_path, corpus_name, month, day)):
            if len(os.listdir(os.path.join(directory_path, corpus_name, month, day, utf))) > 0:
                for item in os.listdir(os.path.join(directory_path, corpus_name, month, day, utf, 'items')):
                    tree = ET.parse(
                        os.path.join(os.path.join(directory_path, corpus_name, month, day, utf, 'items', item)))
                    text = tree.getroot()[0].text
                    # text_tok = spacy_tokenizer(text, True)
                    # text_tok = mystem_tokenizer(text)
                    # if any(word in list_entities_vocab_keys for word in text_tok):
                    #     print(text)
                    print(text)
                    # contexts_for_entities.write(text+'\n')
                for text in os.listdir(os.path.join(directory_path, corpus_name, month, day, utf, 'texts')):
                    f = open(os.path.join(directory_path, corpus_name, month, day, utf, 'texts', text), 'r')
                    # sent_tok = spacy_tokenizer(sent, True)
                    # sent_tok = mystem_tokenizer(sent)
                    # if any(word in list_entities_vocab_keys for word in sent_tok):
                    #     print(sent)
                    # contexts_for_entities.write(sent+'\n')
                    for sent in text2sentences(f.read(), nlp):
                        print(sent)
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


def text2sentences(text, nlp):
    """
    разделение текста на предложения
    """

    # nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    # nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]

    return sentences


def main():
    """
    нужно из словаря тональных существительных вытащить те, которые описывают людей
    и набрать выборку - если встретилось подобное слово - то контекст негативный
    """

    start_time = time.time()

    directory_path = '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis'
    corpus_name = 'Rambler_source_test3'

    # print(mystem_tokenizer(
    #     "инженер-программист Владимир Путин устроил самый настоящий разнос кое-какому губернатору Московской области Борису Громову. Не ветра, ни какого-то урагана!"))

    # пробуем вытащить персональные сущ
    # searching_personal_entities(directory_path, 'nouns_pos', 'nouns_person_pos')

    # entities_with_sentiments = creating_entities_vocab(directory_path,
    #                                                    ['adjs_neg', 'adjs_neg', 'nouns_neg', 'nouns_pos'])

    # разметка и выделение
    # mkdir_labeled_texts(directory_path, corpus_name, 'labeled_items')
    # searching_entities_in_corpus(directory_path, corpus_name, entities_with_sentiments)

    # поиск контекстов для существительных с отрицательной окраской
    # entities_with_sentiments = creating_entities_vocab(directory_path, ['nouns_person_neg'])
    # searching_contexts_by_entities(directory_path, corpus_name, entities_with_sentiments)

    # nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    # nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    #
    # entities_with_sentiments = creating_entities_vocab(directory_path, ['nouns_person_neg'])
    # searching_contexts_by_entities(directory_path, corpus_name, entities_with_sentiments, nlp)

    # fi = open(os.path.join(directory_path, 'Rambler_source_test3/201101/20110101/20110101000749_utf/texts/22655849.txt'), 'r')
    # text = fi.read()
    # # print(mystem_tokenizer(text))
    # print(spacy_tokenizer(text, True))
    # fi.close()

    vocab = []
    vocab_f = open(os.path.join(directory_path, 'nouns_person_neg'), 'r')
    for line in vocab_f:
        vocab.append(line.split(' ')[0])

    fi = open(os.path.join(directory_path, 'contexts_for_labeled_entities_2'), 'r')
    fi_to = open(os.path.join(directory_path, 'testing_parser_results'), 'w')

    for line in fi:
        if myany(regTokenize(line), vocab):
            fi_to.write(line.strip() + '===' + ' '.join(regTokenize(line)) + '\n')
            print(line.strip() + '===' + ' '.join(regTokenize(line)) + '\n')

    fi_to.close()

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))


if __name__ == '__main__':
    main()
