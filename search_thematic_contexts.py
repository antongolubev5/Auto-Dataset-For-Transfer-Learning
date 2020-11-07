from parsing_corpus import *

"""
Я думала, что результат возможно улучшить тем, что в нейтральные контексты набрать именно контексты для банков,
т.е. взять тот же список банков, и если вокруг нет явных тональностей, то считать контекст нейтральным.

применяем весь русентилекс
/home/anton/data/ABSA/contexts/txtRuSentiLex2017_revised.txt

файлы с предложениями:
/home/anton/data/ABSA/contexts/txt/contexts_for_labeled_entities_2
/home/anton/data/ABSA/contexts/txt/contexts_for_labeled_entities_3
"""

from tqdm import tqdm
import os


def searching_contexts(directory_path, entities_vocabs: list, sentences_file, contexts_file, sentence_volume):
    """
    поиск тональных контекстов cреди предложений корпуса
    :param directory_path:
    :param entities_vocabs: список из названий файлов, в которых лежат тональные слова
    :param sentences_file: файл с предложениями из корпуса
    :param contexts_file: файл, в который будут записаны контексты
    :param sentence_volume: сколько предложений из корпуса рассматривать [0:vol]
    """

    vocab = {}
    vocab_neg = open(os.path.join(directory_path, entities_vocabs[0]), 'r')
    vocab_pos = open(os.path.join(directory_path, entities_vocabs[1]), 'r')

    for line in vocab_pos:
        line_info = line.split(', ')
        vocab[line_info[0]] = line_info[3]

    for line in vocab_neg:
        line_info = line.split(', ')
        vocab[line_info[0]] = line_info[3]

    list_entities_vocab_keys = list(vocab.keys())
    contexts = open(os.path.join(directory_path, contexts_file), 'w')

    with open(os.path.join(directory_path, sentences_file), 'r') as corpus_sentences:
        firstNlines = corpus_sentences.readlines()
    cnt = 0
    for line in tqdm(firstNlines):
        line_tok = spacy_tokenizer(line, True)
        if any(word in list_entities_vocab_keys for word in line_tok):
            cnt += 1
            contexts.write(line.strip() + '===' + ' '.join(line_tok))
            print(cnt, line.strip() + '===' + ' '.join(line_tok))

    for file in [vocab_neg, vocab_pos, contexts, corpus_sentences]:
        file.close()


def search_thematic_contexts():
    """
    поиск тематических контекстов
    """
    with open(os.path.join('/home/anton/data/ABSA/contexts/txt', 'contexts_for_labeled_entities_2'),
              'r') as corpus_sentences:
        firstNlines = corpus_sentences.readlines()[:5]
    # nlp = spacy.load('')
    for line in firstNlines:
        print(spacy_tokenizer(line, True, nlp))


def main():
    search_thematic_contexts()


if __name__ == '__main__':
    main()
