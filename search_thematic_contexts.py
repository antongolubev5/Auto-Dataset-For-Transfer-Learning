from parsing_corpus import *
from tqdm import tqdm
import os
from deeppavlov import configs, build_model

"""
Я думала, что результат возможно улучшить тем, что в нейтральные контексты набрать именно контексты для банков,
т.е. взять тот же список банков, и если вокруг нет явных тональностей, то считать контекст нейтральным.

применяем весь русентилекс
/home/anton/data/ABSA/contexts/txtRuSentiLex2017_revised.txt

файлы с предложениями:
/home/anton/data/ABSA/contexts/txt/contexts_for_labeled_entities_2
/home/anton/data/ABSA/contexts/txt/contexts_for_labeled_entities_3
"""


def search_thematic_contexts():
    """
    поиск тематических контекстов (нейтральные банки)
    """
    contexts = []
    tonal_vocab = []
    with open('/home/anton/data/ABSA/contexts/txt/RuSentiLex2017_revised.txt', 'r') as f:
        for line in f.readlines():
            if len(line.strip().split(', ')[0].split()) == 1:
                tonal_vocab.append(line.strip().split(', ')[0])
    tonal_vocab = set(tonal_vocab)

    with open(os.path.join('/home/anton/data/ABSA/contexts/txt', 'contexts_for_labeled_entities_2'),
              'r') as corpus_sentences:
        firstNlines = corpus_sentences.readlines()[:30000]

    nlp = spacy.load('/home/anton/PycharmProjects/spacy-ru/ru2e')
    ner_model = build_model(configs.ner.ner_rus_bert, download=True)

    for line in tqdm(firstNlines):
        line_tok = spacy_tokenizer(line.strip(), True, nlp)
        line_tok_after_ner = ner_model([line])
        if (not set(line_tok).intersection(tonal_vocab)) \
                and 'B-ORG' in line_tok_after_ner[1][0] \
                and 'банк' in line.lower()\
                and len(line_tok) < 400:
            contexts.append(line + '===' + ' '.join(line_tok))

    with open('neutral_bank_contexts.txt', 'w') as f:
        for line in contexts:
            f.write(line + '\n')


def main():
    search_thematic_contexts()


if __name__ == '__main__':
    main()
