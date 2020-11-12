from parsing_corpus import *
import pandas as pd
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
        firstNlines = corpus_sentences.readlines()[700000:1300000]

    nlp = spacy.load('/home/anton/PycharmProjects/spacy-ru/ru2e')
    ner_model = build_model(configs.ner.ner_rus_bert, download=True)

    for line in tqdm(firstNlines):
        line_tok = spacy_tokenizer(line.strip(), True, nlp)
        try:
            line_tok_after_ner = ner_model([line])
            if (not set(line_tok).intersection(tonal_vocab)) \
                    and 'B-ORG' in line_tok_after_ner[1][0] \
                    and 'банк' in line.lower() \
                    and len(line_tok) < 200:
                contexts.append(line + '===' + ' '.join(line_tok))
        except RuntimeError:
            continue

    with open('neutral_bank_contexts_700_1300.txt', 'w') as f:
        for line in contexts:
            f.write(line + '\n')


def txt2csv():
    """
    преобразование формата
    """
    with open('neutral_bank_contexts_700_1300.txt', 'r') as f:
        contexts = f.readlines()
    contexts_cleaned = [context.split('===')[0].strip() for context in contexts if context[0] != '=']
    df = pd.DataFrame(data={'sentence': contexts_cleaned})
    df.to_csv('neutral_banks_contexts_1300.csv', index=False, sep='\t')


def main():
    # search_thematic_contexts()
    txt2csv()


if __name__ == '__main__':
    main()
