from parsing_corpus import *
import pandas as pd
from tqdm import tqdm
import os
from deeppavlov import configs, build_model
import warnings

warnings.filterwarnings("ignore")
"""
Я думала, что результат возможно улучшить тем, что в нейтральные контексты набрать именно контексты для банков,
т.е. взять тот же список банков, и если вокруг нет явных тональностей, то считать контекст нейтральным.

применяем весь русентилекс
/home/anton/data/ABSA/contexts/txtRuSentiLex2017_revised.txt

файлы с предложениями:
/home/anton/data/ABSA/contexts/txt/contexts_for_labeled_entities_2
/home/anton/data/ABSA/contexts/txt/contexts_for_labeled_entities_3
"""


def search_thematic_contexts(banks_or_operators='banks'):
    """
    поиск нейтральных тематических контекстов (банки/операторы)
    """
    if banks_or_operators == 'banks':
        theme_words = ['банк']
    elif banks_or_operators == 'telecom':
        theme_words = ['мтс', 'билайн', 'мегафон']
    else:
        theme_words = []

    contexts = []
    tonal_vocab = []
    with open('/home/anton/data/ABSA/contexts/txt/RuSentiLex2017_revised.txt', 'r') as f:
        for line in f.readlines():
            if len(line.strip().split(', ')[0].split()) == 1:
                tonal_vocab.append(line.strip().split(', ')[0])
    tonal_vocab = set(tonal_vocab)

    with open(os.path.join('/home/anton/data/ABSA/contexts/txt', 'contexts_for_labeled_entities_2'),
              'r') as corpus_sentences:
        firstNlines = corpus_sentences.readlines()[2900000:3300000]

    nlp = spacy.load('/home/anton/PycharmProjects/spacy-ru/ru2e')
    ner_model = build_model(configs.ner.ner_rus_bert, download=True)

    for line in tqdm(firstNlines):
        line_tok = spacy_tokenizer(line.strip(), True, nlp)
        try:
            line_tok_after_ner = ner_model([line])
            # if (not set(line_tok).intersection(tonal_vocab)) \
            #         and 'B-ORG' in line_tok_after_ner[1][0] \
            #         and any(word in theme_words for word in line.lower()) \
            #         and len(line_tok) < 200:
            if (not set(line_tok).intersection(tonal_vocab)) \
                    and ('мтс' in line.lower() or 'мегафон' in line.lower() or 'билайн' in line.lower()) \
                    and len(line_tok) < 200:
                contexts.append(line + '===' + ' '.join(line_tok))
        except RuntimeError:
            continue

    with open('neutral_telecom_contexts_9.txt', 'w') as f:
        for line in contexts:
            f.write(line + '\n')


def txt2csv(file_from, file_to):
    """
    преобразование формата txt -> csv
    """
    with open(file_from, 'r') as f:
        contexts = f.readlines()
    contexts_cleaned = [context.split('===')[0].strip() for context in contexts if context[0] != '=']
    df = pd.DataFrame(data={'sentence': contexts_cleaned})
    df.to_csv(file_to, index=False, sep='\t')


def clean_banks_contexts(file_name, tokenization=False, tf_idf=False, find_entity=False, mask_entity=False):
    """
    чистка банковских контекстов и их подготовка к подаче в BERT
    """
    contexts = pd.read_csv(file_name, sep='\t')
    if tokenization:
        nlp = spacy.load('/home/anton/PycharmProjects/spacy-ru/ru2e')
        contexts['text_tok'] = contexts['text'].apply(lambda x: ' '.join(spacy_tokenizer(x, False, nlp)))

    if tf_idf:
        corpus = list(contexts['text_tok'])
        vectorizer = TfidfVectorizer(min_df=0.002, use_idf=True, ngram_range=(1, 1))
        X = vectorizer.fit_transform(corpus)
        X = cosine_similarity(X)
        pairs = np.argwhere(X > 0.8).T
        diag = pairs[0] != pairs[1]
        pairs = pairs.T[diag]
        numbers = np.unique(np.max(pairs, axis=1))
        contexts = contexts.drop(index=contexts.iloc[numbers].index)

    if find_entity:
        contexts['has_bank'] = contexts['text_tok'].apply(lambda x: x.count('банк'))
        contexts = contexts[contexts['has_bank'] == 1]

    if mask_entity:
        contexts['text_tok'] = contexts['text_tok'].apply(lambda x: re.sub('\s*\S*банк\S*\s*', ' MASK ', x))

    contexts.to_csv(file_name[:-4] + '_cleaned.csv', index=False, sep='\t')


def clean_telecom_contexts(file_name, tokenization=False, tf_idf=False, find_entity=False, mask_entity=False):
    """
    чистка банковских контекстов и их подготовка к подаче в BERT
    """
    contexts = pd.read_csv(file_name, sep='\t')
    if tokenization:
        nlp = spacy.load('/home/anton/PycharmProjects/spacy-ru/ru2e')
        contexts['text_tok'] = contexts['text'].apply(lambda x: ' '.join(spacy_tokenizer(x, False, nlp)))

    if tf_idf:
        corpus = list(contexts['text_tok'])
        vectorizer = TfidfVectorizer(min_df=0.002, use_idf=True, ngram_range=(1, 1))
        X = vectorizer.fit_transform(corpus)
        X = cosine_similarity(X)
        pairs = np.argwhere(X > 0.8).T
        diag = pairs[0] != pairs[1]
        pairs = pairs.T[diag]
        numbers = np.unique(np.max(pairs, axis=1))
        contexts = contexts.drop(index=contexts.iloc[numbers].index)

    if find_entity:
        contexts['has_bank'] = contexts['text_tok'].apply(lambda x: x.count('банк'))
        contexts = contexts[contexts['has_bank'] == 1]

    if mask_entity:
        contexts['text_tok'] = contexts['text_tok'].apply(lambda x: re.sub('\s*\S*банк\S*\s*', ' MASK ', x))

    contexts.to_csv(file_name[:-4] + '_cleaned.csv', index=False, sep='\t')


def main():
    # search_thematic_contexts(banks_or_operators='telecom')
    # txt2csv('neutral_telecom_contexts_9.txt', 'neutral_telecom_contexts_9.csv')
    # clean_banks_contexts('neutral_banks_contexts.csv', tokenization=True, tf_idf=True, find_entity=True,
    #                     mask_entity=True)
    clean_telecom_contexts('neutral_telecom_contexts.csv', tokenization=False, tf_idf=False, find_entity=False,
                           mask_entity=False)


if __name__ == '__main__':
    main()
