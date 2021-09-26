import gensim
from gensim.models import Word2Vec

from parsing_corpus import *
import pandas as pd
from tqdm import tqdm
import os
from deeppavlov import configs, build_model
import warnings

warnings.filterwarnings("ignore")


def search_thematic_contexts(banks_or_operators='banks'):
    """
    поиск нейтральных тематических контекстов в передаваемом предложении
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
    чистка банковских контекстов и их подготовка к подаче в модель BERT
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
    чистка банковских контекстов и их подготовка к подаче в модель BERT
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
        pairs = np.argwhere(X > 0.95).T
        diag = pairs[0] != pairs[1]
        pairs = pairs.T[diag]
        numbers = np.unique(np.max(pairs, axis=1))
        contexts = contexts.drop(index=contexts.iloc[numbers].index)

    if find_entity:
        contexts['has_bank'] = contexts['text_tok'].apply(lambda x: x.count('банк'))
        contexts = contexts[contexts['has_bank'] == 1]

    if mask_entity:
        contexts['texts_tok'] = contexts.apply(lambda row: row.texts_tok.replace(row.entity, 'MASK'), axis=1)

    contexts.to_csv(file_name[:-4] + '_cleaned.csv', index=False, sep='\t')


def multiply_sentences_with_several_entities(file_name):
    """
    для предложений с несколькими объектами дублируем предложения, маскируя все объекты поочередно
    """
    contexts = pd.read_csv(file_name, sep='\t')
    texts = []
    texts_tok = []
    entities = []

    for i in range(len(contexts)):
        if contexts.iloc[i]['mts_in'] == 1:
            texts.append(contexts.iloc[i]['text'])
            texts_tok.append(contexts.iloc[i]['text_tok'])
            entities.append('мтс')

        if contexts.iloc[i]['beeline_in'] == 1:
            texts.append(contexts.iloc[i]['text'])
            texts_tok.append(contexts.iloc[i]['text_tok'])
            entities.append('билайн')

        if contexts.iloc[i]['megafon_in'] == 1:
            texts.append(contexts.iloc[i]['text'])
            texts_tok.append(contexts.iloc[i]['text_tok'])
            entities.append('мегафон')

    contexts_multiplied = pd.DataFrame({'text': texts, 'texts_tok': texts_tok, 'entity': entities})
    contexts_multiplied.to_csv('neutral_telecom_multiplied.csv', index=False, sep='\t')


def calculate_delta():
    """
    вычисление прироста по метрикам после предобучения на собранной выборке тематических контекстов
    """
    operators_vanilla = np.array([[80.47, 72.59, 80.22, 66.95, 69.46],
                                  [82.28, 74.06, 81.24, 69.53, 71.76],
                                  [81.28, 73.34, 81.63, 65.82, 68.03]])

    operators_pretrained = np.array([[80.32, 71.34, 81.39, 66.44, 69.24],
                                     [83.79, 76.17, 82.41, 71.09, 72.97],
                                     [82.72, 75.08, 82.57, 67.71, 70.28]])

    delta = np.zeros(5)
    for i in range(len(delta)):
        delta[i] = np.mean([(operators_pretrained[0][i] - operators_vanilla[0][i]) / operators_vanilla[0][i] * 100,
                            (operators_pretrained[1][i] - operators_vanilla[1][i]) / operators_vanilla[1][i] * 100,
                            (operators_pretrained[2][i] - operators_vanilla[2][i]) / operators_vanilla[2][i] * 100])
    print(delta)


def main():
    search_thematic_contexts(banks_or_operators='telecom')
    txt2csv('neutral_telecom_contexts_9.txt', 'neutral_telecom_contexts_9.csv')
    clean_banks_contexts('neutral_banks_contexts.csv', tokenization=True, tf_idf=True, find_entity=True,
                         mask_entity=True)
    clean_telecom_contexts('neutral_telecom_multiplied.csv', tokenization=False, tf_idf=False, find_entity=False,
                           mask_entity=True)
    multiply_sentences_with_several_entities('neutral_telecom_contexts_cleaned.csv')
    calculate_delta()


if __name__ == '__main__':
    main()
