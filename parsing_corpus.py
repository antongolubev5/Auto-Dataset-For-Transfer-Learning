from tqdm import tqdm
import re
import os
import spacy
import numpy as np
import xml.etree.ElementTree as ET
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
import time
from pymystem3 import Mystem
from string import punctuation
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy2
import rutokenizer
from deeppavlov import configs, build_model
from nltk.tokenize import TweetTokenizer
from razdel import tokenize

spacy.prefer_gpu()


def cosine_similarity_own(a, b):
    return dot(a, b) / (norm(a) * norm(b))


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


def pymorphy_tokenizer(text, tokenizer, morph, lemmatize: bool):
    """
    https://pymorphy2.readthedocs.io/en/latest/user/guide.html#id2
    """
    tokens = tokenizer.tokenize(text)
    punc_list = ' –!"@#$%^&*()*+_,.\:;<>=?[]{}|~`/«»—' + '0123456789'
    if lemmatize:
        return [morph.parse(token)[0].normal_form for token in tokens if
                # token != " " and token.strip() not in set(punctuation + punc_list)]
                token != " "]
    else:
        # return [token for token in tokens if token != " " and token.strip() not in set(punctuation + punc_list)]
        return [token for token in tokens if token != " "]


def spacy_tokenizer(text, lemm: bool, nlp):
    """токенизатор на основе библиотеки spacy, учитывающий особенности русского языка
    spacy_russian_tokenizer --- токенизация
    spacy_ru2 --- лемматизация (как параметр)
    стоп слова?
    не всегда правильно работает (часто плохие леммы), надо разбираться -
    в репо написано использовать ru2e, но не работает
    """

    # nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    # # nlp = spacy.load('ru2', disable=['tagger', 'parser', 'NER'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    # doc = nlp(text)

    # ВЗЯТЬ РУССКИЙ ТОКЕНИЗАТОР https://github.com/antongolubev5/spacy_russian_tokenizer
    # nlp = Russian()
    russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    doc = nlp(text)

    text = [token.lemma_ if lemm else token for token in doc]

    punc_list = set(' –!"@#$%^&*()*+_,.\:;<>=?[]{}|~`/«»—' + '0123456789')
    output = []

    for i in range(len(text)):
        text[i] = re.sub(" +", " ", str(text[i]))
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

    for line in tqdm(firstNlines):
        line_tok = spacy_tokenizer(line, True)
        if any(word in list_entities_vocab_keys for word in line_tok):
            cnt += 1
            contexts.write(line.strip() + '===' + ' '.join(line_tok))
            print(cnt, line.strip() + '===' + ' '.join(line_tok))

    for file in [vocab_neg, vocab_pos, contexts, corpus_sentences]:
        file.close()


def searching_contexts_csv(directory_path, entities_vocab, sentences_file, contexts_file, sentence_volume):
    """
    поиск тональных контекстов cреди предложений корпуса
    :param directory_path:
    :param entities_vocab: название csv файла c тональными словами
    :param sentences_file: csv файл с предложениями из корпуса
    :param contexts_file: csv файл, в который будут записаны контексты
    :param sentence_volume: сколько предложений из корпуса рассматривать [0:vol]
    """

    # entities_vocab = pd.read_csv(os.path.join(directory_path, entities_vocab), sep='\t')
    entities_vocab = entities_vocab.keys()
    # corpus_sentences = pd.read_csv(os.path.join(directory_path, sentences_file), sep='\t')
    corpus_sentences = sentences_file
    contexts = pd.DataFrame(columns=['context', 'context_tokens'])

    for i in tqdm(range(len(corpus_sentences))):
        line_tok = spacy_tokenizer(corpus_sentences.iloc[i][0], True)
        if any(word in entities_vocab for word in line_tok):
            cnt += 1
            contexts = contexts.append(
                pd.Series([corpus_sentences.iloc[i][0].strip(), ' '.join(line_tok)], index=contexts.columns),
                ignore_index=True)
            print(cnt, corpus_sentences.iloc[i][0].strip() + '===' + ' '.join(line_tok))

    contexts.to_csv(os.path.join(directory_path, contexts_file), index=False, sep='\t')


def divide_contexts(directory_path, entities_vocab, contexts, positive_contexts, negative_contexts):
    """
    разделение имеющихся контекстов на 2 позитивные и негативные
    смешанные контексты выбрасываются
    :param directory_path: путь
    :param entities_vocab: словарь тональных сущностей
    :param contexts: имя файла, в котором лежат все контексты
    :param positive_contexts: имя файла, в который записываем + контексты
    :param negative_contexts: имя файла, в который записываем - контексты
    :return:
    """

    positive_contexts = open(os.path.join(directory_path, positive_contexts), 'w')
    negative_contexts = open(os.path.join(directory_path, negative_contexts), 'w')

    with open(os.path.join(directory_path, contexts), 'r') as contexts:
        contexts_lines = contexts.readlines()

    for line in tqdm(contexts_lines):
        line_text = line.split('===')[0]
        line_tok = line.split('===')[1].strip()
        flag, lst = check_tones(line_tok.split(" "), entities_vocab)

        if flag == 1:
            positive_contexts.write(line_text + '===' + line_tok + '===' + ' '.join(lst) + '===' + '1' + '\n')
        elif flag == -1:
            negative_contexts.write(line_text + '===' + line_tok + '===' + ' '.join(lst) + '===' + '-1' + '\n')
        else:
            print(line_text)

    for file in [contexts, positive_contexts, negative_contexts]:
        file.close()


def divide_contexts_csv(contexts_all, entities_vocab):
    """
    разделение имеющихся контекстов на 2 позитивные и негативные
    смешанные контексты в отдельный df
    """
    cnt = 1
    contexts_labeled = pd.DataFrame(columns=['text', 'text_tok', 'tonal_word', 'label'])
    for i in range(len(contexts_all)):
        print(cnt, '/', len(contexts_all), ' = ', round(cnt / len(contexts_all) * 100, 2), '%...')
        line_text = contexts_all.iloc[i]['context']
        line_tok = contexts_all.iloc[i]['context_tokens']
        flag, lst = check_tones(line_tok.split(" "), entities_vocab)

        if flag == 1:
            contexts_labeled = contexts_labeled.append(
                pd.Series([line_text, line_tok, ' '.join(lst), 1], index=contexts_labeled.columns),
                ignore_index=True)
        elif flag == -1:
            contexts_labeled = contexts_labeled.append(
                pd.Series([line_text, line_tok, ' '.join(lst), -1], index=contexts_labeled.columns),
                ignore_index=True)
        elif flag == 0:  # mixed
            contexts_labeled = contexts_labeled.append(
                pd.Series([line_text, line_tok, ' '.join(lst), 0], index=contexts_labeled.columns),
                ignore_index=True)
        else:
            print(line_text)
        cnt += 1
    return contexts_labeled


def edit_csv_data(entities_vocab, contexts_all):
    """
    чистка выборки:
    удаление коротких контекстов (<10 слов)
    удаление контекстов, в которых оценочное слово расположено в кавычках (попытка)
    """
    text = []
    text_tok = []
    tonal_word = []
    label = []
    sent_type = []
    for i in tqdm(range(len(contexts_all))):
        if any(word in entities_vocab for word in contexts_all.iloc[i]['text_tok'].split()) and len(
                contexts_all.iloc[i]['text_tok'].split()) > 10 \
                and not tonal_word_in_quotes(contexts_all.iloc[i]['text'], contexts_all.iloc[i]['tonal_word']):
            text.append(contexts_all.iloc[i]['text'])
            text_tok.append(contexts_all.iloc[i]['text_tok'])
            tonal_word.append(contexts_all.iloc[i]['tonal_word'])
            label.append(contexts_all.iloc[i]['label'])
            sent_type.append(contexts_all.iloc[i]['sent_type'])
    data = {'text': text, 'text_tok': text_tok, 'tonal_word': tonal_word, 'label': label, 'sent_type': sent_type}
    return pd.DataFrame.from_dict(data)


def drop_multi_entities_sentences(contexts_all):
    """
    удаление из выборки предложений, содержащих несколько сущностей
    """
    text = []
    text_tok = []
    tonal_word = []
    label = []
    sent_type = []
    for i in tqdm(range(len(contexts_all))):
        if len(contexts_all.iloc[i]['tonal_word'].split()) == 1:
            text.append(contexts_all.iloc[i]['text'])
            text_tok.append(contexts_all.iloc[i]['text_tok'])
            tonal_word.append(contexts_all.iloc[i]['tonal_word'])
            label.append(contexts_all.iloc[i]['label'])
            sent_type.append(contexts_all.iloc[i]['sent_type'])
    data = {'text': text, 'text_tok': text_tok, 'tonal_word': tonal_word, 'label': label, 'sent_type': sent_type}
    return pd.DataFrame.from_dict(data)


def plot_words_distribution(df, sentiment, volume, save: bool):
    """
    построение гистограммы тональных слов из контекстов
    sentiment: 1 if pos else neg
    volume: сколько слов рисовать
    если volume==-1, гистограмма по всевозможным словам
    """
    if volume == -1:
        volume = len(set(df[df['label'] == sentiment]['tonal_word']))
    df = df[df['label'] == sentiment]['tonal_word']
    cntr = Counter(df).most_common(volume)
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 15))
    sns.set_color_codes("dark")
    fig = sns.barplot(x=[value for key, value in cntr], y=[key for key, value in cntr], label="Total",
                      color="b")
    bias = 10 if sentiment == 1 else 5
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + bias,
                p.get_y() + p.get_height() / 2. + 0.5,
                '{:.0f}'.format(width),
                ha="center")
    # ax.legend(ncol=2, loc="lower right", frameon=True)
    tone = 'положительных' if sentiment == 1 else 'отрицательных'
    ax.set(ylabel="", xlabel="Распределение " + tone + ' слов по выборке')
    sentiment = 'negative' if sentiment == -1 else 'positive'
    figure = fig.get_figure()
    if save:
        figure.savefig('/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/' + sentiment + '_distribution',
                       dpi=600, bbox_inches='tight')
    plt.show()


def check_sentiment_of_sentence(text: list, vocab):
    """
    определение тональности предложения
    вывод тональных слов
    1 = good, -1 = bad, 0 = mixed, -10 = trash
    """

    bad = False
    good = False
    bad_words = []
    good_words = []

    for word in text:
        if word in vocab.keys():
            if vocab[word] == 'positive':
                good = True
                if word not in good_words:
                    good_words.append(word)
            else:
                bad = True
                if word not in bad_words:
                    bad_words.append(word)
            if good and bad:
                return 0, good_words + bad_words

    if good:
        return 1, good_words

    if bad:
        return -1, bad_words

    return -10, []


def tonal_word_in_quotes(text, word):
    """
    проверяем, внутри ли кавычек тональное слово, если да, возможна смена тональности/значения и контекст не нужен
    """
    text = text.lower()
    pos = 0
    if word in text:
        pos = text.index(word)
    elif word[:-1] in text:
        pos = text.index(word[:-1])
    elif word[:-2] in text:
        pos = text.index(word[:-2])
    elif word[:-3] in text:
        pos = text.index(word[:-3])

    return '«' in text[max(pos - 15, 0):pos] or '»' in text[pos:min(pos + 15, len(text) - 1)] or '\"' in text[max(
        pos - 15, 0):pos] or '\"' in text[pos:min(pos + 15, len(text) - 1)]


def check_sentiments(words, entities_vocab):
    """
    классификация предложения на группы:
    """
    neg_words = set([key for key, value in entities_vocab.items() if value == 'negative'])
    pos_words = set([key for key, value in entities_vocab.items() if value == 'positive'])
    words = set(words)
    if len(words.intersection(neg_words)) == 1 and len(words.intersection(pos_words)) == 0:
        return 'neg'
    elif len(words.intersection(neg_words)) == 0 and len(words.intersection(pos_words)) == 1:
        return 'pos'
    elif len(words.intersection(neg_words)) == 1 and len(words.intersection(pos_words)) == 1:
        return 'posneg'
    elif len(words.intersection(neg_words)) == 2:
        return 'negneg'
    elif len(words.intersection(pos_words)) == 2:
        return 'pospos'


def search_multi_entities_sentences(contexts_all, entities_vocab):
    """
    поиск мультитональных (2 тональности) предложений в выборке
    """
    contexts_all['tonal_words_cnt'] = contexts_all['tonal_word'].apply(lambda x: len(x.split()))
    contexts_all = contexts_all.loc[contexts_all['tonal_words_cnt'] == 2]
    contexts_all['sent_type'] = contexts_all['tonal_word'].apply(lambda x: check_sentiments(x, entities_vocab))
    return contexts_all


def vocab_from_file(directory_path, file_names):
    vocab = dict()
    for file_name in file_names:
        with open(os.path.join(directory_path, file_name), 'r') as f:
            for line in f:
                vocab[line.strip().split(', ')[0]] = line.strip().split(', ')[3]
    return vocab


def create_balanced_samples(contexts_all, volumes, volume_neutral, top_words, drop_volume):
    """
    из обычной выборки делаем сбалансированную:
    drop_volume = [...neg, ...pos] сколько первых слов отсекаем для повторного извлечения в случае нехватки
    volumes = [volume_neg, volume_pos]
    top_words = [top_words_neg, top_words_pos]
    len(each_word) =  volume / 25 (берем 25 наиболее популярных положительных и отрицательных слов)
    """
    contexts_balanced = pd.DataFrame(columns=contexts_all.columns)
    for label, volume, top_word, drop_first in zip([-1, 1], volumes, top_words, drop_volume):
        word_volume = volume // top_word
        cntr = Counter(contexts_all[contexts_all['label'] == label]['tonal_word']).most_common(top_word)
        words_to_take_later = [key for key, value in cntr][drop_first:]
        for key, value in cntr:
            word_batch = contexts_all[contexts_all['tonal_word'] == key][:word_volume]
            contexts_balanced = contexts_balanced.append(word_batch)
            contexts_all = contexts_all.drop(word_batch.index)
        real_len = len(contexts_balanced[contexts_balanced['label'] == label])
        if real_len != volume:
            contexts_balanced = contexts_balanced.append(
                contexts_all[contexts_all['tonal_word'].isin(words_to_take_later)].sample(n=volume - real_len,
                                                                                          random_state=2))
    contexts_balanced = contexts_balanced.append(
        contexts_all[contexts_all['label'] == 0].sample(n=volume_neutral, random_state=2))
    return contexts_balanced.drop_duplicates()


def drop_same_sentences(contexts_all):
    """
    почему-то разные типы кавычек и дефисов не почистились на этапе предобработки
    удаление одинаковых предложений с разными типами кавычек
    """
    cleaned_texts = contexts_all['text'].apply(lambda x: x.replace('«', '\"')).apply(
        lambda x: x.replace('»', '\"')).apply(lambda x: x.replace('-', '-'))
    contexts_all['text'] = cleaned_texts
    return contexts_all.drop_duplicates()


def from_raw_sentences_to_dataset(raw_data, entities_vocab):
    """
    pipeline создания сбалансированной выборки из сырых контекстов
    не более двух сущностей в одном контексте
    не менее 10 слов в контексте
    return 1-сущностные контексты, 2-сущностные контексты
    """
    text = []
    text_tok = []
    tonal_word = []
    label = []
    sent_type = []

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()
    morph = pymorphy2.MorphAnalyzer()

    for i in tqdm(range(len(raw_data))):
        # context_text = raw_data.iloc[i]['sentence']
        context_text = raw_data[i].strip()
        context_tok = pymorphy_tokenizer(context_text, tokenizer, morph)
        if any(word in entities_vocab for word in context_tok) and len(context_tok) > 10 and len(context_tok) < 40:
            flag, sentiment_words = check_sentiment_of_sentence(context_tok, entities_vocab)
            quotes = False
            for sentiment_word in sentiment_words:
                if tonal_word_in_quotes(context_text, sentiment_word):
                    quotes = True
            if len(sentiment_words) <= 2 and not quotes:
                if flag == 1:
                    text.append(context_text)
                    text_tok.append(' '.join(context_tok))
                    tonal_word.append(' '.join(sentiment_words))
                    label.append(1)
                    sent_type.append(check_sentiments(context_tok, entities_vocab))
                elif flag == -1:
                    text.append(context_text)
                    text_tok.append(' '.join(context_tok))
                    tonal_word.append(' '.join(sentiment_words))
                    label.append(-1)
                    sent_type.append(check_sentiments(context_tok, entities_vocab))
                elif flag == 0:
                    text.append(context_text)
                    text_tok.append(' '.join(context_tok))
                    tonal_word.append(' '.join(sentiment_words))
                    label.append(0)
                    sent_type.append(check_sentiments(context_tok, entities_vocab))
                else:
                    print(context_text)

    data = {'text': text, 'text_tok': text_tok, 'tonal_word': tonal_word, 'label': label, 'sent_type': sent_type}
    contexts = pd.DataFrame.from_dict(data)
    multi_contexts = contexts[~contexts['sent_type'].isin(['pos', 'neg'])]
    single_contexts = contexts[contexts['sent_type'].isin(['pos', 'neg'])]
    return single_contexts, multi_contexts


def drop_similar_contexts_tfidf(contexts_all):
    """
    в выборке много перепечатанных и очень похожих контекстов
    нужно прорядить ее одним из методов
    """
    set_tonal_words = set(contexts_all['tonal_word'])
    for tonal_word in tqdm(set_tonal_words):
        context_word = contexts_all[contexts_all['tonal_word'] == tonal_word]
        corpus = list(context_word['text_tok'])
        vectorizer = TfidfVectorizer(min_df=0.002, use_idf=True, ngram_range=(1, 1))
        X = vectorizer.fit_transform(corpus)
        X = cosine_similarity(X)
        pairs = np.argwhere(X > 0.5).T
        diag = pairs[0] != pairs[1]
        pairs = pairs.T[diag]
        numbers = np.unique(np.max(pairs, axis=1))
        contexts_all = contexts_all.drop(index=context_word.iloc[numbers].index)
    return contexts_all


def extract_neutral_contexts(directory_path):
    """
    извлечение нейтральных контекстов
    Обрабатываем тексты системой извлечения именованных сущностей (deep pavlov), Пусть нас пока интересуют только персоны.
    Смотрим на заголовки текстов и ищем такие заголовки, где упоминается персона и нет никаких оценочных слов из оценочного словаря.
    Вообще никаких. Тогда можно предположить, что и в тексте тоже отношение к этой персоне нейтральное.
    Берем первое предложение, где есть эта персона из заголовка в качестве нейтрального
    (и здесь мы уже не обращаем внимание на наличие оценочных слов).
    """
    tonal_words = []

    with open(os.path.join(directory_path, 'RuSentiLex2017_revised.txt')) as f:
        for line in f:
            line = re.sub(r"[\"]", "", line).lower()
            words = line.strip().split(', ')
            tonal_words.append(words[0])
            tonal_words.append(words[2])
            if len(words) > 5:
                tonal_words += words[5:]
    tonal_words = set(tonal_words)

    ner_model = build_model(configs.ner.ner_rus_bert, download=True)
    nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()
    morph = pymorphy2.MorphAnalyzer()

    bodies = []
    bodies_tok = []
    persons = []

    for month in tqdm(os.listdir(os.path.join(directory_path, 'Rambler_source'))):
        if month == '201101':
            for day in tqdm(os.listdir(os.path.join(directory_path, 'Rambler_source', month))):
                for utf in tqdm(os.listdir(os.path.join(directory_path, 'Rambler_source', month, day))):
                    if os.path.exists(os.path.join(directory_path, 'Rambler_source', month, day, utf, 'items')):
                        for xml_file in os.listdir(
                                os.path.join(directory_path, 'Rambler_source', month, day, utf, 'items')):
                            tree = ET.parse(os.path.join(
                                os.path.join(directory_path, 'Rambler_source', month, day, utf, 'items', xml_file)))
                            title = tree.getroot()[0].text
                            title_tok = pymorphy_tokenizer(title, tokenizer, morph, lemmatize=False)
                            if len(title_tok) < 250:
                                title_after_ner = ner_model([title])
                                if title_after_ner[1][0].count('B-ORG') == 1 and not set(title_tok).intersection(
                                        tonal_words):
                                    person_in_title_start_idx = title_after_ner[1][0].index('B-ORG')
                                    person_in_title_idxs = [person_in_title_start_idx]
                                    for i in range(person_in_title_start_idx + 1, len(title_after_ner[1][0])):
                                        if title_after_ner[1][0][i] == 'I-PER':
                                            person_in_title_idxs.append(i)
                                        else:
                                            break
                                    person_in_title_full = [title_after_ner[0][0][i] for i in person_in_title_idxs]
                                    person_in_title_full = ' '.join(person_in_title_full)
                                    if os.path.exists(
                                            os.path.join(directory_path, 'Rambler_source', month, day, utf,
                                                         'texts', xml_file[:-4] + '.txt')):
                                        with open(os.path.join(directory_path, 'Rambler_source', month, day, utf,
                                                               'texts',
                                                               xml_file[:-4] + '.txt')) as f_body:
                                            body = text2sentences(f_body.read(), nlp)
                                        for sentence in body:
                                            sentence_tok = pymorphy_tokenizer(sentence, tokenizer, morph,
                                                                              lemmatize=False)
                                            if all(x in sentence.lower() for x in
                                                   person_in_title_full.lower().split()) and len(
                                                sentence_tok) > 10 and len(sentence_tok) < 40:
                                                sentence_after_ner = ner_model([sentence])
                                                if 'B-ORG' in sentence_after_ner[1][0]:
                                                    person_in_body_start_idx = sentence_after_ner[1][0].index(
                                                        'B-ORG')
                                                    person_in_body_idxs = [person_in_body_start_idx]
                                                    for i in range(person_in_body_start_idx + 1,
                                                                   len(sentence_after_ner[1][0])):
                                                        if sentence_after_ner[1][0][i] == 'I-ORG':
                                                            person_in_body_idxs.append(i)
                                                        else:
                                                            break
                                                    person_in_body_full = [sentence_after_ner[0][0][i] for i in
                                                                           person_in_body_idxs]
                                                    sentence = sentence.replace(' '.join(person_in_body_full), 'MASK')
                                                    sentence_tok = pymorphy_tokenizer(sentence, tokenizer, morph,
                                                                                      lemmatize=False)
                                                    bodies.append(sentence)
                                                    bodies_tok.append(' '.join(sentence_tok).lower())
                                                    persons.append(' '.join(person_in_body_full).lower())
                                                    break

    data = {'person': persons, 'body': bodies, 'body_tok': bodies_tok}
    df = pd.DataFrame.from_dict(data)
    # df['label'] = df['body_tok'].apply(lambda x: len(set(x.split()).intersection(tonal_words)))
    # df = df[df['label'] == 0]
    # df = df.drop(['label'], axis=1)
    #
    # df['has_body'] = df['body_tok'].apply(lambda x: 'mask' not in x)
    # df = df[df['has_body'] == False]
    # df = df.drop(['has_body'], axis=1)
    #
    # corpus = list(df['body_tok'])
    # vectorizer = TfidfVectorizer(min_df=0.002, use_idf=True, ngram_range=(1, 1))
    # X = vectorizer.fit_transform(corpus)
    # X = cosine_similarity(X)
    # pairs = np.argwhere(X > 0.5).T
    # diag = pairs[0] != pairs[1]
    # pairs = pairs.T[diag]
    # numbers = np.unique(np.max(pairs, axis=1))
    # df = df.drop(index=df.iloc[numbers].index)

    return df


def extract_neutral_banks_telecoms_contexts(directory_path):
    """
    извлечение нейтральных контекстов про банки/операторы для расширения тренировочной выборки sentirueval2016
    """
    tonal_words = []
    with open(os.path.join(directory_path, 'RuSentiLex2017_revised.txt')) as f:
        for line in f:
            line = re.sub(r"[\"]", "", line).lower()
            words = line.strip().split(', ')
            tonal_words.append(words[0])
            tonal_words.append(words[2])
            if len(words) > 5:
                tonal_words += words[5:]
    tonal_words = set(tonal_words)

    # ner_model = build_model(configs.ner.ner_rus_bert, download=True)
    nlp = spacy.load('/media/anton/ssd2/data/datasets/spacy-ru/ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()
    morph = pymorphy2.MorphAnalyzer()

    bodies = []
    bodies_tok = []
    persons = []

    for month in tqdm(os.listdir(os.path.join(directory_path, 'Rambler_source'))):
        # if month == '201101':
        for day in tqdm(os.listdir(os.path.join(directory_path, 'Rambler_source', month))):
            # if day == '20110125':
            for utf in tqdm(os.listdir(os.path.join(directory_path, 'Rambler_source', month, day))):
                if os.path.exists(os.path.join(directory_path, 'Rambler_source', month, day, utf, 'items')):
                    for xml_file in os.listdir(
                            os.path.join(directory_path, 'Rambler_source', month, day, utf, 'items')):
                        tree = ET.parse(os.path.join(
                            os.path.join(directory_path, 'Rambler_source', month, day, utf, 'items', xml_file)))
                        title = tree.getroot()[0].text
                        title_tok, entity = tweet_tokenizer(title, 'banks')
                        title_tok_lemmatized = pymorphy_tokenizer(title, tokenizer, morph, lemmatize=True)
                        if 7 < len(title_tok.split()) < 250 and entity is not None and not set(
                                title_tok_lemmatized).intersection(tonal_words):
                            bodies.append(title)
                            bodies_tok.append(title_tok)
                            persons.append(entity)

    data = {'person': persons, 'body': bodies, 'body_tok': bodies_tok}
    df = pd.DataFrame.from_dict(data)
    df['label'] = df['body_tok'].apply(lambda x: len(set(x.split()).intersection(tonal_words)))
    df = df[df['label'] == 0]
    df = df.drop(['label'], axis=1)

    # df['has_body'] = df['body_tok'].apply(lambda x: 'mask' not in x)
    # df = df[df['has_body'] == False]
    # df = df.drop(['has_body'], axis=1)

    corpus = list(df['body_tok'])
    vectorizer = TfidfVectorizer(min_df=0.002, use_idf=True, ngram_range=(1, 1))
    X = vectorizer.fit_transform(corpus)
    X = cosine_similarity(X)
    pairs = np.argwhere(X > 0.5).T
    diag = pairs[0] != pairs[1]
    pairs = pairs.T[diag]
    numbers = np.unique(np.max(pairs, axis=1))
    df = df.drop(index=df.iloc[numbers].index)

    return df


def tweet_tokenizer(text, task):
    text = text.lower()
    # for element in 'ё/,!':
    #     text = re.sub(element, ' ' + element + ' ', text)
    text = re.sub('&gt;', ' ', text)
    text = re.sub('&amp;quot', ' ', text)
    entity = None

    if task == 'banks':
        patterns = {'альф': ' альфабанк ',
                    # 'alfa': 'альфабанк ',
                    'bm_twit': ' банкмосквы ',
                    'атб': ' банкмосквы ',
                    'sberbank': ' сбербанк ',
                    # 'sber': 'сбербанк ',
                    'сбербанк': ' сбербанк ',
                    # 'сбер': 'сбербанк ',
                    'vtb': ' втб ',
                    'юникредит': ' юникредит ',
                    'бтв': ' втб ',
                    'втб': ' втб ',
                    'внешторгбанк': ' втб ',
                    'raif': ' райффайзенбанк ',
                    'райф': ' райффайзенбанк ',
                    'rshb': ' россельхозбанк ',
                    'рсхб': ' россельхозбанк ',
                    'россельхозб': ' россельхозбанк ',
                    'промсвязьбанк': ' промсвязьбанк ',
                    'gazprombank': ' газпромбанк ',
                    'газпромбанк': ' газпромбанк ',
                    # 'газпром': 'газпромбанк ',
                    # 'газром': 'газпромбанк ',
                    'уралсиб': ' уралсиб ',
                    'мдм': ' мдм ',
                    'бинбанк': ' бинбанк ',
                    'транскредитбанк': ' транскредитбанк ',
                    'инвестторгбанк': ' инвестторгбанк ',
                    }
        bank = {'альф': 'банк ',
                # 'alfa': 'bank ',
                'bm_twit': 'банк ',
                'атб': 'банк ',
                # 'sber': 'bank ',
                'sberbank': 'bank ',
                # 'sberbank cib': 'сбербанк',
                # 'сбер': 'банк ',
                'сбербанк': 'банк ',
                'юникредит': 'банк ',
                'vtb': 'bank ',
                'транскредитбанк': 'банк ',
                'промсвязьбанк': 'банк ',
                'бтв': 'банк ',
                'втб': 'банк ',
                'внешторгбанк': 'банк ',
                'raif': 'bank ',
                'райф': 'банк ',
                'rshb': 'bank ',
                'рсхб': 'банк ',
                'россельхозб': 'банк ',
                'gazprombank': 'bank ',
                'газпромбанк': 'банк ',
                # 'газпром': 'банк ',
                # 'газром': 'банк ',
                'уралсиб': 'банк ',
                'мдм': 'банк ',
                'бинбанк': 'банк ',
                'инвестторгбанк': ' банк ',
                }
        for pattern in patterns.keys():
            text = re.sub(r'[\s]{0,}' + pattern + '[\w]{0,}(([\s-]{0,}' + bank[pattern] +
                          '[^\s\.,!?-]{0,2})[\s\.,!?-]){0,}', ' ' + patterns[pattern], text)
        text = text.replace('банк москвы', 'банкмосквы ')
        text = text.replace('банка москвы', 'банкмосквы ')
        text = re.sub('втб\s+24', 'втб', text)
        text = text.replace('сбербанк россии', 'сбербанк ')
        for value in patterns.values():
            if value in text:
                entity = value
                break
    elif task == 'telecoms':
        patterns = {'билайн': ' билайн ',
                    'билаин': ' билайн ',
                    'биллайн': ' билайн ',
                    'beeline': ' билайн ',
                    'пчелайн': ' билайн ',
                    'вымпелком': ' билайн ',
                    'vimpelcom': ' билайн ',
                    'мегафон': 'мегафон ',
                    'мегафно': 'мегафон ',
                    'megafon': ' мегафон ',
                    'мтс': ' мтс ',
                    'mts': ' мтс ',
                    'ростел': ' ростелеком ',
                    'rostelecom': ' ростелеком ',
                    'теле2': ' теледва ',
                    'tele2': ' теледва ',
                    'skylink': ' скайлинк ',
                    }
        for pattern in patterns.keys():
            text = re.sub('[^\s]{0,}' + pattern + '[\w2]{0,}', patterns[pattern], text)
        for element in ['теле 2', 'теле-2', 'tele 2']:
            text = text.replace(element, ' теледва ')
        text = text.replace('мобильные телесистемы', ' мтс ')
        for value in patterns.values():
            if value in text:
                entity = value
                break
    else:
        return 'no such option'

    text = re.sub('rt', '', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http:/[^\w]))', '', text)
    text = re.sub('@[^\s]{0,}', '', text)
    text = re.sub('#[^\s]{0,}', '', text)
    text = re.sub('[^a-zA-Zа-яА-Я0-9():\-\,\.!?]+', ' ', text)
    text = re.sub('-банк[\w\s]', '', text)
    text = re.sub('\sбанк\s', '', text)
    text = list(tokenize(text))
    text = [_.text for _ in text]
    text = ' '.join(text)

    return text, entity


def create_semeval_dataset(directory_path, file_name):
    """
    приведение данных из соревнования semeval к виду текущей постановки задачи
    :param directory_path: корневая директория
    :param file_name: имя csv-файла
    """
    vocab = {'sberbank': 'сбербанк',
             'raiffeisen': 'райффайзенбанк',
             'vtb': 'втб',
             'rshb': 'россельхозбанк',
             'alfabank': 'альфабанк',
             'gazprom': 'газпромбанк',
             'bankmoskvy': 'банкмосквы',
             'beeline': 'билайн',
             'megafon': 'мегафон',
             'mts': 'мтс',
             'rostelecom': 'ростелеком',
             'skylink': 'скайлинк',
             'tele2': 'теледва',
             'komstar': 'комстар',
             }
    has_entity = []
    contexts_all = pd.read_csv(os.path.join(directory_path, file_name), sep='\t')
    contexts_all['rus_entity'] = contexts_all['entity'].apply(lambda x: vocab[x])
    contexts_all['text_tok'] = contexts_all['text'].apply(lambda x: tweet_tokenizer(x, 'telecoms'))

    for i in range(len(contexts_all)):
        has_entity.append(1 if contexts_all.iloc[i]['rus_entity'] in contexts_all.iloc[i]['text_tok'] else 0)
    contexts_all['has_entity'] = has_entity

    print(contexts_all['has_entity'].value_counts())
    contexts_all = contexts_all.sort_values(by=['has_entity'])
    contexts_all.to_csv(os.path.join(directory_path, file_name[:-4] + '_cleaned.csv'), index=False, sep='\t')


def simple_tokenizer(text):
    text = text.lower()
    text = re.sub('\(\[\]\)', '', text)
    text = re.sub('\d\d:\d\d:\d\d', '', text)
    text = re.sub('\d\d:\d\d', '', text)
    text = re.sub('\d\d\d\d-\d\d-\d\d', '', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http:/[^\w]))', '', text)
    text = re.sub('@[^\s]{0,}', '', text)
    text = re.sub('#[^\s]{0,}', '', text)
    text = re.sub('[^a-zA-Zа-яА-Я0-9():\-\,\.!?]+', ' ', text)
    text = tokenize(text)
    text = [_.text for _ in text]
    return ' '.join(text)


def main():
    start_time = time.time()

    directory_path = '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis'
    entities_vocab = vocab_from_file(directory_path, ['nouns_person_neg', 'nouns_person_pos'])

    df = pd.read_csv(os.path.join(directory_path, 'new.csv'), sep='\t')

    # contexts_task = pd.read_csv(os.path.join(directory_path, 'tkk_train_2016_cleaned.csv'), sep='\t')
    # contexts_all = pd.read_csv(os.path.join(directory_path, 'single_contexts.csv'), sep='\t')
    #
    # for label, num in contexts_task['label'].value_counts().items():
    #     contexts_task = contexts_task.append(contexts_all[contexts_all['label'] == label][:num // 3])
    #
    # contexts_task.to_csv(os.path.join(directory_path, 'task_and_weighted_contexts.csv'), sep='\t', index=False)

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))


if __name__ == '__main__':
    main()
    # TODO 3 статьи хабр + статья гарвард
    # TODO почитать про flair
    # TODO извлечь нейтрально-смешанные контексты с двумя сущностями
    # TODO расширить текущую выборку на двойные + смешанные контексты

    # TODO дообучение модели
    # TODO набрать нейтральных банковских/операторных контекстов из рамблера и попробовать еще раз
