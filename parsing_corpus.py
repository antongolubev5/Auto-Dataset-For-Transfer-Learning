import re
import os
import spacy
import xml.etree.ElementTree as ET
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS
import time
from pymystem3 import Mystem
from string import punctuation
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
import itertools

# import ru2e

spacy.prefer_gpu()


def cosine_similarity(a, b):
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


def spacy_tokenizer(text, lemm: bool):
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
    cnt = 0

    with open(os.path.join(directory_path, sentences_file), 'r') as corpus_sentences:
        firstNlines = corpus_sentences.readlines()

    cnt = 1

    for line in firstNlines:
        print(cnt, '/', len(firstNlines), ' = ', round(cnt / len(firstNlines) * 100, 2), '%...')
        line_tok = spacy_tokenizer(line, True)
        if any(word in list_entities_vocab_keys for word in line_tok):
            cnt += 1
            contexts.write(line.strip() + '===' + ' '.join(line_tok))
            print(cnt, line.strip() + '===' + ' '.join(line_tok))
        cnt += 1

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
    cnt = 1

    for i in range(len(corpus_sentences)):
        print(cnt, '/', len(corpus_sentences), ' = ', round(cnt / len(corpus_sentences) * 100, 2), '%...')
        line_tok = spacy_tokenizer(corpus_sentences.iloc[i][0], True)
        if any(word in entities_vocab for word in line_tok):
            cnt += 1
            contexts = contexts.append(
                pd.Series([corpus_sentences.iloc[i][0].strip(), ' '.join(line_tok)], index=contexts.columns),
                ignore_index=True)
            print(cnt, corpus_sentences.iloc[i][0].strip() + '===' + ' '.join(line_tok))
        cnt += 1

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
    cnt = 1

    with open(os.path.join(directory_path, contexts), 'r') as contexts:
        contexts_lines = contexts.readlines()

    for line in contexts_lines:
        print(cnt, '/', len(contexts_lines), ' = ', round(cnt / len(contexts_lines) * 100, 2), '%...')
        line_text = line.split('===')[0]
        line_tok = line.split('===')[1].strip()
        flag, lst = check_tones(line_tok.split(" "), entities_vocab)

        if flag == 1:
            positive_contexts.write(line_text + '===' + line_tok + '===' + ' '.join(lst) + '===' + '1' + '\n')
        elif flag == -1:
            negative_contexts.write(line_text + '===' + line_tok + '===' + ' '.join(lst) + '===' + '-1' + '\n')
        else:
            print(line_text)

        cnt += 1

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
    posneg_new = pd.DataFrame(columns=contexts_all.columns)
    j = 0
    for i in range(len(contexts_all)):
        if any(word in entities_vocab for word in contexts_all.iloc[i]['text_tok'].split()) and len(
                contexts_all.iloc[i]['text_tok'].split()) > 10 \
                and not tonal_word_in_quotes(contexts_all.iloc[i]['text'], contexts_all.iloc[i]['tonal_word']):
            posneg_new = posneg_new.append(contexts_all.iloc[i], ignore_index=True)
        print(i, i / len(contexts_all) * 100)
    return posneg_new


def drop_multi_entities_sentences(contexts_all):
    """
    удаление из выборки предложений, содержащих несколько сущностей
    """
    contexts = pd.DataFrame(columns=contexts_all.columns)
    j = 1
    for i in range(len(contexts_all)):
        if len(contexts_all.iloc[i]['tonal_word'].split()) == 1:
            contexts = contexts.append(contexts_all.iloc[i])
            j += 1
        print(i, i / len(contexts_all) * 100)
    return contexts


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
    bias = 40 if sentiment == 1 else 140
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
    классификация предложения на 3 группы по тональности: смешанная, положительная, отрицательная
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


def create_balanced_samples(contexts_all, volumes, top_words, drop_volume):
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
    entities_vocab = entities_vocab
    contexts = pd.DataFrame(columns=['text', 'text_tok', 'tonal_word', 'label', 'sent_type'])
    cnt = 1

    for i in range(len(raw_data)):
        print(cnt, '/', len(raw_data), ' = ', round(cnt / len(raw_data) * 100, 2), '%...')
        context_text = raw_data.iloc[i]['sentence']
        context_tok = spacy_tokenizer(context_text, True)
        if any(word in entities_vocab for word in context_tok) and len(context_tok) > 10:
            flag, sentiment_words = check_sentiment_of_sentence(context_tok, entities_vocab)
            quotes = False
            for sentiment_word in sentiment_words:
                if tonal_word_in_quotes(context_text, sentiment_word):
                    quotes = True
            if len(sentiment_words) <= 2 and not quotes:
                if flag == 1:
                    contexts = contexts.append(
                        pd.Series(
                            [context_text, ' '.join(context_tok), ' '.join(sentiment_words), 1,
                             check_sentiments(context_tok, entities_vocab)], index=contexts.columns),
                        ignore_index=True)
                elif flag == -1:
                    contexts = contexts.append(
                        pd.Series([context_text, ' '.join(context_tok), ' '.join(sentiment_words), -1,
                                   check_sentiments(context_tok, entities_vocab)], index=contexts.columns),
                        ignore_index=True)
                elif flag == 0:
                    contexts = contexts.append(
                        pd.Series([context_text, ' '.join(context_tok), ' '.join(sentiment_words), 0,
                                   check_sentiments(context_tok, entities_vocab)], index=contexts.columns),
                        ignore_index=True)
                else:
                    print(context_text)
        cnt += 1

    contexts = drop_same_sentences(contexts)
    multi_contexts = contexts[~contexts['sent_type'].isin(['pos', 'neg'])]
    single_contexts = contexts[contexts['sent_type'].isin(['pos', 'neg'])]
    return single_contexts, multi_contexts


def drop_similar_contexts_tfidf(contexts_all):
    """
    в выборке много перепечатанных и очень похожих контекстов
    нужно прорядить ее одним из методов
    """
    similarities = pd.DataFrame(columns=['similarity', 'id1', 'id2'])
    corpus = list(contexts_all['text_tok'])
    vectorizer = TfidfVectorizer(min_df=0.002, use_idf=True, ngram_range=(1, 1))
    X = vectorizer.fit_transform(corpus).toarray()
    cnt = 0
    for i, j in itertools.combinations(contexts_all.index, 2):
        print(cnt / ((len(contexts_all) * len(contexts_all) - 1) / 2) * 100)
        value = cosine_similarity(X[i], X[j])
        # if value > 0.001:
        #     similarities = similarities.append(pd.Series([value, i, j], index=similarities.columns), ignore_index=True)
        cnt += 1
    # similarities = pd.DataFrame.from_dict(dictinary_list)


def main():
    start_time = time.time()

    directory_path = '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis'
    corpus_name = 'Rambler_source'
    entities_vocab = vocab_from_file(directory_path, ['nouns_person_neg', 'nouns_person_pos'])
    # entities_vocab = vocab_from_file(directory_path, ['nouns_person_pos'])

    # raw_data = pd.read_csv(os.path.join(directory_path, 'unlabeled_contexts.csv'), sep='\t')
    # single, multi = from_raw_sentences_to_dataset(raw_data, entities_vocab)
    #
    # single.to_csv(os.path.join(directory_path, 'new_single_contexts.csv'), index=False, sep='\t')
    # multi.to_csv(os.path.join(directory_path, 'new_multi_contexts.csv'), index=False, sep='\t')

    contexts_all = pd.read_csv(os.path.join(directory_path, 'single_contexts.csv'), sep='\t')
    drop_similar_contexts_tfidf(contexts_all[:3000])

    # contexts_all = create_balanced_samples(contexts_all, volumes=[5000, 5000], top_words=[40, 30], drop_volume=[3, 2])
    # plot_words_distribution(contexts_all, sentiment=-1, volume=-1, save=False)
    # plot_words_distribution(contexts_all, sentiment=1, volume=-1, save=False)
    # contexts_all.to_csv(os.path.join(directory_path, 'single_balanced_contexts_50_50.csv'), index=False, sep='\t')

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))


if __name__ == '__main__':
    main()
