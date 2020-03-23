import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath
from keras import Input, layers
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from gensim.models.wrappers import FastText
from navec import Navec


def embeddings_download(file_path):
    """
    загрузка представлений слов русскоязычной модели с rusvectores
    :param file_path: директория с файлами
    :return: model
    """
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    # embeddings_index = {}
    # f = open(file_path, 'r', encoding='utf-8')
    #
    # for line in f:
    #     values = line.split()
    #     word = values[0].split("_")[0]
    #     coeffs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coeffs
    # f.close()

    return model


def data_download(directory_path, files: list):
    """
    загрузка +/- контекстов в соотношении 50/50
    :param directory_path: директория с файлами
    :param files: имена файлов, откуда грузить
    ПОРЯДОК СКАЧИВАНИЯ И ЗАПИСИ: -/0/+
    +удаление тональных слов из выборки
    """

    contexts = []
    texts = []

    for file in files:
        with open(os.path.join(directory_path, file), 'r') as contexts_file:
            contexts.append(contexts_file.readlines())

    min_len = min(len(contexts[0]), len(contexts[1]))

    for i in range(len(contexts)):
        texts += contexts[i][:min_len]

    for i in range(len(texts)):
        tonal_words = texts[i].strip().split("===")[2].split(" ")
        texts[i] = texts[i].strip().split("===")[1].split(" ")
        for tonal_word in tonal_words:
            texts[i].remove(tonal_word)

    labels = [0] * min_len + [1] * min_len

    return texts, labels


def texts_len_distribution(texts: list):
    """
    построение распределения длин текстов коллекции
    """

    plt.style.use('ggplot')
    # plt.figure(figsize=(16, 9))
    # facecolor='g'
    n, bins, patches = plt.hist([len(text) for text in texts], 50, density=True)
    plt.xlabel('Number of words in a text')
    plt.ylabel('Share of texts')
    plt.axis([0, 40, 0, 0.12])
    plt.grid(True)
    plt.show()


def text_vectorization(texts, labels, embeddings_model):
    """
    векторизация текстов с помощью предварительно обученных embeddings
    приведение метод к нормальной форме
    :param embeddings_model: предварительно обученная модель
    :param labels: метки текстов
    :param texts: тексты
    :return:
    """

    embed_len = len(embeddings_model['лес'])
    text_len = 20

    y = np.asarray(labels)
    X = np.zeros((len(texts), text_len, embed_len), dtype=np.float16)

    for i in range(len(texts)):
        for j in range(min(len(texts[i]), text_len)):
            if texts[i][j] in embeddings_model.wv.index2word:
                X[i][j] = embeddings_model[texts[i][j]]

    return X, y


def nonlinear_svm(X_train, X_test, y_train, y_test):
    """
    svm с нелинейным ядром
    :return:
    """
    clf_SVC = SVC(C=0.1, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True,
                  probability=False, tol=0.001, cache_size=1000, class_weight=None,
                  verbose=True, max_iter=-1, decision_function_shape="ovr", random_state=0)
    clf_SVC.fit(X_train, y_train)

    print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))
    print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

    # confusion_matrix
    print(confusion_matrix(clf_SVC.predict(X_test), y_test))


def build_model_rnn(embed_len):
    """
    построение модели rnn
    :param embed_len: длина векторного представления
    :return:
    """
    model = Sequential()
    model.add(layers.SimpleRNN(embed_len, return_sequences=True))
    model.add(layers.SimpleRNN(embed_len, return_sequences=True))
    model.add(layers.SimpleRNN(embed_len))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_cnn():
    """
    построение модели cnn
    :return:
    """
    model = Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_multi_cnn(text_len, embed_len):
    """
    построение модели cnn
    :return:
    """
    input_tensor = Input(shape=(text_len, embed_len))
    branch_a = layers.Conv1D(32, 2, activation='relu')(input_tensor)
    branch_a = layers.MaxPooling1D(2)(branch_a)
    branch_a = layers.Conv1D(32, 3, activation='relu')(branch_a)
    branch_a = layers.GlobalMaxPooling1D()(branch_a)

    branch_b = layers.Conv1D(32, 3, activation='relu')(input_tensor)
    branch_b = layers.MaxPooling1D(2)(branch_b)
    branch_b = layers.Conv1D(32, 3, activation='relu')(branch_b)
    branch_b = layers.GlobalMaxPooling1D()(branch_b)

    branch_c = layers.Conv1D(32, 4, activation='relu')(input_tensor)
    branch_c = layers.MaxPooling1D(2)(branch_c)
    branch_c = layers.Conv1D(32, 2, activation='relu')(branch_c)
    branch_c = layers.GlobalMaxPooling1D()(branch_c)

    branch_d = layers.Conv1D(32, 5, activation='relu')(input_tensor)
    branch_d = layers.MaxPooling1D(2)(branch_d)
    branch_d = layers.Conv1D(32, 2, activation='relu')(branch_d)
    branch_d = layers.GlobalMaxPooling1D()(branch_d)

    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

    x = layers.Dropout(0.4)(output)
    x = layers.Dense(30, activation='relu')(x)
    output_tensor = layers.Dense(1, activation='sigmoid')(x)

    model = Model(input_tensor, output_tensor)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_multi_cnn_with_embed(embed_len, vocab_power, sentence_len, embedding_matrix):
    """
    построение модели cnn с начальным embedding слоем
    :return:
    """
    tweet_input = Input(shape=(26,), dtype='int32')
    tweet_encoder = Embedding(vocab_power, embed_len, input_length=sentence_len,
                              weights=[embedding_matrix], trainable=True)(tweet_input)

    x = layers.Dropout(0.2)(tweet_encoder)

    branch_a = layers.Conv1D(32, 2, activation='relu')(x)
    branch_a = layers.MaxPooling1D(2)(branch_a)
    branch_a = layers.Conv1D(32, 3, activation='relu')(branch_a)
    branch_a = layers.GlobalMaxPooling1D()(branch_a)

    branch_b = layers.Conv1D(32, 3, activation='relu')(x)
    branch_b = layers.MaxPooling1D(2)(branch_b)
    branch_b = layers.Conv1D(32, 3, activation='relu')(branch_b)
    branch_b = layers.GlobalMaxPooling1D()(branch_b)

    branch_c = layers.Conv1D(32, 4, activation='relu')(x)
    branch_c = layers.MaxPooling1D(2)(branch_c)
    branch_c = layers.Conv1D(32, 2, activation='relu')(branch_c)
    branch_c = layers.GlobalMaxPooling1D()(branch_c)

    branch_d = layers.Conv1D(32, 5, activation='relu')(x)
    branch_d = layers.MaxPooling1D(2)(branch_d)
    branch_d = layers.Conv1D(32, 2, activation='relu')(branch_d)
    branch_d = layers.GlobalMaxPooling1D()(branch_d)

    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=1)

    x = layers.Dropout(0.4)(output)
    x = layers.Dense(30, activation='relu')(x)
    output_tensor = layers.Dense(1, activation='sigmoid')(x)

    model = Model(tweet_input, output_tensor)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_sequences(tokenizer, x, sentence_len):
    """
    text_to_seq - преобразовывает текст в последовательность чисел (номеров слов)
    pad_sequences - добавляет нули в начале до необходимой длины текста
    :param tokenizer: обученный токенизатор
    :param x: текст твита
    :param sentence_len: максимальная длина твита
    :return:
    """
    sequences = tokenizer.texts_to_sequences(x)
    return pad_sequences(sequences, maxlen=sentence_len)


def build_model_multi_cnn_his(text_len, embed_len):
    """
    построение модели cnn из статьи
    :return:
    """
    input_tensor = Input(shape=(text_len, embed_len))
    branch_a = layers.Conv1D(filters=1, kernel_size=2, padding='valid', activation='relu')(input_tensor)
    branch_a = layers.GlobalMaxPooling1D()(branch_a)

    branch_b = layers.Conv1D(filters=1, kernel_size=3, padding='valid', activation='relu')(input_tensor)
    branch_b = layers.GlobalMaxPooling1D()(branch_b)

    branch_c = layers.Conv1D(filters=1, kernel_size=4, padding='valid', activation='relu')(input_tensor)
    branch_c = layers.GlobalMaxPooling1D()(branch_c)

    branch_d = layers.Conv1D(filters=1, kernel_size=5, padding='valid', activation='relu')(input_tensor)
    branch_d = layers.GlobalMaxPooling1D()(branch_d)

    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

    x = layers.Dropout(0.2)(output)
    x = layers.Dense(30, activation='relu')(x)
    output_tensor = layers.Dense(1, activation='sigmoid')(x)

    model = Model(input_tensor, output_tensor)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_model_rnn_with_embedding(max_features, embed_len):
    """
    построение модели rnn со слоем embedding
    :param max_features: кол во слов в алфавите
    :param embed_len: длина векторного представления
    :return:
    """
    model = Sequential()
    model.add(layers.Embedding(max_features, embed_len))
    model.add(layers.LSTM(embed_len))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def loss_graph(model_history):
    """
    график потерь на этапах обучения и проверки
    :return:
    """
    history_dict = model_history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(model_history.epoch) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def accuracy_graph(model_history):
    """
    график точности на этапах обучения и проверки
    :return:
    """
    history_dict = model_history.history
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(model_history.epoch) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, cmap=plt.cm.Blues, my_tags=[0, 1]):
    """
    построение матрицы ошибок в форме оттенков цветов
    :param cm: вычисленная матрица ошибок
    :param cmap: карта цветов
    :param my_tags: метки классов
    :return:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(my_tags))
    target_names = my_tags
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_confusion_matrix_pro(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.winter):
    """
    визуализация матрицы ошибок
    :param cm:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt


def load_language_model(language_models_path, language_model_name, texts, self_train: bool, save_self_model: bool,
                        self_model_name):
    """
    загрузка/обучение языковой модели
    :param language_models_path: директория с яз. моделями
    :param language_model_name: имя яз. модели
    :param texts: корпус текстов (если обучаем свою модель)
    :param self_train: обучаем ли свою
    :param save_self_model: сохранять ли свою обученную
    :param self_model_name: имя своей модели
    :return:
    """
    if self_train:
        language_model = Word2Vec(texts, min_count=0, size=300)
        if save_self_model:
            language_model.save(os.path.join(language_models_path, self_model_name + '.w2v'))
    else:
        if language_model_name[-3:] == 'bin':
            language_model = KeyedVectors.load_word2vec_format(
                datapath(os.path.join(language_models_path, language_model_name)), binary=True, )
        elif language_model_name[-3:] == 'bin':
            language_model = FastText.load_fasttext_format(os.path.join(language_models_path, language_model_name))
        else:
            language_model = Word2Vec.load(os.path.join(language_models_path, language_model_name))

    return language_model


def main():
    start_time = time.time()

    directory_path = '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis'
    language_models_path = '/media/anton/ssd2/data/datasets/language_models'
    language_model_name = 'tweets_model.w2v'

    texts, labels = data_download(directory_path, ['negative_contexts', 'positive_contexts'])
    language_model = load_language_model(language_models_path, language_model_name, texts, self_train=True,
                                         save_self_model=False, self_model_name=None)
    X, y = text_vectorization(texts, labels, language_model)

    # сохранение векторизованных текстов для дальнейшего использования
    # np.save(save_arrays_path + '\\X_len20.npy', X)
    # np.save(save_arrays_path + '\\y.npy', y)

    # загрузка векторизованных текстов
    # X = np.load(save_arrays_path + '\\X_len20.npy')
    # y = np.load(save_arrays_path + '\\y.npy')

    # разделение выборки на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
    del X

    # nonlinear svm
    # nonlinear_svm(X_train, X_test, y_train, y_test)

    # RNN
    # mdl = build_model_rnn(300)
    #
    # history = mdl.fit(X_train,
    #                   y_train,
    #                   epochs=8,
    #                   batch_size=128,
    #                   validation_split=0.2)
    # loss_graph(history)
    # accuracy_graph(history)
    # print(mdl.evaluate(X_test, y_test))
    # print(mdl.summary())

    # CNN
    # mdl = build_model_cnn(300)
    #
    # history = mdl.fit(X_train,
    #                   y_train,
    #                   epochs=10,
    #                   batch_size=128,
    #                   validation_split=0.2)
    # loss_graph(history)
    # accuracy_graph(history)
    # print(mdl.evaluate(X_test, y_test))
    # print(mdl.summary())

    # multi CNN
    mdl = build_model_multi_cnn(X_train.shape[1], X_train.shape[2])

    history = mdl.fit(X_train,
                      y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)
    loss_graph(history)
    accuracy_graph(history)
    print(mdl.evaluate(X_test, y_test))
    print(mdl.summary())

    # multi CNN with embedding layer

    # # создаем и обучаем токенизатор
    # tokenizer = Tokenizer(num_words=100000)
    # tokenizer.fit_on_texts(X_train)
    #
    # # отображаем каждый текст в массив идентификаторов токенов
    # x_train_seq = get_sequences(tokenizer, X_train)
    # x_test_seq = get_sequences(tokenizer, X_test)
    #
    # # загружаем обученную модель word2vec
    # # w2v_model = Word2Vec.load(r'D:\datasets\языковые модели\tweets.model')
    # embed_len = own_model.vector_size
    # vocab_power = 100000
    # sentence_len = 26
    #
    # # инициализируем матрицу embedding слоя нулями
    # embedding_matrix = np.zeros((vocab_power, embed_len))
    # # Добавляем NUM=100000 наиболее часто встречающихся слов из обучающей выборки в embedding слой
    # for word, i in tokenizer.word_index.items():
    #     if i >= vocab_power:
    #         break
    #     if word in own_model.wv.vocab.keys():
    #         embedding_matrix[i] = own_model.wv[word]
    #
    # # mdl = build_model_multi_cnn_with_embed(X_train.shape[0], X_train.shape[1], vocab_power, sentence_len,
    # #                                        embedding_matrix)
    #
    # mdl = build_model_rnn()
    #
    # history = mdl.fit(X_train,
    #                   y_train,
    #                   epochs=10,
    #                   batch_size=128,
    #                   validation_split=0.2)
    # loss_graph(history)
    # accuracy_graph(history)
    # print(mdl.evaluate(X_test, y_test))
    # print(mdl.summary())

    # mdl.save(save_model_path + 'cnn_many_layers.h5')

    # загрузка обученной предсказывающей модели
    # mdl = load_model(save_model_path + 'cnn.h5')

    # проверка классификации
    # tweet = "какой хороший сегодня день"
    # print(tweet, predict_tweet(tweet, mdl, own_model))

    # confusion matrix
    cm = confusion_matrix(y_test, np.around(mdl.predict(X_test)))
    plot_confusion_matrix(cm, cmap=plt.cm.Blues, my_tags=[0, 1])

    print('f1 score = {}!'.format(f1_score(y_test, np.around(mdl.predict(X_test)))))

    total_time = round((time.time() - start_time))
    print("Time elapsed: %s minutes %s seconds" % ((total_time // 60), round(total_time % 60)))


if __name__ == "__main__":
    main()
