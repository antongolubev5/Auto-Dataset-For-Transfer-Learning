def main():
    """Нужно достать из него отдельные существительные, которые однозначно являются отрицательными или положительными — две группы слов.
        Возможно придется почистить и оставить только названия людей.
        Также нужно достать прилагательные — однозначно положительные или отрицательные.
        Скорее всего понадобятся.
        Эти данные мы будем использовать для нахождения отрицательных и положительных сущностей в текстах.
    """
    ru_senti_lex = open('/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/RuSentiLex', 'r')
    nouns_pos = open('/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/nouns_pos', 'w')
    nouns_neg = open('/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/nouns_neg', 'w')
    adjs_pos = open('/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/adjs_pos', 'w')
    adjs_neg = open('/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/adjs_neg', 'w')

    for line in ru_senti_lex:
        line_info = line.strip().split(', ')
        if line_info[1] == 'Noun':
            if line_info[3] == 'positive':
                nouns_pos.write(line)
            elif line_info[3] == 'negative':
                nouns_neg.write(line)
        elif line_info[1] == 'Adj':
            if line_info[3] == 'positive':
                adjs_pos.write(line)
            elif line_info[3] == 'negative':
                adjs_neg.write(line)

    for file in [ru_senti_lex, nouns_pos, nouns_neg, adjs_pos, adjs_neg]:
        file.close()


if __name__ == "__main__":
    main()
