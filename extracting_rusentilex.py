def extract_nouns_and_adjs():
    """
    Выделение существительных и прилагательных из полного словаря Русентилекс
    """
    ru_senti_lex = open('/home/anton/data/ABSA/contexts/txt/RuSentiLex', 'r')
    nouns_pos = open('/home/anton/data/ABSA/contexts/txt/nouns_pos', 'w')
    nouns_neg = open('/home/anton/data/ABSA/contexts/txt/nouns_neg', 'w')
    adjs_pos = open('/home/anton/data/ABSA/contexts/txt/adjs_pos', 'w')
    adjs_neg = open('/home/anton/data/ABSA/contexts/txt/adjs_neg', 'w')

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


def main():
    extract_nouns_and_adjs()


if __name__ == "__main__":
    main()
