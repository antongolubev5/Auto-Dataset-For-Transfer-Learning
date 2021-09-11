import os
import patoolib
from patoolib.util import PatoolError


def parsing_corpus():
    """
    разархивация, поврежденные архивы удаляются
    """
    directory_path = '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source'
    bad_files = []

    for month in os.listdir(directory_path):
        for day in os.listdir(os.path.join(directory_path, month)):
            for rar_file in os.listdir(os.path.join(directory_path, month, day)):
                rarname = os.path.join(directory_path, month, day, rar_file)
                dirname = os.path.join(directory_path, month, day, rar_file[:-4])
                if not (os.path.exists(dirname)):
                    os.mkdir(dirname)
                    try:
                        patoolib.extract_archive(rarname, 0, dirname)
                    except PatoolError:
                        bad_files.append(rarname)
                        pass
                    os.remove(rarname)
    print(bad_files)


def main():
    parsing_corpus()


if __name__ == '__main__':
    main()