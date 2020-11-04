import os
import patoolib
from patoolib.util import PatoolError


def main():
    """разархивация корпуса, плохие архивы удаляются
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


if __name__ == '__main__':
    main()
    # 10 архивов повреждено, распаковать не удалось
# [
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201103/20110321/20110321015054_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201103/20110321/20110321021057_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201103/20110321/20110321020103_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201103/20110313/20110313181758_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201101/20110105/20110105105014_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201101/20110105/20110105092645_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201101/20110105/20110105091640_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201101/20110105/20110105093705_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201101/20110105/20110105110108_utf.rar',
#     '/media/anton/ssd2/data/datasets/aspect-based-sentiment-analysis/Rambler_source/201101/20110105/20110105103935_utf.rar']
