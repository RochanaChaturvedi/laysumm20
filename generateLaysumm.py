from BART.generate_bart_summ import generate_bart_summ
from Utilities.prepare_data import prepareData
from Utilities.merge_summaries import mergeSumm
from wMVC.generate_wMVC_summ import generate_wMVC

from os import listdir, mkdir
from warnings import filterwarnings

filterwarnings('ignore')


def main():
    directories = listdir('Data/')
    for folder in ['Section-wise-summaries', 'Input-Data', 'Sections-DataFrame', 'Merged-final', 'Input-wMVC',
                   'Input-BART']:
        if folder not in directories:
            mkdir(f'Data/{folder}')

    prepareData()
    generate_wMVC()
    generate_bart_summ()
    mergeSumm()


if __name__ == '__main__':
    main()
