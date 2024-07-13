import os

from utils import LOGGER, colorstr
from utils.filesys_utils import txt_write



def print_samples(source, target, prediction):
    LOGGER.info('\n' + '-'*100)
    LOGGER.info(colorstr('SRC       : ') + source)
    LOGGER.info(colorstr('GT        : ') + target)
    LOGGER.info(colorstr('Prediction: ') + prediction)
    LOGGER.info('-'*100 + '\n')


def cal_multi_bleu_perl(ref, pred):
    r = [s+'\n' for s in ref]
    p = [s+'\n' for s in pred]

    txt_write('etc/ref.txt', r)
    txt_write('etc/pred.txt', p)

    cmd = 'etc/multi_bleu.perl ' + 'etc/ref.txt < ' + 'etc/pred.txt'
    os.system(cmd)

    os.remove('etc/ref.txt')
    os.remove('etc/pred.txt')