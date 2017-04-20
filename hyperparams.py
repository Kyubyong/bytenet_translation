class Hp:
    '''Hyperparameters'''
    #     DE_TRAIN = '../corpora/train.tags.de-en.de'
    #     EN_TRAIN = '../corpora/train.tags.de-en.en'
    #     DE_TEST = '../corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    #     EN_TEST = '../corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    de_train = '/mnt/git/quasi-rnn/corpora/train.tags.de-en.de'
    en_train = '/mnt/git/quasi-rnn/corpora/train.tags.de-en.en'
    de_test = '/mnt/git/quasi-rnn/corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    en_test = '/mnt/git/quasi-rnn/corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    maxlen = 150 # Maximum sentence length
    batch_size = 16
    hidden_units = 320
    num_blocks = 7
    logdir = 'asset/train'
