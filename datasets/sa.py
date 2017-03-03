# Sentiment analysis dataset

from os.path import join


datadir = '/data/sentiment_analysis/processed_stars'
domains = ['kitchen', 'books', 'dvd', 'electronics']

def read_sa_file(featfile):
    with open(featfile, 'rb') as f:
        r = f.read().splitlines()
    return r

def line2instance(line):
    v = line.split()
    word_counts = {vv.split(':')[0]: int(vv.split(':')[1]) for vv in v[:-1]}
    rating = float(v[-1].split(':')[1])
    return (word_counts, rating)


def load_dset(domain, dset):
    featfile = join(datadir, domain, dset)
    r = read_sa_file(featfile)
    ir = [line2instance(line) for line in r]
    
    return ir


def add_worddict(word_dict, ir):
#    word_dict = {}
    for instance in ir:
        for (word, count) in instance[0].iteritems():
            if word_dict.has_key(word):
                word_dict[word] += count
            else:
                word_dict[word] = count
    return word_dict




ir_kitchen_tr = load_dset('kitchen', 'train')
ir_kitchen_te = load_dset('kitchen', 'test')
ir_books_tr = load_dset('books', 'train')
ir_books_te = load_dset('books', 'test')
ir_dvd_tr = load_dset('dvd', 'train')
ir_dvd_te = load_dset('dvd', 'test')
ir_electronics_tr = load_dset('dvd', 'train')
ir_electronics_te = load_dset('dvd', 'test')

word_dict = add_worddict({}, ir_kitchen_tr)
word_dict = add_worddict(word_dict, ir_books_tr)
word_dict = add_worddict(word_dict, ir_dvd_tr)
word_dict = add_worddict(word_dict, ir_electronics_tr)

# keep words which appear >5 times in all domains


word2index = {w:i for (i,w) in enumerate(word_dict.keys())}

# 
