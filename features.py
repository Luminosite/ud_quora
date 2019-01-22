from math import log
import _pickle as cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')


def getLogger():
    import logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    ch.setFormatter(formatter)
    logger.handlers=[]
    logger.addHandler(ch)
    return logger


import os

def read_or_gen_by_list(l):
    current = len(l) - 1
    if current < 0:
        print("Error: input list is empty")
        return None
    return r_or_g_by_list_n(l, current)


def r_or_g_by_list_n(l, i):
    name = l[i][0]
    func = l[i][1]
    path = "data/{name}.csv".format(name=name)
    if os.path.exists(path):
        print("read the data for {name}".format(name=name))
        d = pd.read_csv(path, sep=",")
        print("data for {name} read".format(name=name))
        return d
    else:
        d = None
        print("generate the data for {name}".format(name=name))
        if i == 0:
            d = func()
        else:
            parameter = r_or_g_by_list_n(l, i-1)
            d = func(parameter)
        d.to_csv(path, index=False)
        print("data for {name} generated".format(name=name))
        return d


def wmd(s1, s2, model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2, norm_model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def base_feature_with_size(n, data_file, start):
    def basic_feature():
        data = pd.read_csv('data/{f}'.format(f=data_file), sep=',')
        if n > 0:
            data = data[start:n]
        if data_file != "test.csv":
            data = data.drop(['id', 'qid1', 'qid2'], axis=1)
        print("basic data is ready")

        data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
        data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
        data['diff_len'] = data.len_q1 - data.len_q2
        data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
        data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
        data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
        data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
        data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
        data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        return data
    return basic_feature


def wmd_feature(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'], model), axis=1)
    return data

def norm_wmd_feature(data):
    norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    norm_model.init_sims(replace=True)
    data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2'], norm_model), axis=1)
    return data


def sent2vec(s, model):
    words = str(s).lower()#.decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def dist_features(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    question1_vectors = np.zeros((data.shape[0], 300))
    error_count = 0

    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q, model)

    question2_vectors  = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = sent2vec(q, model)

    data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

    cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
    cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)
    
    return data


def gen_feature(n=0, data_file="train.csv", start=0):
    tag = data_file.split('.')[0]
    if(n>0):
        tag = "{t}_{start}_{end}".format(t=tag, start=start, end=n)
    operations = [
        ("basic_feature_{t}".format(t=tag), base_feature_with_size(n, data_file, start))
        ,("wmd_feature_{t}".format(t=tag), wmd_feature)
        ,("norm_wmd_feature_{t}".format(t=tag), norm_wmd_feature)
        ,("dist_features_{t}".format(t=tag), dist_features)
        #,("add_normal_wmd_feature", normal_wmd_feature)
    ]

    features_data = read_or_gen_by_list(operations)
    return features_data


def gen_ratio(col, data):
    data['{c}_ratio_ln'.format(c=col)] = data.apply(
        lambda x: log(x['{c}1'.format(c=col)]/x['{c}2'.format(c=col)]),
        axis=1)
    return data


def common_ratio_feature(data):
    data['common_ratio'] = data.apply(lambda x: 2 * float(x['common_words']) /
                                                (float(x['len_word_q1']) + float(x['len_word_q2'])),
                                      axis=1)
    cols = ["len_q", "len_char_q", "len_word_q"]
    for col in cols:
        data = gen_ratio(col, data)
    return data


def gen_common_ratio_feature(n=0, data_file="train.csv", start=0):
    tag = data_file.split('.')[0]
    if n > 0:
        tag = "{t}_{start}_{end}".format(t=tag, start=start, end=n)
    operations = [
        ("basic_feature_{t}".format(t=tag), base_feature_with_size(n, data_file, start))
        , ("wmd_feature_{t}".format(t=tag), wmd_feature)
        , ("norm_wmd_feature_{t}".format(t=tag), norm_wmd_feature)
        , ("dist_features_{t}".format(t=tag), dist_features)
        , ("common_ratio_features_{t}".format(t=tag), common_ratio_feature)
    ]

    features_data = read_or_gen_by_list(operations)
    return features_data
