from math import log
from poolprocess import process_data
import _pickle as cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import os

# from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

stop_words = stopwords.words('english')

# evaluation for size of train.csv
block_size = 400000


def time_cnt(f, tag="func"):
    print("function '%s' starts" % tag)
    t_start = time()
    ret = f()
    t_end = time()
    t_used = t_end - t_start
    print("function '%s' use: %f s" % (tag, t_used))
    return ret


def get_logger():
    import logging
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)
    return logger


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
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2, norm_model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def lemmatize_all(sentence):
    wnl = nltk.WordNetLemmatizer()
    for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
        if tag.startswith('NN'):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
        else:
            yield word


def rebuild_sentence(s):
    sentence = str(s).lower()
    return ' '.join(lemmatize_all(sentence))


def rebuild_question(data):
    feats = ["question%d" % x for x in range(1, 3)]
    for f in feats:
        data[f] = data[f].apply(lambda x: rebuild_sentence(x))
    return data


def base_process_thread(data):
    data = time_cnt(lambda: rebuild_question(data), tag="clean question string")

    data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split())
                                                    .intersection(set(str(x['question2']).lower().split()))), axis=1)
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(
        str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(
        str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(
        str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(
        str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(
        str(x['question1']), str(x['question2'])), axis=1)
    return data


def base_feature_with_size(n, data_file, start):

    def basic_feature():
        data = pd.read_csv('data/{f}'.format(f=data_file), sep=',')
        if n > 0:
            data = data[start:n]
        if data_file != "test.csv":
            data = data.drop(['id', 'qid1', 'qid2'], axis=1)
        print("basic data is read")
        ret = process_data(base_process_thread, data, n=8, tag="base_feature")
        return ret
        
    return basic_feature


def wmd_thread(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2'], model), axis=1)
    return data


def wmd_feature(ret):
    ret = process_data(wmd_thread, ret, n=3, tag="wmd_feature")
    return ret


def norm_wmd_tread(data):
    norm_model = gensim.models.KeyedVectors.load_word2vec_format(
        'data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    norm_model.init_sims(replace=True)
    data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2'], norm_model), axis=1)
    return data


def norm_wmd_feature(ret):
    ret = process_data(norm_wmd_tread, ret, n=3, tag="norm_wmd_feature")
    return ret


def sent2vec(s, model):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if w.isalpha()]
    m = []
    for w in words:
        try:
            m.append(model[w])
        except Exception as _:
            continue
    m = np.array(m)
    v = m.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def dist_thread(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    question1_vectors = np.zeros((data.shape[0], 300))

    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q, model)

    question2_vectors = np.zeros((data.shape[0], 300))
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


def dist_features(ret):
    ret = process_data(dist_thread, ret, n=3, tag='dist_feature')
    return ret


def gen_ratio(col, data):
    data['{c}_ratio_ln'.format(c=col)] = data.apply(
        lambda x: log(x['{c}1'.format(c=col)]/x['{c}2'.format(c=col)]),
        axis=1)
    return data


def common_ratio_thread(data):
    data['common_ratio'] = data.apply(
        lambda x: 2*float(x['common_words']) / (float(x['len_word_q1']) + float(x['len_word_q2'])), axis=1)
    cols = ["len_q", "len_char_q", "len_word_q"]
    for col in cols:
        data = gen_ratio(col, data)
    return data


def common_ratio_feature(ret):
    ret = process_data(common_ratio_thread, ret, n=6, tag="common_ratio")
    return ret


def dist_features_for(data, q1, q2, tag="tfidf"):
    q1 = np.nan_to_num(q1)
    q2 = np.nan_to_num(q2)

    def add_dist_for(func):
        col_name = '{d}_distance_{t}'.format(d=func.__name__, t=tag)
        data[col_name] = [func(x, y) for (x, y) in zip(q1, q2)]

    add_dist_for(cosine)
    add_dist_for(cityblock)
    add_dist_for(jaccard)
    add_dist_for(canberra)
    add_dist_for(euclidean)
    add_dist_for(minkowski)
    add_dist_for(braycurtis)

    data['skew_q1vec_{t}'.format(t=tag)] = [skew(x) for x in q1]
    data['skew_q2vec_{t}'.format(t=tag)] = [skew(x) for x in q2]
    data['kur_q1vec_{t}'.format(t=tag)] = [kurtosis(x) for x in q1]
    data['kur_q2vec_{t}'.format(t=tag)] = [kurtosis(x) for x in q2]

    return data


def tfidf_gen(t_data):
    ft = ['question1', "question2"]
    train = t_data.loc[:, ft]
    
    print('Generate tfidf')
    feats = ft
    vect_orig = TfidfVectorizer(max_features=None, ngram_range=(1, 1), min_df=3)

    corpus = []
    for f in feats:
        train.loc[:, f] = train.loc[:, f].astype(str)
        corpus += train[f].values.tolist()
    vect_orig.fit(corpus)
    
    train_tfidf = vect_orig.transform(corpus)
    return train_tfidf


def attach_vec_directly(data, q1, q2, tag):
    # print("attach head:", data.shape, q1.shape, q2.shape)
    width = int(q1.shape[1])
    short_tag = tag.replace(' ', '')
    cols = [["q{i}_{s}_{sub_num}".format(i=i, s=short_tag, sub_num=x) for x in range(width)] for i in range(1, 3)]
    print("to attach cols:", cols)
    
    dq1 = pd.DataFrame(q1, columns=cols[0])
    dq2 = pd.DataFrame(q2, columns=cols[1])
    # print("to attach", data.shape, dq1.shape, dq2.shape)
    re_index_data = data.reset_index(drop=True)
    ret = pd.concat((re_index_data, dq1, dq2), axis=1)
    # print("attach ret", ret.shape, data.shape, dq1.shape, dq2.shape)
    return ret


def prepare_vec_dist_data(data, vec, tag="tag"):
    single_set_size = int(vec.shape[0]/2)
    q1 = vec[:single_set_size]
    q2 = vec[single_set_size:]
    
    dist_features_data = data
    if q1.shape[1] > 28:
        dist_features_data = dist_features_for(data, q1, q2, tag=tag)
    
    attached = dist_features_data
    if q1.shape[1] < 50:
        print("before attach for %s" % tag, dist_features_data.shape)
        attached = attach_vec_directly(dist_features_data, q1, q2, tag)
        print("after attach for %s" % tag, attached.shape)
    return attached

 
def transfer_vec_cal_dist(data, ms):
    tfidf = tfidf_gen(data)

    model_func, n_list = ms
    for n in n_list:
        tag = "%s_%d" % (model_func.__name__, n)
        print("transfer sparse vec for %s" % tag)
        model = model_func(n_components=n)
        transferred = model.fit_transform(tfidf)
        print("prepare features for %s" % tag)
        data = prepare_vec_dist_data(data, transferred, tag=tag)
    return data


def svd300(d):
    return transfer_vec_cal_dist(d, (TruncatedSVD, [300]))


def svd300_feature(data):
    data_block_number = int((data.shape[0]-1) / block_size) + 1
    data = process_data(svd300, data, n=2, tag='topic model', data_n=data_block_number)
    return data


def svd25(d):
    return transfer_vec_cal_dist(d, (TruncatedSVD, [25]))


def svd25_feature(data):
    data_block_number = int((data.shape[0]-1) / block_size) + 1
    data = process_data(svd25, data, n=2, tag='topic model', data_n=data_block_number)
    return data


def nmf30(d):
    return transfer_vec_cal_dist(d, (NMF, [30]))


def nmf30_feature(data):
    data_block_number = int((data.shape[0]-1) / block_size) + 1
    data = process_data(nmf30, data, n=2, tag='topic model', data_n=data_block_number)
    return data


###################################################################
# local word2vec distance
###################################################################
def get_words(data):
    w1 = data['question1'].apply(lambda x: str(x).split()) #  [s.split(' ') for s in str(data['question1'].values)]
    w2 = data['question2'].apply(lambda x: str(x).split()) #  [s.split(' ') for s in str(data['question2'].values)]
    return w1, w2


def sentence2vec(words, model, vec_size):
    m = []
    for w in words:
        if w in model:
            m.append(model[w])
    m = np.array(m)
    v = m.sum(axis=0)
    ex_sum = (v ** 2).sum()
    ret = (v / np.sqrt(ex_sum)) if ex_sum > 0 else (np.ones(vec_size) * np.sqrt(1.0/vec_size))
    return ret


def local_vec(data):
    vec_size = 200
    w1, w2 = get_words(data)
    words = np.concatenate((w1, w2), axis=0)
    model = gensim.models.word2vec.Word2Vec(words, min_count=5, size=vec_size, window=5)

    q1vec = [sentence2vec(w, model, vec_size) for w in w1]
    q2vec = [sentence2vec(w, model, vec_size) for w in w2]
    return dist_features_for(data, q1vec, q2vec, tag="local_word2vec")


def local_vec_feature(data):
    data_block_number = int((data.shape[0]-1) / block_size) + 1
    print("block num for loc vec", data_block_number, data.shape[0])
    data = process_data(local_vec, data, n=4, tag='local word2vec', data_n=data_block_number)
    return data


###################################################################
# pos vec distance
###################################################################
pos_base = {'WP': 0, 'VB': 0, 'DT': 0, 'NN': 0, 'IN': 0, 'RB': 0, 'TO': 0,
            'JJ': 0, 'WRB': 0, 'MD': 0, 'PRP': 0, 'CC': 0, 'VBP': 0}
pos_features = list(pos_base)


def get_pos_vec(sentence):
    dic = pos_base.copy()
    for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
        if tag in dic:
            dic[tag] = dic[tag] + 1
    return [dic[x] for x in pos_features]


def attach_vec_with_tags(data, q1, q2, tags):
    cols = [["q{i}_{t}".format(i=i, t=short_tag) for short_tag in tags] for i in range(1, 3)]
    print("to attach cols:", cols)
    dq1 = pd.DataFrame(q1, columns=cols[0])
    dq2 = pd.DataFrame(q2, columns=cols[1])
    re_index = data.reset_index(drop=True)
    return pd.concat([re_index, dq1, dq2], axis=1)


def pos_vec(data):
    print("process pos vec features for", pos_features)
    q1 = data['question1'].apply(lambda x: get_pos_vec(str(x))).values.tolist()
    q2 = data['question2'].apply(lambda x: get_pos_vec(str(x))).values.tolist()

    data = dist_features_for(data, q1, q2, tag='pos_vec')
    return attach_vec_with_tags(data, q1, q2, pos_features)


def pos_vec_feature(data):
    data_block_number = int((data.shape[0]-1) / block_size) + 1
    data = process_data(pos_vec, data, n=8, tag='pos vec', data_n=data_block_number)
    return data
###################################################################
# graph feature
###################################################################


###################################################################
# process interface
###################################################################
operations = (
    ("basic_feature_", base_feature_with_size),
    ("wmd_feature_", wmd_feature),
    ("norm_wmd_feature_", norm_wmd_feature),
    ("dist_features_", dist_features),
    ("common_ratio_features_", common_ratio_feature),
    ("svd300_features_", svd300_feature),
    ("svd25_features_", svd25_feature),
    ("nmf30_features_", nmf30_feature),
    ("local_vec_", local_vec_feature),
    ("pos_vec_", pos_vec_feature)
)


def feature_generation(n=0, data_file="train.csv", start=0, operation_list=operations):
    tag_base = data_file.split('.')[0]
    tag = tag_base if n <= 0 else "{t}_{start}_{end}".format(t=tag_base, start=start, end=n)
    ops = []
    for i, itr in enumerate(operation_list):
        base_name, func = itr
        gen_func = func(n, data_file, start) if i == 0 else func
        full_tag = base_name + tag
        ops.append((full_tag, gen_func))
    features_data = read_or_gen_by_list(ops)
    return features_data


def gen_transferred_dist_feature(n=0, data_file="train.csv", start=0):
    return feature_generation(n=n, data_file=data_file, start=start, operation_list=operations[:8])


def gen_n_feature(n=0, data_file="train.csv", start=0, nf=10):
    return feature_generation(n=n, data_file=data_file, start=start, operation_list=operations[:nf])
