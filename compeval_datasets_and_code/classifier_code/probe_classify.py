from __future__ import print_function

import random
import os
import gzip
import argparse
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def load_embdict(embfile):
    embdict = {}
    if embfile.endswith('.gz'):
        file = gzip.open(embfile, 'rU')
    else:
        file = open(embfile, 'rU')
    for line in file:
        if line[0] == '{' or len(line.strip()) < 1: continue
        id,sent,emb = line.strip().split('\t')
        embdict[id] = (sent,emb)
    file.close()
    return embdict

def get_xyclass_ftrs(file_path,sentembdict,word2id):
    X_loc = []
    X_randprobe = []
    X_randsent = []
    y = []
    ids = []
    sents = []
    allprobes = []
    cnt = 0
    with open(file_path) as fh:
        for ln in fh.readlines():
            label, id , sent = ln.strip().split('\t')[:3]
            probes = ln.strip().split('\t')[3:]
            sent2,sentemb = sentembdict[id]
            if sent != sent2:
                raise Exception('NOT EQUAL: %s || %s'%(sent,sent2))
            sentemb = [float(v) for v in sentemb.split()]


            probe1hot = []
            randprobe = []

            for probe in probes:

                probeid = word2id[probe]
                sg_probe1hot = [0] * len(word2id)
                sg_probe1hot[probeid] = 1
                probe1hot += sg_probe1hot

                sg_randprobe = [0] * len(word2id)
                randid = np.random.randint(len(word2id))
                sg_randprobe[randid] = 1
                randprobe += sg_randprobe

            randsent = list(np.random.randn(len(sentemb)))

            y.append(label)
            locftrs = sentemb + probe1hot
            randpftrs = sentemb + randprobe
            randsftrs = randsent + probe1hot

            X_loc.append(locftrs)
            X_randprobe.append(randpftrs)
            X_randsent.append(randsftrs)

            ids.append(id)
            sents.append(sent)
            allprobes.append(probes)
            cnt += 1
    return (np.asarray(X_loc),np.asarray(X_randprobe),np.asarray(X_randsent),np.asarray(y),ids,sents,allprobes)

def xy_classify(train_X,train_y,test_X,test_y,report_f,test_ids,test_sents,test_probes):

    #encoding class labels
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_y = np.asarray(le.transform(train_y))

    hidden_layer_sizes =(train_X.shape[1])
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation='relu',max_iter=700)

    clf.fit(train_X, train_y)


    y_pred = le.inverse_transform(clf.predict(test_X))
    dist_pred = clf.predict_proba(test_X)


    report = classification_report(test_y, y_pred)
    acc = accuracy_score(test_y, y_pred)
    accuracy = 'classification accuracy: %0.4f' % acc

    with open(report_f,'w') as out:
        out.write('REPORT\n')
        out.write(report + '\n\n')
        out.write(accuracy + '\n\n')
        out.write('\t'.join(['ID','PRED','TRUTH','CORR','SENT'])+'\n\n')
        for i in range(len(test_y)):
            pred = y_pred[i]
            truelab = test_y[i]
            id = test_ids[i]
            sent = test_sents[i]
            probes = test_probes[i]
            if pred == truelab:
                predcorr = '1'
            else:
                predcorr = '0'

            out.write('\t'.join([id,pred,truelab,predcorr,sent] + probes)+'\n')
    return report,accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('traintestdir')
    parser.add_argument('sentembfile')
    args = parser.parse_args()

    ttdir = args.traintestdir
    sentembfile = args.sentembfile

    s = ''

    vocab = []
    with open(os.path.join(ttdir,'vocab.txt')) as vocfile:
        for line in vocfile:
            vocab.append(line.strip())
    word2id = {}
    for i,word in enumerate(vocab): word2id[word] = i

    resultsdir = os.path.join(ttdir,'results')
    if not os.path.isdir(resultsdir): os.mkdir(resultsdir)


    sentembdict = load_embdict(sentembfile)

    X_train_loc,X_train_randp,X_train_rands,y_train,_,_,_ = get_xyclass_ftrs(os.path.join(ttdir,'train.txt'),sentembdict,word2id)
    X_test_loc,X_test_randp,X_test_rands,y_test,test_ids,test_sents,test_probes = get_xyclass_ftrs(os.path.join(ttdir,'test.txt'),sentembdict,word2id)


    s += '\nLOCALIST CLASSIFICATION\n'
    rep,acc = xy_classify(X_train_loc,y_train,X_test_loc,y_test,os.path.join(resultsdir,'loc_results.txt'),test_ids,test_sents,test_probes)
    s += '%s\n%s\n'%(rep,acc)
    s += '\nRANDOM PROBEVEC CLASSIFICATION\n'
    rep,acc = xy_classify(X_train_randp,y_train,X_test_randp,y_test,os.path.join(resultsdir,'randp_results.txt'),test_ids,test_sents,test_probes)
    s += '%s\n%s\n'%(rep,acc)
    s += '\nRANDOM SENTVEC CLASSIFICATION\n'
    rep,acc = xy_classify(X_train_rands,y_train,X_test_rands,y_test,os.path.join(resultsdir,'rands_results.txt'),test_ids,test_sents,test_probes)
    s += '%s\n%s\n'%(rep,acc)

    print(s)
    with open(os.path.join(resultsdir,'full_results.txt'),'w') as out:
        out.write(s)
