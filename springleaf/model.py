#!/usr/bin/env python

import sys
import argparse
import csv
import copy
import numpy as np
from sklearn import ensemble, preprocessing
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import *
import math
from sklearn import metrics
from sklearn.tree import *
from parsedatav2 import *
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]"')
    parser.add_argument('--train',dest='train',help='train', default='')
    parser.add_argument('--test',dest='test',help='test', default='')
    
    args = parser.parse_args()
    if args.train == '' or args.test == '':
        parser.print_help()
        print ''
        print 'You forgot to provide some data files!'
        print 'Current options are:'
        print args
        sys.exit(1) 
    return args


def trimfrq(des):
    desmap = map(list, zip(*des))
    desindex = []
    for index,eachdes in enumerate(desmap):
        eachdes = list(set(eachdes))
        if len(eachdes) == 1:
            desindex.append(index)
    return desindex

def indexTodata(data, indices):
    newdata = []
    for eachline in data:
        temp = []
        for i in indices:
            try:
                temp.append(eachline[i])
            except:
                print (len(eachline))
                exit(0)
        newdata.append(temp)
        
    return newdata       

def normalize(x, mean=None, std=None):
    flist = map(list, zip(*x))
    
    count = len(x)
    if mean is None:
        mean = []
        for i in flist:
            mean.append(np.mean(i))
    if std is None:
        std = []
        for i in flist:
            std.append(np.std(i))
    for i in range(count):
        for j in range(len(x[i])):
            x[i][j] = (x[i][j]-mean[j])/std[j]
    return x, mean, std

def checkeachClassfier(train_x, train_y, test_x, test_y):
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(class_weight='auto'),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        DecisionTreeClassifier(class_weight='auto'),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        RandomForestClassifier(class_weight='auto'),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()]
    
    classtitle = ["KNeighborsClassifier",
                  "SVC",
                  "SVC weighted",
                  "SVC(gamma=2, C=1)",
                  "DecisionTreeClassifier",
                  "DecisionTreeClassifier weighted",
                  "RandomForestClassifier",
                  "RandomForestClassifier weighted",
                  "AdaBoostClassifier",
                  "GaussianNB",
                  "LDA",
                  "QDA"]
    
    for i in range(len(classtitle)):
        try:
            ctitle = classtitle[i]
            clf = classifiers[i]
            clf.fit(train_x, train_y)
            train_pdt = clf.predict(train_x)
            MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
            print ctitle+":"
            print "MCC, Acc_p , Acc_n, Acc_all(train): "
            print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
            test_pdt = clf.predict(test_x)
            MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
            print "MCC, Acc_p , Acc_n, Acc_all(test): "
            print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
            fn = "submission_%s.csv" % ctitle
            fout=open(fn,'w') 
            fout.write("ID,target\n")
            for index, eachline in enumerate(test_pdt):
                fout.write("%s,%s\n" % (str(int(test_x[index][0])),str(test_pdt[index])))
            fout.close()       
        except:
            print ctitle+": error" 
        print

def get_Accs(ty,pv):
    if len(ty) != len(pv):
        raise ValueError("len(ty) must equal to len(pv)")
    tp = tn = fp = fn = 0
    for v, y in zip(pv, ty):
        if int(y) == int(v):
            if int(y) == 1:
                tp += 1
            else:
                tn += 1
        else:
            if int(y) == 1:
                fn +=1
            else:
                fp +=1
    tp=float(tp)
    tn=float(tn)
    fp=float(fp)
    fn=float(fn)

    MCC_x = tp*tn-fp*fn
    a = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    MCC_y = float(math.sqrt(a))
    if MCC_y == 0:
        MCC = 0
    else:
        MCC = float(MCC_x/MCC_y)
    try:
        Acc_p=tp/(tp+fn)
    except:
        Acc_p=0
        
    try:
        Acc_n=tn/(tn+fp)
    except:
        Acc_n=0
        
    Acc_all = (tp + tn)/(tp + tn + fp + fn)
    return (MCC, Acc_p , Acc_n, Acc_all)

##########################################
## Master function
##########################################
def main():
    args = getOptions()
    print "train file read"
    train_x, train_y = readfile(args.train,'train')
    print "test file read"
    test_x, test_y = readfile(args.test,'test')

    #remove feature with no distinction and less important
    print "remove feature with no distinction and less important"
    indices = [i for i in range(len(train_x[0]))]
    frqIndex = trimfrq(train_x)
    
    for i in frqIndex:
        indices.remove(i)
    train_x_uniq = indexTodata(train_x, indices)
    test_x_uniq = indexTodata(test_x, indices)
    
    #normalization
    print "normalization"
    train_x_nor, mean, std = normalize(train_x_uniq)
    test_x_nor, mean, std = normalize(test_x_uniq, mean, std)
    
    #feature selection
    print "feature selection"
    ftsel = ExtraTreesClassifier()
    ftsel.fit(train_x_nor, train_y)
#     importances = ftsel.feature_importances_
#     indices_test = np.argsort(importances)[::-1]
#     indices_test = indices_test.tolist()
    train_x_trans = ftsel.transform(train_x_nor)
    test_x_trans = ftsel.transform(test_x_nor)
    
    #modelsing
    print "modelsing"
    checkeachClassfier(train_x_trans, train_y, test_x_trans, test_y)

    
if __name__ == "__main__":
    main()
