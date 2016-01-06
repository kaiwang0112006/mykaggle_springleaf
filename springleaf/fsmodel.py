#!/usr/bin/env python

import sys
import argparse
import csv
import copy
import numpy as np
from model import *
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from operator import itemgetter
from sklearn.ensemble import *

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

def ftransform(x,sel,k):
    score = list(enumerate(sel.scores_))
    score.sort(key=itemgetter(1),reverse=1) 
    
    xsor = []
    
    count = 0
    
    for eachline in x:
        temp = []
        count = 0
        for s in score:
            if count >k :
                break
            temp.append(eachline[s[0]])
        xsor.append(temp)
            
    
    return xsor
        
    
    

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
    del train_x_uniq
    del test_x_uniq
    #feature selection
    print "feature selection model"
    scoreFuncs = [f_classif]
    for k in range(0, len(scoreFuncs)):
        for j in range(500,len(train_x_nor[0])-1,500):
            featureSelector = SelectKBest(score_func=scoreFuncs[k], k=j)
            XSel = featureSelector.fit(train_x_nor, train_y)
            train_x_trans = ftransform(train_x_nor,XSel,j)
            test_x_trans = ftransform(test_x_nor,XSel,j)
            print scoreFuncs[k]
            print j
            clf = GradientBoostingClassifier(loss='deviance', 
                                     learning_rate=0.05,
                                     n_estimators=500,
                                     max_depth=3,
                                     verbose=0)
            clf.fit(train_x_trans, train_y)
            train_pdt = clf.predict(train_x_trans)
            MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
            print "MCC, Acc_p , Acc_n, Acc_all(train): "
            print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
            test_pdt = clf.predict_proba(test_x_trans)
            #MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
            #print "MCC, Acc_p , Acc_n, Acc_all(test): "
            #print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))   
            fn = "submission_fun%s_%s.csv" % (str(k),str(j))            
            fout=open(fn,'w')
            fout.write("ID,target\n")
            for index, eachline in enumerate(test_pdt):
                fout.write("%s,%s\n" % (str(int(test_x[index][0])),str(test_pdt[index])))
            fout.close()


    
if __name__ == "__main__":
    main()
