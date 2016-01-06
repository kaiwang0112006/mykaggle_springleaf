#!/usr/bin/env python

import sys
import argparse
import csv
import copy
import numpy as np
from model import *
import xgboost as xgb

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
    gbm = xgb.XGBClassifier(max_depth=10, n_estimators=500, learning_rate=0.03).fit(train_x_trans, train_y,eval_metric='auc')
    train_pdt = gbm.predict(train_x_trans)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
    print "MCC, Acc_p , Acc_n, Acc_all(train): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
    test_pdt = gbm.predict(test_x_trans)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
    print "MCC, Acc_p , Acc_n, Acc_all(test): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))   
    
    fout=open("submissionp3.csv",'w')
    fout.write("ID,target\n")
    for index, eachline in enumerate(test_pdt):
        fout.write("%s,%s\n" % (str(int(test_x[index][0])),str(test_pdt[index])))
    fout.close()

    
if __name__ == "__main__":
    main()
