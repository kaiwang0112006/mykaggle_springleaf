#!/usr/bin/env python

import sys
import argparse
import csv
import copy
import numpy as np
from model import *
import xgboost as xgb
from featureSel import *
from sklearn.feature_selection import VarianceThreshold

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
    train_x, train_y = readfile_noid(args.train,'train')
    train_x_new, id = extractID(train_x)
    del id
    print "test file read"
    test_x, test_y = readfile_noid(args.test,'test')
    test_x_new, id = extractID(test_x)

    #remove feature with no distinction and less important
    print "remove feature with no distinction and less important"
    
    sel = VarianceThreshold()
    train_x_uniq = sel.fit_transform(train_x_new)
    test_x_uniq = sel.transform(test_x_new)
    
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
    train = xgb.DMatrix(train_x_trans,label=train_y)
    test = xgb.DMatrix(test_x_trans,label=test_y)
    gbm = xgb.train({'max_depth':3, 'n_estimators':1500, 'learning_rate':0.1 ,'objective':'binary:logistic','eval_metric':'auc'},train)
    train_pdt = gbm.predict(train)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
    print "MCC, Acc_p , Acc_n, Acc_all(train): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
    test_pdt = gbm.predict(test)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
    print "MCC, Acc_p , Acc_n, Acc_all(test): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))   
    
    fout=open("submission_xgbtrain.csv",'w')
    fout.write("ID,target\n")
    for index, eachline in enumerate(test_pdt):
        fout.write("%s,%s\n" % (str(int(id[index])),str(test_pdt[index])))
    fout.close()

    
if __name__ == "__main__":
    main()
