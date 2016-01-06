# -*- coding: utf-8 -*- 
#######################################################################
# only use a small part of train.csv to do exclusive feature selection
#####################################################################
from parsedatav2 import *
from model import *
from featureSel import *
from sklearn.ensemble import *
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.metrics.ranking import auc
import time
import itertools

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]"',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train',dest='train',help='train', default='')
    #parser.add_argument('--test',dest='test',help='test', default='')

    args = parser.parse_args()
    if args.train == '':
        parser.print_help()
        print ''
        print 'You forgot to provide some data files!'
        print 'Current options are:'
        print args
        sys.exit(1) 
    return args

def exclusivefs(train_x, train_y):
    long = len(train_x[0])
    findex = range(long)
    splitnum = int(len(train_x)/2)
    tr_x = train_x[:splitnum]
    tr_y = train_y[:splitnum]
    te_x = train_x[splitnum:]
    te_y = train_y[splitnum:]
    y = [int(i) for i in te_y]

    maxauc = 0
    for i in range(long,0,-1):
        print i
        iter = itertools.combinations(findex,i)
        for ilist in iter:
            print ilist
            clf = GradientBoostingClassifier()
            
            tr_x_new = trimx(tr_x,ilist)
            te_x_new = trimx(te_x,ilist)
            clf.fit(tr_x_new, tr_y)
            
            train_pdt = clf.predict(tr_x_new)
            test_pdt = clf.predict(te_x_new)

            trprt = [int(i) for i in test_pdt]
            
            fpr, tpr, thresholds = metrics.roc_curve(y, trprt, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print auc
            if auc > maxauc and auc >0.78:
                fn = "exclusive/outputexcluv2_%s_%s_%s.thread" % (str(i), str(j), str(time.time()).replace('.','dian'))
                output = open(fn, 'w') 
                output.write("%s\t%s\n" % (str(ilist),str(auc)))
                maxauc = auc
                output.close()
            del tr_x_new, te_x_new, train_pdt, test_pdt


def trimx(x,flist):
    x_new = []
    for eachline in x:
        line = []
        for i in flist:
            line.append(eachline[i])
        x_new.append(line)
    return x_new

def main():
    args = getOptions()
    print args

    print "train file read"
    train_x, train_y = readfile_noid(args.train,'train',4)
    train_x_new, id = extractID(train_x)
    del train_x
    train_x_clean, contentdict = cityclean(train_x_new)
    del id, train_x_new
    
    #remove feature with no distinction and less important
    print "remove feature with no distinction and less important"
    sel = VarianceThreshold()
    train_x_uniq = sel.fit_transform(train_x_clean)
    del train_x_clean
    
    #normalization
    print "normalization"
    train_x_nor, mean, std = normalize(train_x_uniq)
    del train_x_uniq
    
    #feature selection and modeling
    print "feature selection and modeling"
    exclusivefs(train_x_nor, train_y)

if __name__ == "__main__":
    main()