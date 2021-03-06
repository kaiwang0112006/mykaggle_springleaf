from parsedatav2 import *
from model import *
from featureSel import *
from sklearn.ensemble import *
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.metrics.ranking import auc
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import os

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]"',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

def exclusivefs(train_x, train_y, test_x, test_y):
    long = len(train_x[0])
    findex = range(long)
    splitnum = int(len(train_x)/2)
    global tr_x 
    global tr_y 
    global te_x 
    global te_y 
    global y
    tr_x = train_x[:splitnum]
    tr_y = train_y[:splitnum]
    te_x = train_x[splitnum:]
    te_y = train_y[splitnum:]
    y = [int(i) for i in te_y]

    exclusivelist = []
    pool = Pool(10)
    print "exclusivelist"
    exclusivelist = pool.map(iterindex,range(long,0,-1))
    exclusivelistv2 = []
    for i in exclusivelistv2:
        for j in i:
            exclusivelistv2.append(j)
            
    print "exclusivemodel"
    print "all %d runs" % len(exclusivelistv2)
    pool = Pool(10)
    pool.map(modelworker,exclusivelistv2)

def iterindex(i):
    global findex
    print i
    iresult = []
    iter = itertools.combinations(findex,i)
    for ilist in iter:
        iresult.append(ilist)
    return iresult
    
def modelworker(ilist):
    global tr_x 
    global tr_y 
    global te_x 
    global te_y 
    global y

#     print "modelworker",i
#     iter = itertools.combinations(findex,i)
#     j = 1
#     for ilist in iter:
#         domodel(tr_x,tr_y,te_x,te_y,y,ilist,i,j)
#         j += 1

    print 'process id:', os.getpid()
    clf = GradientBoostingClassifier()
    
    tr_x_new = trimx(tr_x,ilist)
    te_x_new = trimx(te_x,ilist)
    clf.fit(tr_x_new, tr_y)
    
    train_pdt = clf.predict(tr_x_new)
    test_pdt = clf.predict(te_x_new)

    trprt = [int(i) for i in test_pdt]
    
    fpr, tpr, thresholds = metrics.roc_curve(y, trprt, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    if auc > 0.78:
        fn = "exclusive/outputv2_%s_%s.thread" % (suffix_i, suffix_j)
        output = open(fn, 'w') 
        output.write("%s\t%s\n" % (str(ilist),str(auc)))
        maxauc = auc
        output.close()
        

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
    train_x, train_y = readfile_noid(args.train,'train')
    train_x_new, id = extractID(train_x)
    train_x_clean, contentdict = cityclean(train_x_new)
    del id
    print "test file read"
    test_x, test_y = readfile_noid(args.test,'test')
    test_x_new, id = extractID(test_x)
    test_x_clean, contentdict = cityclean(test_x_new, contentdict)
    del contentdict
    
    #remove feature with no distinction and less important
    print "remove feature with no distinction and less important"
    sel = VarianceThreshold()
    train_x_uniq = sel.fit_transform(train_x_clean)
    test_x_uniq = sel.transform(test_x_clean)
    
    #normalization
    print "normalization"
    train_x_nor, mean, std = normalize(train_x_uniq)
    test_x_nor, mean, std = normalize(test_x_uniq, mean, std)
    
    #feature selection and modeling
    print "feature selection and modeling"
    exclusivefs(train_x_nor, train_y, test_x_nor, test_y)

if __name__ == "__main__":
    main()