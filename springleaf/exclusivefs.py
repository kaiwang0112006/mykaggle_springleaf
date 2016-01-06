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

def exclusivefs(train_x, train_y, test_x, test_y, id=[]):
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
            clf = GradientBoostingClassifier(loss='exponential', 
                                             learning_rate=0.1,
                                             n_estimators=1500,
                                             max_depth=3,
                                             verbose=0)
            
            tr_x_new = trimx(tr_x,ilist)
            te_x_new = trimx(te_x,ilist)
            clf.fit(tr_x_new, tr_y)
            
            train_pdt = clf.predict(tr_x_new)
            test_pdt = clf.predict(te_x_new)

            trprt = [int(i) for i in test_pdt]
            
            fpr, tpr, thresholds = metrics.roc_curve(y, trprt, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            
            if auc > maxauc and auc >0.78:
                output = open('exclusive/output.txt', 'a') 
                output.write("%s\t%s\n" % (str(ilist),str(auc)))
                maxauc = auc
                output.close()
                print("%s\t%s\n" % (str(ilist),str(auc)))
                test_x_new = trimx(test_x,ilist)
                test_pdt = clf.predict_proba(test_x_new)
                
                fn = "exclusive/submission_auc%s_time%s" % (str(auc).replace('.','dian'),str(time.time()).replace('.','dian'))
                
                fout=open(fn,'w')
                fout.write("ID,target\n")
                for index, eachline in enumerate(test_pdt):
                    fout.write("%s,%s\n" % (str(int(id[index])),str(test_pdt[index][1])))
                fout.close()


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
    train_x, train_y = readfile_noid(args.train,'train',40000)
    train_x_new, id = extractID(train_x)
    del train_x
    train_x_clean, contentdict = cityclean(train_x_new)
    del id, train_x_new
    print "test file read"
    test_x, test_y = readfile_noid(args.test,'test')
    test_x_new, id = extractID(test_x)
    del test_x
    test_x_clean, contentdict = cityclean(test_x_new, contentdict)
    del test_x_new
    del contentdict, 
    
    #remove feature with no distinction and less important
    print "remove feature with no distinction and less important"
    sel = VarianceThreshold()
    train_x_uniq = sel.fit_transform(train_x_clean)
    del train_x_clean
    test_x_uniq = sel.transform(test_x_clean)
    del test_x_clean
    
    #normalization
    print "normalization"
    train_x_nor, mean, std = normalize(train_x_uniq)
    del train_x_uniq
    test_x_nor, mean, std = normalize(test_x_uniq, mean, std)
    del test_x_uniq
    
    #feature selection and modeling
    print "feature selection and modeling"
    exclusivefs(train_x_nor, train_y, test_x_nor, test_y, id)

if __name__ == "__main__":
    main()