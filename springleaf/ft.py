from parsedatav2 import *
from model import *
from featureSel import *
from sklearn.ensemble import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV

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

def main():
    args = getOptions()
    print args

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
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(train_y, 10),
                  scoring='accuracy')
    rfecv.fit(train_x_nor, train_y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)

    
if __name__ == "__main__":
    main()
