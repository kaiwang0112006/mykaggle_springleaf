from parsedatav2 import *
from model import *
from featureSel import *
from sklearn.ensemble import *
from sklearn.feature_selection import VarianceThreshold

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]"',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train',dest='train',help='train', default='')
    parser.add_argument('--test',dest='test',help='test', default='')
    parser.add_argument('--fts',dest='fts',help='feture selection et. extraTrees, cor', default='extraTrees')

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
    fn = "destreeSub.csv"
    print fn
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
#     indices = [i for i in range(len(train_x[0]))]
#     frqIndex = trimfrq(train_x)
#     for i in frqIndex:
#         indices.remove(i)
#     train_x_uniq = indexTodata(train_x, indices)
#     test_x_uniq = indexTodata(test_x, indices)

    #normalization
    print "normalization"
    train_x_nor, mean, std = normalize(train_x_uniq)
    test_x_nor, mean, std = normalize(test_x_uniq, mean, std)

    #feature selection
    print "feature selection"
    if args.fts == 'cor':
        train_x_sel, test_x_sel = correlationSelect(train_x_nor, train_y, test_x_nor)
    elif args.fts == 'extraTrees':
        train_x_sel, test_x_sel = ExtraTreesSelect(train_x_nor, train_y, test_x_nor)
    else:
        train_x_sel = copy.deepcopy(train_x_nor)
        test_x_sel = copy.deepcopy(test_x_nor)
    del train_x_nor, test_x_nor, train_x_uniq, test_x_uniq
    print "modelsing"
    clf = ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=0, class_weight='auto')
    clf.fit(train_x_sel, train_y)
    train_pdt = clf.predict(train_x_sel)
    MCC, Acc_p , Acc_n, Acc_all = get_Accs(train_y, train_pdt) 
    print "MCC, Acc_p , Acc_n, Acc_all(train): "
    print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))
    test_pdt = clf.predict_proba(test_x_sel)
#     MCC, Acc_p , Acc_n, Acc_all = get_Accs(test_y, test_pdt) 
#     print "MCC, Acc_p , Acc_n, Acc_all(test): "
#     print "%s,%s,%s,%s" % (str(MCC), str(Acc_p) , str(Acc_n), str(Acc_all))   
    
    fout=open(fn,'w')
    fout.write("ID,target\n")
    for index, eachline in enumerate(test_pdt):
        fout.write("%s,%s\n" % (str(int(id[index])),str(test_pdt[index][1])))
    fout.close()
    
if __name__ == "__main__":
    main()
