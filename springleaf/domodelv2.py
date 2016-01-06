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
    parser.add_argument('--fts',dest='fts',help='feture selection et. extraTrees, cor, randomTree', default='extraTrees')
    parser.add_argument('--model',dest='model',help="Classifiers, {'gBoosting', 'randomForest'}", default='gBoosting')
    #GradientBoostingClassifier
    parser.add_argument('--lrate',dest='lrate',help='learning_rate',type=float, default=0.3)
    parser.add_argument('--nest',dest='nest',help='n_estimators', type=int, default=100)
    parser.add_argument('--dep',dest='maxdepth',help='max_depth',type=int, default=3)
    parser.add_argument('--loss',dest='loss',help="loss function to be optimized,{'deviance', 'exponential'}", default='deviance')
    parser.add_argument('--minsamplessplit',dest='minsamplessplit',help="The minimum number of samples required to split an internal node.", type=int, default=2)
    #RandomForestClassifier
    #parser.add_argument('--nest',dest='nest',help='n_estimators', type=int, default=100)
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
    if args.model == 'gBoosting':
        fn = ("submission_%s_gBoosting_%s_%s_%s_%s_%s.csv" % (args.fts, args.loss, str(args.minsamplessplit), str(args.lrate).replace('.','dian'),str(args.nest),str(args.maxdepth)))
    elif args.model == 'randomForest':
        fn = ("submission_%s_randomForest_%s.csv" % (args.fts, args.nest))
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
    elif args.fts == 'randomTree':
        train_x_sel, test_x_sel = randomTreesSelect(train_x_nor, train_y, test_x_nor)
    else:
        train_x_sel = copy.deepcopy(train_x_nor)
        test_x_sel = copy.deepcopy(test_x_nor)
    print len(train_x_nor[0])
    print len(train_x_sel[0])
    
    del train_x_nor, test_x_nor, train_x_uniq, test_x_uniq
    print "modelsing"
    if args.model == 'gBoosting':
        clf = GradientBoostingClassifier(loss=args.loss, 
                                         learning_rate=args.lrate,
                                         n_estimators=args.nest,
                                         max_depth=args.maxdepth,
                                         min_samples_split=args.minsamplessplit,
                                         verbose=1)
    elif args.model == 'randomForest':
        clf = RandomForestClassifier(n_estimators=args.nest, class_weight='auto')
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
