from parsedatav2 import *
from model import *
from featureSel import *
from sklearn.ensemble import *
from sklearn.feature_selection import VarianceThreshold

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]"')
    parser.add_argument('--train',dest='train',help='train', default='')
    parser.add_argument('--test',dest='test',help='test', default='')
    parser.add_argument('--lrate',dest='lrate',help='learning_rate',type=float, default=0.3)
    parser.add_argument('--nest',dest='nest',help='n_estimators', type=int, default=100)
    parser.add_argument('--dep',dest='maxdepth',help='max_depth',type=int, default=3)
    
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
    fn = ("submission_cor_%s_%s_%s.csv" % (str(args.lrate).replace('.','dian'),str(args.nest),str(args.maxdepth)))
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
    train_x_sel, test_x_sel = correlationSelect(train_x_nor, train_y, test_x_nor)
#     ftsel = correlationSel()
#     ftsel.dosel(train_x_nor,train_y)
#     train_x_sel = ftsel.transform(train_x_nor)
#     test_x_sel = ftsel.transform(test_x_nor)
    print "modelsing"
    clf = GradientBoostingClassifier(loss='deviance', 
                                     learning_rate=args.lrate,
                                     n_estimators=args.nest,
                                     max_depth=args.maxdepth,
                                     verbose=1)
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
