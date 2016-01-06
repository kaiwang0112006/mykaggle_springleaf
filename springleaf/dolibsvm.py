
from model import *
import argparse
import csv
from svmutil import *

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

def libsvm(train_x, train_y, test_x, test_y):
    c_begin, c_end, c_step = -10,  10, 5
    g_begin, g_end, g_step = -10,  10, 5
    best_c=best_g=0.0
    best_rate_train=best_rate_test=0.0
    
    for i in range(int(c_begin), int(c_end), int(c_step)):
        for j in range(g_begin, g_end, g_step):
            c=2**(i)
            g=2**(j)
            param="-c "+str(c)+" -g "+str(g)
            
            pv,acc_test = get_pred_label(train_y, train_x, test_y, test_x, param)
            if acc_test>best_rate_test:
                best_rate_test=acc_test
                bestpv=pv
                pv,acc_train = get_pred_label(train_y, train_x, train_y, train_x, param)
                best_rate_train=acc_train
                best_c=c
                best_g=g
    print 
    print
    print "result\n"
    print("c=%s, g=%s, train=%s test=%s" % (best_c, best_g, best_rate_train,best_rate_test))

def get_pred_label(train_y, train_x, test_y, test_x, param):
    model = svm_train(train_y, train_x, param)
    py, evals, deci = svm_predict(test_y, test_x, model)
    return py,evals[0]
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
    del train_x
    del test_x    
    #normalization
    print "normalization"
    train_x_nor, mean, std = normalize(train_x_uniq)
    test_x_nor, mean, std = normalize(test_x_uniq, mean, std)
    del train_x_uniq
    del test_x_uniq
    libsvm(train_x_nor, train_y, test_x_nor, test_y)
    
    
if __name__ == "__main__":
    main()
