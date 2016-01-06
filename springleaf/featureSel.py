from scipy.stats.stats import pearsonr   
from sklearn.ensemble import *
import copy

class correlationSel(object):
    def __init__(self):
        self.corlist = []
        self.fdel = []

    def dosel(self, x, y, inner=0.9,outter=0.1):
        self.calcor(x,y)

        lf = len(x[0])
        findex = range(lf)
        self.fdel = []
        #delet non-relevant feature with activity
        for f in findex:
            if self.corlist[f][-1] <= outter:
                self.fdel.append(f)
        
        #delet non-relevant feature pairs
        start = 1
        for f in findex:
            if f not in self.fdel:
                for i in range(start,len(findex)):
                    if findex[i] not in self.fdel:
                        if abs(self.corlist[f][findex[i]]) >= inner:
                            if abs(self.corlist[f][-1]) < abs(self.corlist[findex[i]][-1]):
                                self.fdel.append(f)
                                break
                            elif abs(self.corlist[f][-1]) > abs(self.corlist[findex[i]][-1]):
                                self.fdel.append(findex[i])
            start += 1
                          
    def calcor(self,x,y):
        flist = map(list, zip(*x))
        lf = len(flist)
        
        for i in range(lf):
            temp = []
            for j in range(lf):
                cor = pearsonr(flist[i],flist[j])[0]
                temp.append(cor)
            cor = pearsonr(flist[i],y)[0]
            temp.append(cor)
            self.corlist.append(temp)
    
    def transform(self,x):  
        xtransform = []
        for eachline in x:
            temp = []
            for i,v in enumerate(eachline):
                if i not in self.fdel:
                    temp.append(v)
            xtransform.append(temp)
        return xtransform
    
def correlationSelect(train_x, train_y, test_x):
    ftsel = correlationSel()
    ftsel.dosel(train_x,train_y)
    train_x_sel = ftsel.transform(train_x)
    test_x_sel = ftsel.transform(test_x)
    return train_x_sel, test_x_sel

def ExtraTreesSelect(train_x, train_y, test_x):
    ftsel = ExtraTreesClassifier()
    ftsel.fit(train_x, train_y)
#     importances = ftsel.feature_importances_
#     indices_test = np.argsort(importances)[::-1]
#     indices_test = indices_test.tolist()
    train_x_sel = ftsel.transform(train_x)
    test_x_sel = ftsel.transform(test_x)
    return train_x_sel, test_x_sel

def randomTreesSelect(train_x, train_y, test_x):
    ftsel = RandomForestClassifier(class_weight='auto')
    ftsel.fit(train_x, train_y)
#     importances = ftsel.feature_importances_
#     indices_test = np.argsort(importances)[::-1]
#     indices_test = indices_test.tolist()
    train_x_sel = ftsel.transform(train_x)
    test_x_sel = ftsel.transform(test_x)
    return train_x_sel, test_x_sel