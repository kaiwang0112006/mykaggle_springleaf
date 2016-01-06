'''
Created on 2015.9.21

@author: milo
'''

def head(file,out):
    fin = open(file)
    fout = open(out,'w')
    count = 0
    for i in range(5):
        eachline = fin.readline()
        fout.write(eachline)
    fin.close()
    fout.close()

trf = "train.csv"
tef = "test.csv"

head(trf,"train_5.csv")
head(tef,"test_5.csv")
