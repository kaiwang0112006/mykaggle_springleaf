'''
Created on 2015.9.21

@author: milo
'''

import argparse

def head(file,out):
    fin = open(file)
    fout = open(out,'w')
    count = 0
    for i in range(100):
        eachline = fin.readline()
        fout.write(eachline)
    fin.close()
    fout.close()

trf = "train.csv"
tef = "test.csv"

head(trf,"train_head.csv")
head(tef,"test_head.csv")
