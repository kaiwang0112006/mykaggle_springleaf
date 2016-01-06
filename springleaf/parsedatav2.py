#!/usr/bin/env python

import sys
import subprocess
import argparse
import shutil
import os
import csv
import copy
global content 

content = {}

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

#To read parsed train and test file for modeling
def readfile_noid(file,ftype,lineNo=0):
    x = []
    y = []
    csvfile = open(file)
    fin = csv.reader(csvfile)
    
    for eachline in fin:
        if (fin.line_num == 1 or fin.line_num % lineNo != 0) and lineNo != 0: 
            continue
#         if fin.line_num == lineNo:
#             break
        if ftype == "train":
            x.append([float(i) for i in eachline[:-1]])
            y.append(float(eachline[-1]))
        elif ftype == "test":
            x.append([float(i) for i in eachline])
            y.append(1.0)

    csvfile.close()
    return x,y

#first column is id, so delet it before modeling
def extractID(x, index=0):
    x_new = []
    id = []
    for eachline in x:
        id.append(eachline[0])
        temp = [eachline[i] for i in range(len(eachline)) if i!=0]
        x_new.append(temp)
    return x_new, id

def judegvalue(cell,i):
    global content
    if "\"" in cell:
        cell = cell.strip()[1:-1]
    try:
        return float(cell)
    except:
        if cell.upper() == 'FALSE' or cell == "NA" or cell == "" or cell == "[]":
            return 0
        elif cell.upper() == 'TRUE':
            return 1
        else:
            if not i in content:
                content[i] = {cell:1}
                return 1
            else:
                if cell in content[i]:
                    return content[i][cell]
                else:
                    maxval = max(content[i].values())
                    content[i][cell] = maxval + 1
            return maxval + 1

def parse(filename,ftype='train'):
    if ftype == 'train':
        output = 'train_out.csv'
    elif ftype == 'test':
        output = 'test_out.csv'
    else:
        print ftype
        print 'exit'
        exit(0)
        
        
        
    csvfile = open(filename)
    fout = open(output,'w')
    
    fin = csv.reader(csvfile)
    for eachline in fin:
        if fin.line_num == 1:
            fout.write(','.join(eachline)+'\n')
        else:
            for index,cell in enumerate(eachline):
                value = judegvalue(cell,index)
                fout.write(str(value))
                fout.write("%s" % ("," if index!=len(eachline)-1 else '\n'))

    fout.close() 

##########################################
# for city(column 200), zip(column 236) and state(column 239) clean up
##########################################
def cityclean(x, contentdict={}):
    x_new = []
    for eachline in x:
        state_zip = str(int(eachline[238])) + str(int(eachline[235]))
        if not state_zip in contentdict:
            if contentdict == {}:
                contentdict[state_zip] = 1
            else:
                maxval = max(contentdict.values())
                contentdict[state_zip] = maxval + 1
                
        line = copy.deepcopy(eachline)
        line[199] = contentdict[state_zip]
        line[238] = 0
        line[235] = 0
        x_new.append(line)

    return x_new, contentdict

##########################################

##########################################
## Master function
##########################################
def main():
    global content
    args = getOptions()
    print args
    parse(args.train,'train')
    parse(args.test,'test')
 
    
if __name__ == "__main__":
    main()
