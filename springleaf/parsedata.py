#!/usr/bin/env python

import sys
import subprocess
import argparse
import shutil
import os


##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]"')
    parser.add_argument('-i,--input',dest='input',help='input', default='')
    parser.add_argument('-o,--output',dest='output',help='output', default='')
    
    args = parser.parse_args()
    if args.input == '' or args.output == '':
        parser.print_help()
        print ''
        print 'You forgot to provide some data files!'
        print 'Current options are:'
        print args
        sys.exit(1) 
    return args


def judegvalue(cell,i):
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
            v = 0.0
            for s in cell:
                v += float(ord(s))
            return v
    
##########################################
## Master function
##########################################
def main():
    args = getOptions()
    
    fin = open(args.input)
    fout = open(args.output,'w')
    
    count = 0
    for eachline in fin:
        if count == 0:
            fout.write(eachline)
        else:
            line = eachline.strip().split(',')
            for index,cell in enumerate(line):
                value = judegvalue(cell,index)
                fout.write(str(value))
                fout.write("%s" % ("," if index!=len(line)-1 else '\n'))
        count += 1 
    fin.close()
    fout.close()
    
if __name__ == "__main__":
    main()