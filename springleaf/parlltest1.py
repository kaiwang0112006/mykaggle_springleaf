# -*- coding: utf-8 -*- 
#! /usr/local/bin/python2.7
# test.py

import time
import pprocess
import threading
from multiprocessing import Process,Queue
from multiprocessing.dummy import Pool as ThreadPool 

def f(qlist):
    q= qlist[0]
    i = qlist[1]
    q.put([42, None, 'hello',i])

if __name__ == '__main__':
    q = Queue()
    pl = []
    for i in range(2):
        pl.append(Process(target=f, args=([q,i],)))
        
    for p in pl:
        p.start()

    
    for p in pl:
        p.join()

    print q.get()  
    print q.get() 
