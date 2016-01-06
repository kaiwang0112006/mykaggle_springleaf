# -*- coding: utf-8 -*- 
#! /usr/local/bin/python2.7
# test.py

import time
import pprocess # ��ģ��ֻ����linux��ʹ��
import threading
from multiprocessing import Process 
from multiprocessing.dummy import Pool as ThreadPool 
import urllib2 

def takeuptime(n):
    urllib2.urlopen(n)

if __name__ == '__main__':
    list_of_args = [
    'http://www.python.org', 
    'http://www.python.org/about/',
    'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
    'http://www.python.org/doc/',
    'http://www.python.org/download/',
    'http://www.python.org/getit/',
    'http://www.python.org/community/',
    'https://wiki.python.org/moin/',
    'http://planet.python.org/',
    'https://wiki.python.org/moin/LocalUserGroups',
    'http://www.python.org/psf/',
    'http://docs.python.org/devguide/',
    'http://www.python.org/community/awards/'
    # etc.. 
    ]

    # Serial computation
    start = time.time()
    serial_results = [takeuptime(args) for args in list_of_args]
    print "%f s for traditional, serial computation." % (time.time() - start)

    # Parallel computation
    nproc = 4 # maximum number of simultaneous processes desired
    results = pprocess.Map(limit=nproc, reuse=1)
    parallel_function = results.manage(pprocess.MakeReusable(takeuptime))
    start = time.time()
    # Start computing things
    for args in list_of_args:
        parallel_function(args)
    parallel_results = results[:]
    print "%f s for parallel computation." % (time.time() - start)

    # Multithreading computation
    nthead = 4 # number of threads
    threads = [threading.Thread(target=takeuptime, args=(list_of_args[i],)) for i in range(nthead)]
    start = time.time()
    # Start threads one by one
    for thread in threads:
        thread.start()
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    print "%f s for multithreading computation." % (time.time() - start)


    # Multiprocessing computation
    process = []
    nprocess = 4 # number of processes
    for i in range(nprocess):
        process.append(Process(target=takeuptime, args=(list_of_args[i],)))
    start = time.time()
    # Start processes one by one
    for p in process:
        p.start()
    # Wait for all processed to finish
    for i in process:
        p.join()
    print "%f s for multiprocessing computation." % (time.time() - start)

     
    # ------- 4 Pool ------- # 
    start = time.time()
    pool = ThreadPool(4) 
    results = pool.map(takeuptime, list_of_args)
    print "%f s for 4 pool computation." % (time.time() - start) 
    # ------- 8 Pool ------- # 
    start = time.time()
    pool = ThreadPool(8) 
    results = pool.map(takeuptime, list_of_args)
    print "%f s for 8 pool computation." % (time.time() - start) 
    # ------- 13 Pool ------- # 
    start = time.time()
    pool = ThreadPool(13) 
    results = pool.map(takeuptime, list_of_args)
    print "%f s for 13 pool computation." % (time.time() - start) 
