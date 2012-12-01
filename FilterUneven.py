#!/usr/bin/env python
# -*- coding: Latin-1 -*-
# vim: tabstop=4 shiftwidth=4 expandtab

""" Filter uneven spaced data.
"""

import numpy
from numpy import ma
from UserDict import UserDict

import threading

max_connections = 3
semaphore = threading.BoundedSemaphore(max_connections)
semaphore.acquire() # decrements the counter

semaphore.release() # increments the counter

class FilterUneven(UserDict):
    """
    """
    def __init__(self,data):
        """
	"""
	self.data = data
	return

from multiprocessing import BoundedSemaphore
from multiprocessing import Pool
from multiprocessing import Lock
from numpy import ma

data=numpy.random.random((3,3))
output=ma.ones(data.shape)*1e20


sema=BoundedSemaphore(5)

import time

p = multiprocessing.Process(target=time.sleep, args=(1000,))


def ff(i,j):
    #l.acquire()
    output[i,j,]=data.mean()+data[i,j]
    #l.release()
    return

lock = Lock()
data[i,j],output[i,j]
pool = Pool(processes=4)
pool.map(ff, [range(10),range(10)])
ff(i,j,lock)
data[i,j],output[i,j]

def f(x):
    from numpy.random import random
    I,J=d.shape
    tmp=0
    for i in range(I):
        for j in range(J):
	    tmp=tmp+data[i,j]
    tmp=tmp/(I*J)
    return x,tmp


pool = Pool(processes=4)
print pool.map(f, range(10))




def semaphore_func(sema, mutex, running):
    sema.acquire()
    mutex.acquire()
    running.value += 1
    print running.value, 'tasks are running'
    mutex.release()
    random.seed()
    time.sleep(random.random()*2)
    mutex.acquire()
    running.value -= 1
    print '%s has finished' % multiprocessing.current_process()
    mutex.release()
    sema.release()

def test_semaphore():
    sema = multiprocessing.Semaphore(3)
    mutex = multiprocessing.RLock()
    running = multiprocessing.Value('i', 0)
    processes = [
        multiprocessing.Process(target=semaphore_func,
                                args=(sema, mutex, running))
        for i in range(10)
        ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def test_semaphore2():
    pool = multiprocessing.Pool(4)



#from multiprocessing import Pool
#pool = Pool(processes=4)

import multiprocessing
from numpy import random
import time


#def semaphore_func(sema, mutex, running):
def semaphore_func(running):
    #sema.acquire()
    #mutex.acquire()
    running.value += 1
    print running.value, 'tasks are running'
    #mutex.release()
    random.seed()
    time.sleep(random.random()*5)
    #mutex.acquire()
    running.value -= 1
    print '%s has finished' % multiprocessing.current_process().name
    #mutex.release()
    #sema.release()
    return


import multiprocessing
from numpy import random
import time

def bunda(name):
    print "value: ",name
    random.seed()
    time.sleep(random.random()*5)
    print '%s has finished' % multiprocessing.current_process().name
    return



PROCESSES = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=PROCESSES)
map_it = pool.map(bunda, range(10))




def test_semaphore():
    sema = multiprocessing.Semaphore(3)
    #mutex = multiprocessing.RLock()
    running = multiprocessing.Value('i', 0)
    processes = [
        multiprocessing.Process(target=semaphore_func,
                                args=(sema, running))
        for i in range(10)
        ]
    for p in processes:
        p.start()
    print "Done"
    return

test_semaphore()


sema = multiprocessing.Semaphore(3)

