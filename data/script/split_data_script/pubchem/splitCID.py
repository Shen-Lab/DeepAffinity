#coding:utf-8
from __future__ import division
import re
import pybel
import csv
import sys
from subprocess import call

def main():
    count = 0
    fileC = 1
    i = 0
    with open('uniqueCID') as f:
        fileName = 'uniqueCID' + str(fileC)
        w = open(fileName,'w+')
        for row in f:
            w.write(row)
            i += 1
            if i > 200000:
                w.close()
                fileC += 1
                fileName = 'uniqueCID' + str(fileC)
                w = open(fileName,'w+')
                i = 0
            count += 1
    print("cid number: %d" %count)
    # print total





if __name__ == '__main__':
    main()