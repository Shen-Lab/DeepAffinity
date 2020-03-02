#coding:utf-8
from __future__ import division
import re
import pybel

def main():
    i = 0
    num = 0
    fileC = 0
    # w = open('Seq'+str(fileC),'w+')
    w = open('uniqueProtein.fasta','w+')
    with open('uniqueProtein') as f:
        for line in f:
            if line.strip() == '':
                print('error')
                
            else:
                w.write('>'+str(num)+'\n')
                w.write(line.replace(' ',''))
                num += 1
            # i += 1
            # if i >= 1000:
            #     fileC += 1
            #     w.close()
            #     w = open('Seq'+str(fileC),'w+')
            #     i = 0
    print(num)






if __name__ == '__main__':
    main()