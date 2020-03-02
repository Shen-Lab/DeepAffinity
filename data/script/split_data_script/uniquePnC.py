#coding:utf-8
from __future__ import division
import re
import pybel
import csv
import sys
from subprocess import call

def main():
    countP = 0
    countC = 0
    dropM = 0
    proteinDic = set()
    cidDic = set()
    with open('BindingDB_All_firststep_noMulti.tsv') as tsvfile, open("uniqueProtein",'w+') as uniqueP, open("uniqueCID",'w+') as uniqueC:
        reader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
        # line = {}
        # k = ''
        # writerP = csv.writer(uniqueP, dialect='excel-tab')
        # writerP = csv.writer(uniqueC, dialect='excel-tab')
        # writer.writeheader()
        # print reader.fieldnames
        header = reader.next()
        # writerP.writerow(header)
        # writerC.writerow(header)
        for row in reader:
            # total += 1
            # seq = re.sub(row[37].upper())
            # if seq == None:
            #     dropM = 1
            #     none += 1
            # elif seq = 'N/A':
            #     na +=1
            #     dropM = 1
            # elif seq == '' or 'X' in seq:
            #     dropM = 1
            # if row[]
            seq = row[37].strip()
            cid = row[28].strip()
            if seq not in proteinDic:
                uniqueP.write(seq + '\n')
                proteinDic.add(seq)
                countP += 1
            if cid not in cidDic:
                uniqueC.write(cid + '\n')
                cidDic.add(cid)
                countC += 1
    print("cid number: %d" %countC)
    print("protein number: %d" %countP)
    # print total





if __name__ == '__main__':
    main()
