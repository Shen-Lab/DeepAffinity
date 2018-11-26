#coding:utf-8
from __future__ import division
import re
import csv
import sys
from subprocess import call

def main():
    # i = 0
    uniID = set()
    count = 0
    noCount = 0
    noUniName = 0
    with open('../BindingDB_All_firststep_noMulti_can.tsv') as tsvfile, open('Uniprot_ID','w+') as w:
        reader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
        header = reader.next()
        for row in reader:
            uid = row[41].strip()
            if uid == '':
                noCount += 1
            elif uid not in uniID:
                uniID.add(uid)
                w.write(uid+'\n')
                count += 1




if __name__ == '__main__':
    main()
