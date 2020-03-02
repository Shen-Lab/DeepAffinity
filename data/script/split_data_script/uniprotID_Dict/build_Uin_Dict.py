#coding:utf-8
from __future__ import division
import re
import csv
import sys
import urllib2
from subprocess import call

def main():
    uniClass = ''
    mark = 0
    GPCRM = 0
    kinaseM = 0
    ERM = 0
    channelM = 0
    kinaseDic = set()
    GPCRDic = set()
    channelDic = set()
    ERDic = set()
    ERgene = ['ESR1', 'ESR2']
    with open('channel_ID') as f:
        for line in f:
            channelDic.add(line.strip())
    with open('ER_ID') as f:
        for line in f:
            ERDic.add(line.strip())
    with open('kinase_ID') as f:
        for line in f:
            kinaseDic.add(line.strip())
    with open('GPCR_ID') as f:
        for line in f:
            GPCRDic.add(line.strip())
    with open('uniID_keywords.tsv') as tsvfile, open('uniID_class','w+') as w:
        reader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
        header = reader.next()
        # writer = csv.writer(tsvout, dialect='excel-tab')
        # writer.writerow(header)
        for row in reader:
            uniID = row[0]
            gene = row[1].upper()
            goID = row[4].split('|')
            for key in goID:
                key = key.strip()
                if key in GPCRDic:
                    GPCRM = 1
                    if uniClass != 'GPCR':
                        uniClass = 'GPCR'
                        mark += 1
                if key in ERDic and gene in ERgene:
                    ERM = 1
                    if uniClass != 'ER':
                        uniClass = 'ER'
                        mark += 1
                if key in channelDic:
                    channelM = 1
                    if uniClass != 'channel':
                        uniClass = 'channel'
                        mark += 1
                if key in kinaseDic:
                    kinaseM = 1
                    if uniClass != 'kinase':
                        uniClass = 'kinase'
                        mark += 1
                    
            if mark == 0:
                uniClass = 'others'
            if mark >= 2:
                mark = 0
                kwName = row[3]
                if 'G-protein coupled receptor' in kwName:
                    uniClass = 'GPCR'
                    mark += 1
                if 'Tyrosine-protein kinase' in kwName:
                    uniClass = 'kinase'
                    mark += 1
                if 'Ion channel' in kwName:
                    uniClass = 'channel'
                    mark += 1
            if mark >= 2:
                print(row)
            else:
                w.write(uniID + '\n' + uniClass + '\n')
            mark = 0
            GPCRM = 0
            kinaseM = 0
            ERM = 0
            channelM = 0
            uinClass = ''
            secondClass = ''
            lastID = ''





if __name__ == '__main__':
    main()