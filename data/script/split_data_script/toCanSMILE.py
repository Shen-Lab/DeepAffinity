#coding:utf-8
from __future__ import division
import re
import pybel
import csv
import sys
from subprocess import call

def main():
    # i = 0
    countDrop = 0
    dropM = 0
    total = 0
    cid_can = {}
    # total = 0
    # maxInt = sys.maxsize
    # decrement = True
    # while decrement:
    # # decrease the maxInt value by factor 10 
    # # as long as the OverflowError occurs.
    #     decrement = False
    #     try:
    #         csv.field_size_limit(maxInt)
    #     except OverflowError:
    #         maxInt = int(maxInt/10)
    #         decrement = True
    with open('CID_Smi_Feature') as f:
        for line in f:
            if line.strip() == '> <PUBCHEM_COMPOUND_CID>':
                cid = f.next().strip()
            elif line.strip() == '> <PUBCHEM_OPENEYE_CAN_SMILES>':
                can = f.next().strip()
            elif line.strip() == '$$$$':
                cid_can[cid] = can
    print("Dictionary Loaded!")
    with open('BindingDB_All_firststep_noMulti.tsv') as tsvfile, open('BindingDB_All_firststep_noMulti_can.tsv','w+') as csvout:
        reader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
        writer = csv.writer(csvout, dialect='excel-tab')
        header = reader.next()
        writer.writerow(header)
        for row in reader:
            cid = row[28]
            can = cid_can[cid]
            row[1] = can
            if len(can.strip()) > 100:
                dropM = 1 
            if dropM == 1:
                countDrop += 1
            else:
                total += 1
                writer.writerow(row)
            dropM = 0
    print(total)
    # print countDrop





if __name__ == '__main__':
    main()
