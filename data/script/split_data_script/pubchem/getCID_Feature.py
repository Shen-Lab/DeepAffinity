#coding:utf-8
from __future__ import division
import re
import csv
import sys
from subprocess import call

str2bin = {'0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61, '+': 62, '/': 63,
           'A': 0, 'C': 2, 'B': 1, 'E': 4, 'D': 3, 'G': 6, 'F': 5, 'I': 8, 'H': 7, 'K': 10, 'J': 9,
           'M': 12, 'L': 11, 'O': 14, 'N': 13, 'Q': 16, 'P': 15, 'S': 18, 'R': 17, 'U': 20, 'T': 19,
           'W': 22, 'V': 21, 'Y': 24, 'X': 23, 'Z': 25, 'a': 26, 'c': 28, 'b': 27, 'e': 30, 'd': 29,
           'g': 32, 'f': 31, 'i': 34, 'h': 33, 'k': 36, 'j': 35, 'm': 38, 'l': 37, 'o': 40, 'n': 39,
           'q': 42, 'p': 41, 's': 44, 'r': 43, 'u': 46, 't': 45, 'w': 48, 'v': 47, 'y': 50, 'x': 49, 'z': 51}
def decode64(string):
    result = ''
    for c in string:
        if c == '=':
            result = result[:-2]
        elif c == ' ':
            continue
        elif c == '\n':
            continue
        else:
            temp = '{:b}'.format(str2bin[c])
            temp = (6-len(temp)) * '0' + temp
            result += temp
    return result[32:-7]

def main():
    fileNum = 3
    count = 0
    fileC = 1
    i = 0
    w = open("CID_Smi_Feature", 'w+')
    for i in range(1,fileNum+1):
        resultFile = 'CID' + str(i) + '.sdf'
        with open(resultFile) as f:
            storeStr = ''
            for line in f:
                if line.strip() == '> <PUBCHEM_COMPOUND_CID>':
                    storeStr += line
                    line = f.next()
                    storeStr += line
                elif line.strip() == '> <PUBCHEM_CACTVS_SUBSKEYS>':
                    storeStr += line
                    key = f.next().strip()
                    key = decode64(key)
                    storeStr += key + '\n'
                elif line.strip() == '> <PUBCHEM_OPENEYE_CAN_SMILES>':
                    storeStr += line
                    line = f.next()
                    storeStr += line
                elif line.strip() == '$$$$':
                    storeStr += line + '\n'
                    w.write(storeStr)
                    storeStr = ''
     # print total





if __name__ == '__main__':
    main()