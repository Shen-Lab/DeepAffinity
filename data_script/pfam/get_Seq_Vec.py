import re
import csv
# import numpy as np
total_file_number = 7  #the total number of splited files
pf_index = {}
i = 0
with open('Pfam-A.clans.tsv') as f:
    reader = csv.reader(f, delimiter="\t",quoting=csv.QUOTE_NONE)
    for line in reader:
        a = line[0].strip()
        pf_index[a] = i
        i += 1

w1 = open("noResult",'w+')
reResult = r'[\w_.]+[ ]{3,}(PF[0-9]+)[0-9.]+[ ]+([0-9]+)'
w = open(r'Seq_Vector','w+')
num = 0
id_seq = {}
id_pf = {}
fileName = "uniqueProtein.fasta"
fileName_result = "uniqueProtein.output"
with open(fileName) as f1, open(fileName_result) as f2:
    for line in f1:
        if '>' in line:
            index = re.sub(r'>','',line).strip()
        else:
            seq = line.strip()
            if index in id_seq:
                print 'error'
            id_seq[index] = seq

    for line in f2:
        if '#'in line:
            continue
        else:
            index = re.search(reResult,line).group(2).strip()
            pf = re.search(reResult,line).group(1).strip()
            if "PF" not in pf:
                print line
            if index in id_pf:
                id_pf[index].add(pf)
            else:
                id_pf[index] = set()
                id_pf[index].add(pf)
for index in id_seq:
    pfString = ''
    if index in id_pf:
        # print id_pf[index]
        for pf in id_pf[index]:
            pfString += str(pf_index[pf]) + ' '
    else:
        # print "no pf for %s" %id_seq[index]
        w1.write('>'+index+'\n'+id_seq[index]+'\n')
    w.write(id_seq[index] + '\n' + pfString + '\n')
w.close()
w1.close()
print 'Success!'