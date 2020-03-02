#coding:utf-8
from __future__ import division
import re
import csv
import sys
from subprocess import call

def main():
    count = 0
    countMul = 0
    dropM = 0
    pairDic = {}
    multiMDic = {}
    tempDic = {}
    with open('BindingDB_All.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
        header = reader.next()
        for row in reader:
            # total += 1
            # seq = re.sub(row[37].upper())
            # if seq == None:
            #     dropM = 1
            #     none += 1
            # ic50 = row[9]
            # ki = row[8]
            # kd = row[10]
            seq = row[37].strip()
            smi = row[1].strip()
            if seq in pairDic:
                if smi in pairDic[seq]:
                    # tempList = tempDic[smile]
                    # if ic50 != tempList[0] or ki != tempList[1] or kd != tempList[2]:
                    dropM = 1
                    # idPair += 1
                else:
                    pairDic[seq].append(smi)
            else:
                pairDic[seq] = [smi]
            if dropM == 1:
                if seq in multiMDic:
                    if smi not in multiMDic[seq]:
                        multiMDic[seq].append(smi)
                else:
                    multiMDic[seq] = [smi]
            dropM = 0
    with open('BindingDB_All.tsv') as tsvfile, open("BindingDB_All_firststep_noMulti.tsv",'w+') as csvout:
        reader = csv.reader(tsvfile, delimiter="\t",quoting=csv.QUOTE_NONE)
        # line = {}
        # k = ''
        writer = csv.writer(csvout, dialect='excel-tab')
        # mergeW = csv.writer(merge, dialect='excel-tab')
        # writer.writeheader()
        # print reader.fieldnames
        header = reader.next()
        writer.writerow(header)
        # mergeW.writerow(header)
        for row in reader:
            seq = row[37].strip()
            smi = row[1].strip()
            cid = row[28].strip()
            ic50 = row[9].strip()
            ki = row[8].strip()
            kd = row[10].strip()
            if int(row[36]) >= 2:
                dropM = 1
            elif seq == '' or seq == 'n/a' or seq == 'N/A' or seq == None or seq == 'None':
                dropM = 1
            elif 'X' in seq or 'x' in seq:
                dropM = 1
            elif row[28] == '':
                dropM = 1
            elif re.search(r'[a-z0-9\W]',seq):
                dropM = 1
            elif len(seq)>1500:
                dropM = 1
            elif seq in multiMDic:
                if smi in multiMDic[seq]:
                    dropM = 1
                    if seq in tempDic:
                        if smi in tempDic[seq]:
                            tempDic[seq][smi].append([ic50,ki,kd])
                        else:
                            tempDic[seq][smi] = [row,[ic50,ki,kd]]
                    else:
                        tempDic[seq] = {smi:[row,[ic50,ki,kd]]}
                    countMul += 1
            if dropM == 0:
                writer.writerow(row)
                count += 1 
            dropM = 0
        #print "Multi pairs: %d" %countMul
        #print "before adding multi pairs:%d" %count
        
        for seq in tempDic:
            for smi in tempDic[seq]:
                icDropM = 0
                kiDropM = 0
                kdDropM = 0
                icMean = 1
                kiMean = 1
                kdMean = 1
                icList = []
                kiList = []
                kdList = []
                icNoInList = []
                kiNoInList = []
                kdNoInList = []
                
                row = tempDic[seq][smi][0]
                i = 1
                while i < len(tempDic[seq][smi]):
                    tempList = tempDic[seq][smi][i]
                    if tempList[0] != '':
                        icList.append(tempList[0])
                        icNoInList.append(float(re.sub(r'[<>=]','',tempList[0])))
                    if tempList[1] != '':
                        kiList.append(tempList[1])
                        kiNoInList.append(float(re.sub(r'[<>=]','',tempList[1])))
                    if tempList[2] != '':
                        kdList.append(tempList[2])
                        kdNoInList.append(float(re.sub(r'[<>=]','',tempList[2])))
                    i += 1

                exactVM = 0
                inVM = 0
                power = 0
                if len(icList) > 0:
                    if min(icNoInList) != 0 and max(icNoInList)/min(icNoInList) >= 1000:
                        icDropM = 1
                    elif min(icNoInList) == 0 and max(icNoInList) >= 1000:
                        icDropM = 1
                    else:
                        for j in range(len(icList)):
                            ic = icList[j]
                            if re.search(r'[<>=]',ic):
                                value = float(re.sub(r'[<>=]','',ic))
                                if '<' in ic and value > 0.01:
                                    icList[j] = ''
                                elif '>' in ic and value < 1E7:
                                    icList[j] = ''
                                else:
                                    inVM = 1
                            else:
                                exactVM = 1
                        if inVM == 1 and exactVM == 0:
                            lesM = 0
                            larM = 0
                            for ic in icList:
                                if ic != '':
                                    power += 1
                                    if '<' in ic:
                                        lesM = 1
                                    else:
                                        larM = 1
                                    icMean *= float(re.sub(r'[><=]','',ic))
                            icMean = icMean ** (1.0/power)
                            if lesM == 1 and larM == 0:
                                icMean = '<' + str(icMean)
                            elif lesM == 0 and larM == 1:
                                icMean = '>' + str(icMean)
                            else:
                                print("Error: No Inequality or both directions")
                        elif exactVM == 1:
                            for ic in icList:
                                if ic != '' and not re.search(r'[<>=]',ic):
                                    icMean *= float(ic)
                                    power += 1
                            icMean = str(icMean ** (1.0/power))
                        else:
                            icDropM = 1
                else:
                    icDropM = 1

                exactVM = 0
                inVM = 0
                power = 0
                if len(kiList) > 0:
                    if min(kiNoInList) != 0 and max(kiNoInList)/min(kiNoInList) >= 1000:
                        kiDropM = 1
                    elif min(kiNoInList) == 0 and max(kiNoInList) >= 1000:
                        kiDropM = 1
                    else:
                        for j in range(len(kiList)):
                            ki = kiList[j]
                            if re.search(r'[<>=]',ki):
                                value = float(re.sub(r'[<>=]','',ki))
                                if '<' in ki and value > 0.01:
                                    kiList[j] = ''
                                elif '>' in ki and value < 1E7:
                                    kiList[j] = ''
                                else:
                                    inVM = 1
                            else:
                                exactVM = 1
                        if inVM == 1 and exactVM == 0:
                            lesM = 0
                            larM = 0
                            for ki in kiList:
                                if ki != '':
                                    power += 1
                                    if '<' in ki:
                                        lesM = 1
                                    else:
                                        larM = 1
                                    kiMean *= float(re.sub(r'[><=]','',ki))
                            kiMean = kiMean ** (1.0/power)
                            if lesM == 1 and larM == 0:
                                kiMean = '<' + str(kiMean)
                            elif lesM == 0 and larM == 1:
                                kiMean = '>' + str(kiMean)
                            else:
                                print("Error: No Inequality or both directions")
                        elif exactVM == 1:
                            for ki in kiList:
                                if ki != '' and not re.search(r'[<>=]',ki):
                                    kiMean *= float(ki)
                                    power += 1
                            kiMean = kiMean ** (1.0/power)
                        else:
                            kiDropM = 1
                else:
                    kiDropM = 1

                exactVM = 0
                inVM = 0
                power = 0
                if len(kdList) > 0:
                    if min(kdNoInList) != 0 and max(kdNoInList)/min(kdNoInList) >= 1000:
                        kdDropM = 1
                    elif min(kdNoInList) == 0 and max(kdNoInList) >= 1000:
                        kdDropM = 1
                    else:
                        for j in range(len(kdList)):
                            kd = kdList[j]
                            if re.search(r'[<>=]',kd):
                                value = float(re.sub(r'[<>=]','',kd))
                                if '<' in kd and value > 0.01:
                                    kdList[j] = ''
                                elif '>' in kd and value < 1E7:
                                    kdList[j] = ''
                                else:
                                    inVM = 1
                            else:
                                exactVM = 1
                        if inVM == 1 and exactVM == 0:
                            lesM = 0
                            larM = 0
                            for kd in kdList:
                                if kd != '':
                                    power += 1
                                    if '<' in kd:
                                        lesM = 1
                                    else:
                                        larM = 1
                                    kdMean *= float(re.sub(r'[><=]','',kd))
                            kdMean = kdMean ** (1.0/power)
                            if lesM == 1 and larM == 0:
                                kdMean = '<' + str(kdMean)
                            elif lesM == 0 and larM == 1:
                                kdMean = '>' + str(kdMean)
                            else:
                                print("Error: No Inequality or both directions")
                        elif exactVM == 1:
                            for kd in kdList:
                                if kd != '' and not re.search(r'[<>=]',kd):
                                    kdMean *= float(kd)
                                    power += 1
                            kdMean = str(kdMean ** (1.0/power))
                        else:
                            kdDropM = 1
                else:
                    kdDropM = 1

                if icDropM == 0:
                    row[9] = icMean
                else:
                    row[9] = ''
                if kiDropM == 0:
                    row[8] = kiMean
                else:
                    row[8] = ''
                if kdDropM == 0:
                    row[10] = kdMean
                else:
                    row[10] = ''
                writer.writerow(row)
                # mergeW.writerow(row)
                count += 1
        print("Final number: %d" %count)




if __name__ == '__main__':
    main()
