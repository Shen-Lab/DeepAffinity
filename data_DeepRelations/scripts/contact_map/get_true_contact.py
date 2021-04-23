import Bio.PDB
import numpy as np
import os
import re
from functools import cmp_to_key

pdb2single = {"GLY":"G","ALA":"A","SER":"S","THR":"T","CYS":"C","VAL":"V","LEU":"L","ILE":"I","MET":"M","PRO":"P",\
"PHE":"F","TYR":"Y","TRP":"W","ASP":"D","GLU":"E","ASN":"N","GLN":"Q","HIS":"H","LYS":"K","ARG":"R","UNK":"X",\
 "SEC":"U","PYL":"O","MSE":"M","CAS":"C","SGB":"S","CGA":"E","TRQ":"W","TPO":"T","SEP":"S","CME":"C","FT6":"W","OCS":"C","SUN":"S","SXE":"S"}


def cal_index_diff(current_index,last_index):
    if re.search(r'[a-zA-Z]',current_index):
        current_num = int(current_index[:-1])
    else:
        current_num = int(current_index)
    if re.search(r'[a-zA-Z]',last_index):
        last_num = int(last_index[:-1])
    else:
        last_num = int(last_index)
    if abs(current_num - last_num) <= 1:
        return 1
    else:
        return abs(current_num - last_num)


def align_seq(seq_p,seq_p_index, seq_u):
    # return the start index of seq_p in seq_u. if seq_p starts from the beginning, it will be 1.
    pdb_uniprot_mapping = {}
    if len(seq_p) <= 30:
        print(seq_p)
        return "error"
    i = 0
    while i <= len(seq_p)-18:
        sub_seq = seq_p[i:i+18]
        index = seq_u.find(sub_seq)
        if index != -1:
            current_index_pdb = i
            last_index_pdb = i
            current_index_uniprot = index
            pdb_uniprot_mapping[seq_p_index[current_index_pdb]] = str(current_index_uniprot)  # index of string, start from 0
            while current_index_pdb > 0:
                current_index_pdb -= 1
                diff = cal_index_diff(seq_p_index[current_index_pdb], seq_p_index[last_index_pdb])
                if diff >= 2:
                    sub_seq = seq_p[current_index_pdb - 9:current_index_pdb + 1]
                    index_temp = seq_u[:current_index_uniprot].find(sub_seq)
                    if index_temp != -1:
                        current_index_uniprot = index_temp + 9
                    else:
                        index_temp = seq_u.find(sub_seq)
                        if index_temp != -1:
                            current_index_uniprot = index_temp + 9
                        else:
                            current_index_uniprot -= 1
                else:
                    current_index_uniprot -= diff
                pdb_uniprot_mapping[seq_p_index[current_index_pdb]] = str(current_index_uniprot)
                last_index_pdb = current_index_pdb

            current_index_pdb = i
            last_index_pdb = i
            current_index_uniprot = index
            while current_index_pdb < len(seq_p) - 1:
                current_index_pdb += 1
                diff = cal_index_diff(seq_p_index[current_index_pdb], seq_p_index[last_index_pdb])
                if diff >= 2:
                    try:
                        sub_seq = seq_p[current_index_pdb:current_index_pdb + 10]
                    except:
                        sub_seq = seq_p[current_index_pdb:]
                    index_temp = seq_u[current_index_uniprot + 1:].find(sub_seq)
                    if index_temp != -1:
                        current_index_uniprot += index_temp + 1
                    else:
                        index_temp = seq_u.find(sub_seq)
                        if index_temp != -1:
                            current_index_uniprot = index_temp
                        else:
                            current_index_uniprot += 1
                else:
                    current_index_uniprot += diff
                pdb_uniprot_mapping[seq_p_index[current_index_pdb]] = str(current_index_uniprot)
                last_index_pdb = current_index_pdb
            return pdb_uniprot_mapping
        i+=1
    print(seq_p,seq_p_index, seq_u)
    return "alignment error"


def interaction_chain(interaction_file):
    m = 0
    n = 0
    # site = set()
    with open(interaction_file) as f:
        for line in f:
            old_line = line
            line = re.split(r'[ ]+', line.strip())
            if re.match(r'[0-9]+\.', line[0]) and m == 0:
                chain = line[5]
                m = 1
            elif line[0] == 'PDB' and line[1] == 'code:':
                pid = line[2]
                n = 1
            # if re.match("[0-9]+\.",line[0]):
            #     site.add(int(line[4]))
    if m == 1 and n == 1:
        return chain, pid
    else:
        print ("Error on:",interaction_file)
        return "error"

def get_icode(current_index):
    if re.search(r'[a-zA-Z]',current_index):
        icode = current_index[-1]
        index = int(current_index[:-1])
    else:
        icode = ' '
        index = int(current_index)
    return index, icode


def comp_index(index1,index2):
    index1 = index1[0]
    index2 = index2[0]
    if re.search(r'[A-Z]+',index1):
        v1 = int(index1[:-1]) + 0.1 * (ord(index1[-1]) - ord('A') + 1)
    else:
        v1 = int(index1)
    if re.search(r'[A-Z]+',index2):
        v2 = int(index2[:-1]) + 0.1 * (ord(index2[-1]) - ord('A') + 1)
    else:
        v2 = int(index2)
    return v1 - v2

def pdb2seq(pdbfile):
    seq = {}
    string = ''
    last_index = 0
    last_chain = ''
    seq_index = []
    with open(pdbfile) as f:
        for line in f:
            if line[0:4] == "ATOM" or line[0:6] == "HETATM" and line[17:20].strip() in pdb2single:
                current_chain = line[21]
                long_aa = line[17:20].strip()
                try:
                    aa = pdb2single[long_aa]
                except:
                    print(pdbfile,line)
                    continue
                index = line[22:27].strip()
                if last_chain == current_chain:
                    if index != last_index:
                        string += aa
                        seq_index.append(index)
                        last_index = index
                else:
                    if last_chain != '':
                        if last_chain not in seq:
                            index_seq = sorted(zip(seq_index,list(string)),key = cmp_to_key(comp_index))
                            string = ''.join(x[1] for x in index_seq)
                            seq_index = [x[0] for x in index_seq]
                            seq[last_chain] = [string,seq_index]
                    string = aa
                    seq_index = [index]
                    last_index = index
                    last_chain = current_chain
    if current_chain not in seq:
        index_seq = sorted(zip(seq_index,list(string)),key = cmp_to_key(comp_index))
        string = ''.join(x[1] for x in index_seq)
        seq_index = [x[0] for x in index_seq]
        seq[current_chain] = [string,seq_index]
    return seq


inter_file = set()
cutoff = 8
result_dir = "./contact_matrix_true/"
for name in os.listdir("./split_data"):
    if "inter" in name:
        temp = name.split('_')
        seq_file = '_'.join(temp[:-1]) + '_seq'
        with open("./split_data/" + name) as f, open("./split_data/" + seq_file) as s:
            for line in f:
                seq = next(s).strip()
                inter_file.add(line.strip() + '\t' + seq)

for name in inter_file:
    name = name.split('\t')
    seq_uniprot = name[1]
    inter_name = name[0]
    uid = inter_name.split('_')[0]
    chain, pid = interaction_chain("./interaction_shifted/" + inter_name)
    seq_pdb_list = pdb2seq('./pdb/' + pid + ".pdb")
    structure = Bio.PDB.PDBParser().get_structure(pid, './pdb/' + pid + ".pdb")
    model = structure[0]
    seq_pdb = seq_pdb_list[chain][0]
    seq_pdb_index = seq_pdb_list[chain][1]
    pdb_uniprot_mapping = align_seq(seq_pdb,seq_pdb_index, seq_uniprot)
    contact_matrix = np.zeros([len(seq_uniprot),len(seq_uniprot)])
    pdb_uniprot_mapping_exist = {}
    for current_index in pdb_uniprot_mapping:
        index, icode = get_icode(current_index)
        if index in structure[0][chain]:
            pdb_uniprot_mapping_exist[current_index] = pdb_uniprot_mapping[current_index]

    for current_index1 in pdb_uniprot_mapping_exist:
        current_index_u1 = int(pdb_uniprot_mapping_exist[current_index1])
        current_index1, icode1 = get_icode(current_index1)
        for current_index2 in pdb_uniprot_mapping_exist:
            current_index_u2 = int(pdb_uniprot_mapping_exist[current_index2])
            current_index2, icode2 = get_icode(current_index2)
            
            if "CB" in model[chain][' ', current_index1, icode1]:
                l1 = "CB"
            else:
                l1 = "CA"
            if "CB" in model[chain][' ', current_index2, icode2]:
                l2 = "CB"
            else:
                l2 = "CA"
            try:
                if model[chain][' ', current_index1, icode1][l1] - model[chain][' ', current_index2, icode2][l2] <= cutoff:
                    contact_matrix[current_index_u1][current_index_u2] = 1
            except:
                continue
    np.savetxt(result_dir + uid + "_contactmap.txt",contact_matrix,fmt='%1d')

