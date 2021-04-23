#coding:utf-8
import re
import random
import os
import sys
import csv
import numpy as np
import subprocess
from functools import cmp_to_key
pdb2single = {"GLY":"G","ALA":"A","SER":"S","THR":"T","CYS":"C","VAL":"V","LEU":"L","ILE":"I","MET":"M","PRO":"P",\
"PHE":"F","TYR":"Y","TRP":"W","ASP":"D","GLU":"E","ASN":"N","GLN":"Q","HIS":"H","LYS":"K","ARG":"R", "UNK":"X",\
 "SEC":"U","PYL":"O","MSE":"M","CAS":"C","SGB":"S","CGA":"E","TRQ":"W","TPO":"T","SEP":"S","CME":"C","FT6":"W","OCS":"C","SUN":"S","SXE":"S"}
DNA_list = ['DA','DT','DC','DG','DI','DU','A','C','G','I','U']

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


def align_seq(seq_p, seq_p_index, seq_u):
	# return the start index of seq_p in seq_u. if seq_p starts from the beginning, it will be 1.
	pdb_uniprot_mapping = {}
	if len(seq_p) <= 30:
		# print(seq_p)
		return "alignment error"
	i = 0
	while i <= len(seq_p)-18:
		sub_seq = seq_p[i:i+18]
		index = seq_u.find(sub_seq)
		if index != -1:
			current_index_pdb = i
			last_index_pdb = i
			current_index_uniprot = index
			pdb_uniprot_mapping[seq_p_index[current_index_pdb]] = str(current_index_uniprot) #index of string, start from 0
			while current_index_pdb > 0:
				current_index_pdb -= 1
				diff = cal_index_diff(seq_p_index[current_index_pdb], seq_p_index[last_index_pdb])
				if diff >= 2:
					sub_seq = seq_p[current_index_pdb-9:current_index_pdb+1]
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
						sub_seq = seq_p[current_index_pdb:current_index_pdb+10]
					except:
						sub_seq = seq_p[current_index_pdb:]
					index_temp = seq_u[current_index_uniprot+1:].find(sub_seq)
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
	return "alignment error"



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
	error_chain = set()
	seq_index = []
	with open(pdbfile) as f:
		for line in f:
			if line[0:4] == "ATOM" or line[0:6] == "HETATM" and line[17:20] in pdb2single:
				current_chain = line[21]
				long_aa = line[17:20]
				try:
					aa = pdb2single[long_aa]
				except:
					if long_aa.strip() not in DNA_list:
						error_chain.add(current_chain)
					continue
				index = line[22:27].strip()
				if last_chain == current_chain:
					if index != last_index:
						string += aa
						seq_index.append(index)
						last_index = index
				else:
					if last_chain != '': 
						if current_chain not in error_chain and last_chain not in seq:
							index_seq = sorted(zip(seq_index,list(string)),key = cmp_to_key(comp_index))
							string = ''.join(x[1] for x in index_seq)
							seq_index = [x[0] for x in index_seq]
							seq[last_chain] = [string,seq_index]
						else:
							print(pdbfile + current_chain)
					string = aa
					seq_index = [index]
					last_index = index
					last_chain = current_chain
	if current_chain not in error_chain and current_chain not in seq:
		index_seq = sorted(zip(seq_index,list(string)),key = cmp_to_key(comp_index))
		string = ''.join(x[1] for x in index_seq)
		seq_index = [x[0] for x in index_seq]
		seq[current_chain] = [string,seq_index]
	else:
		print(pdbfile + current_chain)
	return seq

def interaction_chain(interaction_file):
	chain_num = {}
	with open(interaction_file) as f:
		for line in f:
			line = re.split(r'[ ]+', line.strip())
			if re.match(r'[0-9]+\.', line[0]):
				chain = line[5]
				if chain in chain_num:
					chain_num[chain] += 1
				else:
					chain_num[chain] = 1
	if len(chain_num) != 0:
		return max(chain_num, key = (lambda x: chain_num[x]))
	else:
		return "error"

def count_valid_lines(interaction_file, chain):
	count = 0
	ligand_chain = ''
	with open(interaction_file) as f:
		for line in f:
			start = re.split(r'[ ]+', line)[0]
			if re.match(r'[0-9.]+', start) and line[29] == chain:
				if ligand_chain == '':
					ligand_chain = re.split(r'[ ]', line.strip())[11]
				count += 1
	return count, ligand_chain
				


def main():
	uid_pid = {}
	uid_seq = {}
	uid_het = {}
	pid_het = {}
	pid_het1 = {}
	pid_het_new = {}
	dir_path = "./"
	os.system('mkdir interaction_shifted')

	with open('pid_uid') as f:
		for line in f:
			s = line.strip().split('\t')
			pid = s[0]
			uid = s[1]
			if uid in uid_pid:
				uid_pid[uid].add(pid)
			else:
				uid_pid[uid] = {pid}


	with open("uid_het_cid_useq_smi_ki") as f:
		for line in f:
			line = line.strip().split('\t')
			uid = line[0]
			het = line[1]
			seq = line[3]
			if uid in uid_het:
				uid_het[uid].add(het)
			else:
				uid_het[uid] = {het}
			if len(seq) <= 30:
				print(uid,het,seq)
				quit()
			uid_seq[uid] = seq

	for name in os.listdir('./contact'):
		s = name.split('_')
		pid = s[0]
		het = s[1]
		if pid in pid_het_new:
			if het in pid_het_new[pid]:
				pid_het_new[pid][het].append(name)
			else:
				pid_het_new[pid][het] = [name]
		else:
			pid_het_new[pid] = {het:[name]}
	print('Dictionary Loaded')

	wrong_index = set()
	for uid_list in uid_seq.keys():
		full_uid = uid_list
		uid_list = uid_list.split(',')
		for uid in uid_list:
			if uid in uid_pid:
				pid_list = uid_pid[uid]
			else:
				continue
			if uid in uid_het:
				het_list = uid_het[uid]
			else:
				continue
			seq_uniprot = uid_seq[uid]
			for pid in pid_list:
				if pid in pid_het_new:
					het_file = pid_het_new[pid]
					contact_path = dir_path + 'contact/'
				else:
					continue
				
				for het in het_list:
					if het in het_file:
						seq_pdb_list = pdb2seq("./pdb/" + pid + ".pdb")
						satisfied_chain_index = {}
						for chain in seq_pdb_list:
							seq_pdb = seq_pdb_list[chain][0]
							seq_pdb_index = seq_pdb_list[chain][1]
							pdb_uniprot_mapping = align_seq(seq_pdb,seq_pdb_index, seq_uniprot)
							if pdb_uniprot_mapping != "alignment error":
								satisfied_chain_index[chain] = pdb_uniprot_mapping
						for file_name in het_file[het]:
							interact_chain = interaction_chain(contact_path + file_name)
							if interact_chain == "error":
								print(file_name)
							elif interact_chain in satisfied_chain_index:
								pdb_uniprot_mapping = satisfied_chain_index[interact_chain]
								new_file = full_uid + '_' + het
								file_content = ''
								with open(contact_path + file_name) as f:
									for line in f:
										line_split = re.split(r'[ ]+', line.strip())
										if len(line_split) != 1 and re.match(r'[0-9.]+', line_split[0]):
											current_index = line_split[4]
											current_chain = line_split[5]
											if current_chain == interact_chain:
												try:
													res_index = pdb_uniprot_mapping[current_index]
												except:
													print(file_name, pid, het, interact_chain, current_index, pdb_uniprot_mapping,seq_pdb_list)
													quit()
											else:
												continue
											if re.search(r'[a-zA-Z]',current_index):
												original_length = len(line_split[4]) - 1
											else:
												original_length = len(line_split[4])
											new_length = len(res_index)
											line = list(line)
											line[25] = ' '
											if new_length >= original_length:
												for i in range(new_length):
													line[25-new_length+i] = res_index[i]
						
											else:
												dif = original_length - new_length
												for i in range(dif):
													line[25-original_length+i] = ' '
												for i in range(new_length):
													line[25-new_length+i] = res_index[i]
											line = ''.join(line)
										file_content += line
									w = open('temp', 'w+')
									w.write(file_content)
									w.close()
									if new_file not in os.listdir("interaction_shifted"): 
										os.system('cp temp' + ' interaction_shifted/' + new_file)
									else:
										old_count = count_valid_lines('interaction_shifted/' + new_file, interact_chain)
										new_count = count_valid_lines('temp', interact_chain)
										if new_count > old_count:
											os.system('cp temp' + ' interaction_shifted/' + new_file)
									os.system("rm temp")
if __name__ == '__main__':
	main()
