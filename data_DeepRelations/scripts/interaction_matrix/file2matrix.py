#coding:utf-8
import re
import random
import os
import sys
import csv
import numpy as np
import subprocess
import rdkit.Chem.rdPartialCharges as rdPartialCharges
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem, ChemicalFeatures, rdchem
from rdkit import RDConfig

from networkx.algorithms import isomorphism
import networkx as nx

atom_H = ["D","H"]
def interaction_chain(interaction_file):
	with open(interaction_file) as f:
		for line in f:
			line = re.split(r'[ ]+', line.strip())
			if line[0] == '1.':
				return line[5]
	return "error"

def read_graph(smile):
	mol = Chem.MolFromSmiles(smile)
	Chem.SanitizeMol(mol)
	atom_list = []
	for a in mol.GetAtoms():
		m = a.GetSymbol()
		atom_list.append(m)

	adja_mat = Chem.GetAdjacencyMatrix(mol)
	return atom_list,adja_mat

def create_graph_from_adj(adj):
	G=nx.Graph()
	count = 0
	for i in range(len(adj)):
		for j in range(len(adj)):
			if adj[i][j] == 1:
				count += 1
				G.add_edge(i,j)
	return G,count

def map_pdb_smi(pdb_file, smile, chain, ligand):
	atom_list_pdb = []
	index_pdb_list = {}
	map_dic = {}
	list_index = 0
	with open(pdb_file) as f:
		for line in f:
			if line[0:6] == 'HETATM':
				atom = line[12:16].strip()
				if atom[0] in atom_H or line[77] in atom_H:
					continue
				current_ligand = line[17:20].strip()
				current_chain = line[21].strip()
				pdb_index = line[6:11].strip()
				if current_chain == chain and current_ligand == ligand:
					if atom not in atom_list_pdb:
						atom_list_pdb.append(atom)
						index_pdb_list[pdb_index] = list_index
						list_index += 1
			if line[0:6] == 'CONECT':
				temp_list = re.split(r'[ ]+', line[6:].strip())
				if len(temp_list[0]) >= 5:
					line = line[6:].strip()
					temp_list = [line[i:i+5] for i in range(0,len(line),5)]
				current_index = temp_list[0]
				if current_index in index_pdb_list:
					map_dic[current_index] = []
					for temp_index in temp_list:
						if temp_index in index_pdb_list and temp_index != current_index:
							map_dic[current_index].append(temp_index)

	adjacency_matrix = np.zeros((len(atom_list_pdb), len(atom_list_pdb)),dtype=int)
	for start_index in map_dic:
		for map_index in map_dic[start_index]:
			adjacency_matrix[index_pdb_list[start_index]][index_pdb_list[map_index]] = 1
	atom_list_smile,adja_mat_smile = read_graph(smile)
	if len(atom_list_smile) != len(atom_list_pdb):
		print(pdb_file, chain, ligand, atom_list_smile, atom_list_pdb, len(atom_list_smile), len(atom_list_pdb))
	graph_pdb, count_pdb = create_graph_from_adj(adjacency_matrix)
	graph_smile, count_smile = create_graph_from_adj(adja_mat_smile)
	GM = isomorphism.GraphMatcher(graph_pdb,graph_smile)
	if not GM.is_isomorphic():
		if count_pdb != count_smile:
			GM = "count not equal"
		else:
			print(pdb_file,ligand,smile)
			print(np.array(adjacency_matrix))
			print(np.array(adja_mat_smile))
			quit()
	return GM, atom_list_pdb, atom_list_smile

def interaction_chain_pdb(interaction_file):
	chain = ''
	comp_chain = ''
	pdb = ''
	chain_count = {}
	with open(interaction_file) as f:
		for line in f:
			line_split = re.split(r'[ ]+', line.strip())
			if re.match(r'[0-9.]+',line_split[0]):
				prot_chain = line[28:31].strip()
				comp_chain = line[59:62].strip()
				if prot_chain in chain_count:
					chain_count[prot_chain][1] += 1
				else:
					chain_count[prot_chain] = [comp_chain,1]
			elif line_split[0] == "PDB" and line_split[1] == 'code:':
				pdb = line_split[2]
	if len(chain_count) != 0 and pdb != '':
		prot_chain = max(chain_count,key=lambda x: chain_count[x][1])
		comp_chain = chain_count[prot_chain][0]
		return prot_chain, comp_chain, pdb
	return "error"

def write_atom_matrix(contact_list, atom_list, seq, output):
	atom_matrix = np.zeros((len(seq),len(atom_list)),dtype=int)
	for contact in contact_list:
		try:
			atom_matrix[contact[0]][contact[1]] = 1
		except:
			print(contact[0],contact[1],atom_list,output)
			#quit()
			break
	w = open(output, 'w+')
	np.savetxt(w,atom_matrix,fmt='%.0f')
	w.close()


def write_char_matrix(contact_list, atom_list, seq, smile, output):
	char_list = list(smile)
	atom_matrix = np.zeros((len(seq),len(char_list)),dtype=int)
	atom_char_map = {}
	i = 0
	j = 0
	# print(atom_list)
	# print(smile)
	while i < len(smile):
		if re.match(r'[A-GI-Za-gi-z]', char_list[i]):
			length = len(atom_list[j])
			atom_char_map[j] = list(range(i, i+length))
			i += length
			j += 1
		else:
			i += 1

	for contact in contact_list:
		for i in atom_char_map[contact[1]]:
			atom_matrix[contact[0]][i] = 1
	w = open(output, 'w+')
	np.savetxt(w,atom_matrix,fmt='%.0f')
	w.close()

def main():
	os.system("mkdir interaction_matrix")
	os.system("mkdir interaction_matrix/whole_matrix")
	os.system("mkdir interaction_matrix/hydrogen_matrix")
	os.system("mkdir interaction_matrix/nonbonded_matrix")
	het_smile = {}
	uid_seq = {}
	not_equal_count = 0
	with open('uid_het_cid_useq_smi_ki') as f:
		for line in f:
			if line.strip() != '':
				line = line.strip().split('\t')
				uid = line[0]
				het = line[1]
				smile = line[4]
				seq = line[3]
				if het != '':
					het_smile[het] = smile
				uid_seq[uid] = seq


	for file_name in os.listdir('interaction_shifted'):
		hydr_list = []
		nonb_list = []
		try:
			prot_chain,comp_chain, pdb = interaction_chain_pdb("interaction_shifted/" + file_name)
		except:
			print(file_name)
			quit()
		with open("interaction_shifted/" + file_name) as f:
			s = file_name.split('_')
			uid = s[0]
			het = s[1]
			seq = uid_seq[uid]
			if het in het_smile:
				smile = het_smile[het]
				pdb_file = 'pdb/' + pdb + '.pdb'
				gm, atom_list_pdb, atom_list_smile = map_pdb_smi(pdb_file, smile, comp_chain, het)
				if gm != "count not equal" and gm.is_isomorphic() and len(atom_list_pdb) == len(atom_list_smile):
					mapping = gm.mapping
					i = 0
					atom_index_pdb = {}
					for atom in atom_list_pdb:
						atom_index_pdb[atom] = i
						i += 1
					mark = 0
					for line in f:
						s = re.split(r'[ ]+', line.strip())
						if s[0] == 'Hydrogen':
							mark = 1
						elif s[0] == 'Non-bonded':
							mark = 2
						elif re.match(r'[0-9.]+', s[0]) and line[29] == prot_chain:
							res_num = line[21:25].strip()
							atom_name = line[43:48].strip()
							try :
								smile_index = mapping[atom_index_pdb[atom_name]]
							except:
								continue
							if mark == 1:
								if [int(res_num), smile_index] not in hydr_list:
									hydr_list.append([int(res_num), smile_index])
							elif mark == 2:
								if [int(res_num), smile_index] not in nonb_list:
									nonb_list.append([int(res_num), smile_index])
							else:
								print("file error!!")
					write_atom_matrix(hydr_list, atom_list_smile, seq, "interaction_matrix/hydrogen_matrix/" + file_name + '.atom')
					write_atom_matrix(nonb_list, atom_list_smile, seq, "interaction_matrix/nonbonded_matrix/" + file_name + '.atom')
					write_atom_matrix(hydr_list + nonb_list, atom_list_smile, seq, "interaction_matrix/whole_matrix/" + file_name + '.atom')
					write_char_matrix(hydr_list, atom_list_smile, seq, smile, "interaction_matrix/hydrogen_matrix/" + file_name + '.char')
					write_char_matrix(nonb_list, atom_list_smile, seq, smile, "interaction_matrix/nonbonded_matrix/" + file_name + '.char')
					write_char_matrix(hydr_list + nonb_list, atom_list_smile, seq, smile, "interaction_matrix/whole_matrix/" + file_name + '.char')
				else:
					if gm != "count not equal":
						print('graph is not isomorphic: ' +pdb_file, prot_chain, comp_chain, het, smile)
					else:
						print(file_name)
						not_equal_count += 1
	print(not_equal_count)


if __name__ == '__main__':
	main()
