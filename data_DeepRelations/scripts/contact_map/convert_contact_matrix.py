import re
import random
import os
import sys
import csv
import numpy as np
from scipy import stats

def convert_matrix(file_name):
	threshold = 3.0
	matrix = []
	with open(file_name) as f:
		for line in f:
			matrix.append([float(i) for i in re.split(r'[ ]+',line.strip())])
	matrix = np.asarray(matrix)
	matrix = stats.zscore(matrix,axis=None)
	length = len(matrix)
	for i in range(length):
		for j in range(length):
			if matrix[i][j] > threshold:
				matrix[i][j] = 1
			else:
				matrix[i][j] = 0
	return matrix


def main():
	os.system("mkdir contact_matrix")
	directory = "./contact_file"
	seq_uid = {}
	id_seq = {}
	with open("final_seq.txt") as f:
		for line in f:
			if line[0] == '>':	
				id_num = line[1:].strip()
				seq = next(f).strip()
				id_seq[id_num] = seq

	with open("uid_seq") as f:
		for line in f:
			if line[0] == '>':
				uid = line[1:].strip()
				seq = next(f).strip()
				if seq not in seq_uid:
					seq_uid[seq] = [uid]
				else:
					seq_uid[seq].append(uid)

	print(len(id_seq),len(seq_uid))
	for num in id_seq.keys():
		seq = id_seq[num]
		if seq in seq_uid:
			matrix = convert_matrix(directory + "/" +  num + ".txt")
			uidlist = seq_uid[seq]
			for uid in uidlist:
				w = open("contact_matrix/" + uid + "_contactmap.txt",'w+')
				np.savetxt(w,matrix,fmt='%.0f')
				w.close()
if __name__ == '__main__':
	main()
