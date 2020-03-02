#coding:utf-8
import re
import random
import os
import sys
import csv
import numpy as np
def pos2vec(string):
	vector = 16712 * ['0']
	for pos in string.split():
		vector[int(pos)] = '1'
	return ''.join(str(x) for x in vector)

def main(dataDir):
	countic = 0
	countki = 0
	countkd = 0
	countec = 0
	noicC = 0
	nokiC = 0
	nokdC = 0
	noecC = 0
	nouid = 0
	dropic = 0
	dropki = 0
	dropkd = 0
	dropec = 0
	dropicC = 0
	dropkiC = 0
	dropkdC = 0
	dropecC = 0
	count1_ic = 0
	count2_ic = 0
	count3_ic = 0
	count4_ic = 0
	count5_ic = 0
	count6_ic = 0
	count1_ec = 0
	count2_ec = 0
	count3_ec = 0
	count4_ec = 0
	count5_ec = 0
	count6_ec = 0
	count1_ki = 0
	count2_ki = 0
	count3_ki = 0
	count4_ki = 0
	count5_ki = 0
	count6_ki = 0
	count1_kd = 0
	count2_kd = 0
	count3_kd = 0
	count4_kd = 0
	count5_kd = 0
	count6_kd = 0
	mark1 = 0  #train
	mark2 = 0  #GPCR
	mark3 = 0  #ER
	mark4 = 0  #channel
	mark5 = 0  #kinase
	mark6 = 0  #test
	random.seed(1234)
	for folder in ['IC50', 'Ki', 'Kd', 'EC50']:
		if not os.path.exists('./' + folder):
			os.makedirs('./' + folder)
		if not os.path.exists('./' + folder + '/baseline'):
			os.makedirs('./' + folder + '/baseline')
		if not os.path.exists('./' + folder + '/SPS'):
			os.makedirs('./' + folder + '/SPS')
	ic_o1 = open('./IC50/train','w+')
	ic_o2 = open('./IC50/GPCR','w+')
	ic_o3 = open('./IC50/ER','w+')
	ic_o4 = open('./IC50/channel','w+')
	ic_o5 = open('./IC50/kinase','w+')
	ic_o6 = open('./IC50/test','w+')
	ec_o1 = open('./EC50/train','w+')
	ec_o2 = open('./EC50/GPCR','w+')
	ec_o3 = open('./EC50/ER','w+')
	ec_o4 = open('./EC50/channel','w+')
	ec_o5 = open('./EC50/kinase','w+')
	ec_o6 = open('./EC50/test','w+')
	ki_o1 = open('./Ki/train','w+')
	ki_o2 = open('./Ki/GPCR','w+')
	ki_o3 = open('./Ki/ER','w+')
	ki_o4 = open('./Ki/channel','w+')
	ki_o5 = open('./Ki/kinase','w+')
	ki_o6 = open('./Ki/test','w+')
	kd_o1 = open('./Kd/train','w+')
	kd_o2 = open('./Kd/GPCR','w+')
	kd_o3 = open('./Kd/ER','w+')
	kd_o4 = open('./Kd/channel','w+')
	kd_o5 = open('./Kd/kinase','w+')
	kd_o6 = open('./Kd/test','w+')

	ic_o1_writer = csv.writer(ic_o1, dialect='excel-tab')
	ic_o2_writer = csv.writer(ic_o2, dialect='excel-tab')
	ic_o3_writer = csv.writer(ic_o3, dialect='excel-tab')
	ic_o4_writer = csv.writer(ic_o4, dialect='excel-tab')
	ic_o5_writer = csv.writer(ic_o5, dialect='excel-tab')
	ic_o6_writer = csv.writer(ic_o6, dialect='excel-tab')
	ec_o1_writer = csv.writer(ec_o1, dialect='excel-tab')
	ec_o2_writer = csv.writer(ec_o2, dialect='excel-tab')
	ec_o3_writer = csv.writer(ec_o3, dialect='excel-tab')
	ec_o4_writer = csv.writer(ec_o4, dialect='excel-tab')
	ec_o5_writer = csv.writer(ec_o5, dialect='excel-tab')
	ec_o6_writer = csv.writer(ec_o6, dialect='excel-tab')
	ki_o1_writer = csv.writer(ki_o1, dialect='excel-tab')
	ki_o2_writer = csv.writer(ki_o2, dialect='excel-tab')
	ki_o3_writer = csv.writer(ki_o3, dialect='excel-tab')
	ki_o4_writer = csv.writer(ki_o4, dialect='excel-tab')
	ki_o5_writer = csv.writer(ki_o5, dialect='excel-tab')
	ki_o6_writer = csv.writer(ki_o6, dialect='excel-tab')
	kd_o1_writer = csv.writer(kd_o1, dialect='excel-tab')
	kd_o2_writer = csv.writer(kd_o2, dialect='excel-tab')
	kd_o3_writer = csv.writer(kd_o3, dialect='excel-tab')
	kd_o4_writer = csv.writer(kd_o4, dialect='excel-tab')
	kd_o5_writer = csv.writer(kd_o5, dialect='excel-tab')
	kd_o6_writer = csv.writer(kd_o6, dialect='excel-tab')

	#IC50
	bl_ic_w1_ic = open('./IC50/baseline/train_ic50', 'w+')
	bl_ic_w2_ic = open('./IC50/baseline/GPCR_ic50', 'w+')
	bl_ic_w3_ic = open('./IC50/baseline/ER_ic50', 'w+')
	bl_ic_w4_ic = open('./IC50/baseline/channel_ic50', 'w+')
	bl_ic_w5_ic = open('./IC50/baseline/kinase_ic50', 'w+')
	bl_ic_w6_ic = open('./IC50/baseline/test_ic50', 'w+')
	bl_ic_w1_com = open('./IC50/baseline/train_compound_fingerprint', 'w+')
	bl_ic_w2_com = open('./IC50/baseline/GPCR_compound_fingerprint', 'w+')
	bl_ic_w3_com = open('./IC50/baseline/ER_compound_fingerprint', 'w+')
	bl_ic_w4_com = open('./IC50/baseline/channel_compound_fingerprint', 'w+')
	bl_ic_w5_com = open('./IC50/baseline/kinase_compound_fingerprint', 'w+')
	bl_ic_w6_com = open('./IC50/baseline/test_compound_fingerprint', 'w+')
	bl_ic_w1_pro = open('./IC50/baseline/train_protein_domain', 'w+')
	bl_ic_w2_pro = open('./IC50/baseline/GPCR_protein_domain', 'w+')
	bl_ic_w3_pro = open('./IC50/baseline/ER_protein_domain', 'w+')
	bl_ic_w4_pro = open('./IC50/baseline/channel_protein_domain', 'w+')
	bl_ic_w5_pro = open('./IC50/baseline/kinase_protein_domain', 'w+')
	bl_ic_w6_pro = open('./IC50/baseline/test_protein_domain', 'w+')
	bl_ic_w1_seq = open('./IC50/baseline/train_protein_seq', 'w+')
	bl_ic_w2_seq = open('./IC50/baseline/GPCR_protein_seq', 'w+')
	bl_ic_w3_seq = open('./IC50/baseline/ER_protein_seq', 'w+')
	bl_ic_w4_seq = open('./IC50/baseline/channel_protein_seq', 'w+')
	bl_ic_w5_seq = open('./IC50/baseline/kinase_protein_seq', 'w+')
	bl_ic_w6_seq = open('./IC50/baseline/test_protein_seq', 'w+')

	sps_ic_w1_ic = open('./IC50/SPS/train_ic50', 'w+')
	sps_ic_w2_ic = open('./IC50/SPS/GPCR_ic50', 'w+')
	sps_ic_w3_ic = open('./IC50/SPS/ER_ic50', 'w+')
	sps_ic_w4_ic = open('./IC50/SPS/channel_ic50', 'w+')
	sps_ic_w5_ic = open('./IC50/SPS/kinase_ic50', 'w+')
	sps_ic_w6_ic = open('./IC50/SPS/test_ic50', 'w+')
	sps_ic_w1_smi = open('./IC50/SPS/train_smile', 'w+')
	sps_ic_w2_smi = open('./IC50/SPS/GPCR_smile', 'w+')
	sps_ic_w3_smi = open('./IC50/SPS/ER_smile', 'w+')
	sps_ic_w4_smi = open('./IC50/SPS/channel_smile', 'w+')
	sps_ic_w5_smi = open('./IC50/SPS/kinase_smile', 'w+')
	sps_ic_w6_smi = open('./IC50/SPS/test_smile', 'w+')
	sps_ic_w1_sps = open('./IC50/SPS/train_sps', 'w+')
	sps_ic_w2_sps = open('./IC50/SPS/GPCR_sps', 'w+')
	sps_ic_w3_sps = open('./IC50/SPS/ER_sps', 'w+')
	sps_ic_w4_sps = open('./IC50/SPS/channel_sps', 'w+')
	sps_ic_w5_sps = open('./IC50/SPS/kinase_sps', 'w+')
	sps_ic_w6_sps = open('./IC50/SPS/test_sps', 'w+')

	#EC50
	bl_ec_w1_ec = open('./EC50/baseline/train_ec50', 'w+')
	bl_ec_w2_ec = open('./EC50/baseline/GPCR_ec50', 'w+')
	bl_ec_w3_ec = open('./EC50/baseline/ER_ec50', 'w+')
	bl_ec_w4_ec = open('./EC50/baseline/channel_ec50', 'w+')
	bl_ec_w5_ec = open('./EC50/baseline/kinase_ec50', 'w+')
	bl_ec_w6_ec = open('./EC50/baseline/test_ec50', 'w+')
	bl_ec_w1_com = open('./EC50/baseline/train_compound_fingerprint', 'w+')
	bl_ec_w2_com = open('./EC50/baseline/GPCR_compound_fingerprint', 'w+')
	bl_ec_w3_com = open('./EC50/baseline/ER_compound_fingerprint', 'w+')
	bl_ec_w4_com = open('./EC50/baseline/channel_compound_fingerprint', 'w+')
	bl_ec_w5_com = open('./EC50/baseline/kinase_compound_fingerprint', 'w+')
	bl_ec_w6_com = open('./EC50/baseline/test_compound_fingerprint', 'w+')
	bl_ec_w1_pro = open('./EC50/baseline/train_protein_domain', 'w+')
	bl_ec_w2_pro = open('./EC50/baseline/GPCR_protein_domain', 'w+')
	bl_ec_w3_pro = open('./EC50/baseline/ER_protein_domain', 'w+')
	bl_ec_w4_pro = open('./EC50/baseline/channel_protein_domain', 'w+')
	bl_ec_w5_pro = open('./EC50/baseline/kinase_protein_domain', 'w+')
	bl_ec_w6_pro = open('./EC50/baseline/test_protein_domain', 'w+')
	bl_ec_w1_seq = open('./EC50/baseline/train_protein_seq', 'w+')
	bl_ec_w2_seq = open('./EC50/baseline/GPCR_protein_seq', 'w+')
	bl_ec_w3_seq = open('./EC50/baseline/ER_protein_seq', 'w+')
	bl_ec_w4_seq = open('./EC50/baseline/channel_protein_seq', 'w+')
	bl_ec_w5_seq = open('./EC50/baseline/kinase_protein_seq', 'w+')
	bl_ec_w6_seq = open('./EC50/baseline/test_protein_seq', 'w+')

	sps_ec_w1_ec = open('./EC50/SPS/train_ec50', 'w+')
	sps_ec_w2_ec = open('./EC50/SPS/GPCR_ec50', 'w+')
	sps_ec_w3_ec = open('./EC50/SPS/ER_ec50', 'w+')
	sps_ec_w4_ec = open('./EC50/SPS/channel_ec50', 'w+')
	sps_ec_w5_ec = open('./EC50/SPS/kinase_ec50', 'w+')
	sps_ec_w6_ec = open('./EC50/SPS/test_ec50', 'w+')
	sps_ec_w1_smi = open('./EC50/SPS/train_smile', 'w+')
	sps_ec_w2_smi = open('./EC50/SPS/GPCR_smile', 'w+')
	sps_ec_w3_smi = open('./EC50/SPS/ER_smile', 'w+')
	sps_ec_w4_smi = open('./EC50/SPS/channel_smile', 'w+')
	sps_ec_w5_smi = open('./EC50/SPS/kinase_smile', 'w+')
	sps_ec_w6_smi = open('./EC50/SPS/test_smile', 'w+')
	sps_ec_w1_sps = open('./EC50/SPS/train_sps', 'w+')
	sps_ec_w2_sps = open('./EC50/SPS/GPCR_sps', 'w+')
	sps_ec_w3_sps = open('./EC50/SPS/ER_sps', 'w+')
	sps_ec_w4_sps = open('./EC50/SPS/channel_sps', 'w+')
	sps_ec_w5_sps = open('./EC50/SPS/kinase_sps', 'w+')
	sps_ec_w6_sps = open('./EC50/SPS/test_sps', 'w+')

	#Ki
	bl_ki_w1_ki = open('./Ki/baseline/train_ki', 'w+')
	bl_ki_w2_ki = open('./Ki/baseline/GPCR_ki', 'w+')
	bl_ki_w3_ki = open('./Ki/baseline/ER_ki', 'w+')
	bl_ki_w4_ki = open('./Ki/baseline/channel_ki', 'w+')
	bl_ki_w5_ki = open('./Ki/baseline/kinase_ki', 'w+')
	bl_ki_w6_ki = open('./Ki/baseline/test_ki', 'w+')
	bl_ki_w1_com = open('./Ki/baseline/train_compound_fingerprint', 'w+')
	bl_ki_w2_com = open('./Ki/baseline/GPCR_compound_fingerprint', 'w+')
	bl_ki_w3_com = open('./Ki/baseline/ER_compound_fingerprint', 'w+')
	bl_ki_w4_com = open('./Ki/baseline/channel_compound_fingerprint', 'w+')
	bl_ki_w5_com = open('./Ki/baseline/kinase_compound_fingerprint', 'w+')
	bl_ki_w6_com = open('./Ki/baseline/test_compound_fingerprint', 'w+')
	bl_ki_w1_pro = open('./Ki/baseline/train_protein_domain', 'w+')
	bl_ki_w2_pro = open('./Ki/baseline/GPCR_protein_domain', 'w+')
	bl_ki_w3_pro = open('./Ki/baseline/ER_protein_domain', 'w+')
	bl_ki_w4_pro = open('./Ki/baseline/channel_protein_domain', 'w+')
	bl_ki_w5_pro = open('./Ki/baseline/kinase_protein_domain', 'w+')
	bl_ki_w6_pro = open('./Ki/baseline/test_protein_domain', 'w+')
	bl_ki_w1_seq = open('./Ki/baseline/train_protein_seq', 'w+')
	bl_ki_w2_seq = open('./Ki/baseline/GPCR_protein_seq', 'w+')
	bl_ki_w3_seq = open('./Ki/baseline/ER_protein_seq', 'w+')
	bl_ki_w4_seq = open('./Ki/baseline/channel_protein_seq', 'w+')
	bl_ki_w5_seq = open('./Ki/baseline/kinase_protein_seq', 'w+')
	bl_ki_w6_seq = open('./Ki/baseline/test_protein_seq', 'w+')

	sps_ki_w1_ki = open('./Ki/SPS/train_ki', 'w+')
	sps_ki_w2_ki = open('./Ki/SPS/GPCR_ki', 'w+')
	sps_ki_w3_ki = open('./Ki/SPS/ER_ki', 'w+')
	sps_ki_w4_ki = open('./Ki/SPS/channel_ki', 'w+')
	sps_ki_w5_ki = open('./Ki/SPS/kinase_ki', 'w+')
	sps_ki_w6_ki = open('./Ki/SPS/test_ki', 'w+')
	sps_ki_w1_smi = open('./Ki/SPS/train_smile', 'w+')
	sps_ki_w2_smi = open('./Ki/SPS/GPCR_smile', 'w+')
	sps_ki_w3_smi = open('./Ki/SPS/ER_smile', 'w+')
	sps_ki_w4_smi = open('./Ki/SPS/channel_smile', 'w+')
	sps_ki_w5_smi = open('./Ki/SPS/kinase_smile', 'w+')
	sps_ki_w6_smi = open('./Ki/SPS/test_smile', 'w+')
	sps_ki_w1_sps = open('./Ki/SPS/train_sps', 'w+')
	sps_ki_w2_sps = open('./Ki/SPS/GPCR_sps', 'w+')
	sps_ki_w3_sps = open('./Ki/SPS/ER_sps', 'w+')
	sps_ki_w4_sps = open('./Ki/SPS/channel_sps', 'w+')
	sps_ki_w5_sps = open('./Ki/SPS/kinase_sps', 'w+')
	sps_ki_w6_sps = open('./Ki/SPS/test_sps', 'w+')

	# Kd
	bl_kd_w1_kd = open('./Kd/baseline/train_kd', 'w+')
	bl_kd_w2_kd = open('./Kd/baseline/GPCR_kd', 'w+')
	bl_kd_w3_kd = open('./Kd/baseline/ER_kd', 'w+')
	bl_kd_w4_kd = open('./Kd/baseline/channel_kd', 'w+')
	bl_kd_w5_kd = open('./Kd/baseline/kinase_kd', 'w+')
	bl_kd_w6_kd = open('./Kd/baseline/test_kd', 'w+')
	bl_kd_w1_com = open('./Kd/baseline/train_compound_fingerprint', 'w+')
	bl_kd_w2_com = open('./Kd/baseline/GPCR_compound_fingerprint', 'w+')
	bl_kd_w3_com = open('./Kd/baseline/ER_compound_fingerprint', 'w+')
	bl_kd_w4_com = open('./Kd/baseline/channel_compound_fingerprint', 'w+')
	bl_kd_w5_com = open('./Kd/baseline/kinase_compound_fingerprint', 'w+')
	bl_kd_w6_com = open('./Kd/baseline/test_compound_fingerprint', 'w+')
	bl_kd_w1_pro = open('./Kd/baseline/train_protein_domain', 'w+')
	bl_kd_w2_pro = open('./Kd/baseline/GPCR_protein_domain', 'w+')
	bl_kd_w3_pro = open('./Kd/baseline/ER_protein_domain', 'w+')
	bl_kd_w4_pro = open('./Kd/baseline/channel_protein_domain', 'w+')
	bl_kd_w5_pro = open('./Kd/baseline/kinase_protein_domain', 'w+')
	bl_kd_w6_pro = open('./Kd/baseline/test_protein_domain', 'w+')
	bl_kd_w1_seq = open('./Kd/baseline/train_protein_seq', 'w+')
	bl_kd_w2_seq = open('./Kd/baseline/GPCR_protein_seq', 'w+')
	bl_kd_w3_seq = open('./Kd/baseline/ER_protein_seq', 'w+')
	bl_kd_w4_seq = open('./Kd/baseline/channel_protein_seq', 'w+')
	bl_kd_w5_seq = open('./Kd/baseline/kinase_protein_seq', 'w+')
	bl_kd_w6_seq = open('./Kd/baseline/test_protein_seq', 'w+')

	sps_kd_w1_kd = open('./Kd/SPS/train_kd', 'w+')
	sps_kd_w2_kd = open('./Kd/SPS/GPCR_kd', 'w+')
	sps_kd_w3_kd = open('./Kd/SPS/ER_kd', 'w+')
	sps_kd_w4_kd = open('./Kd/SPS/channel_kd', 'w+')
	sps_kd_w5_kd = open('./Kd/SPS/kinase_kd', 'w+')
	sps_kd_w6_kd = open('./Kd/SPS/test_kd', 'w+')
	sps_kd_w1_smi = open('./Kd/SPS/train_smile', 'w+')
	sps_kd_w2_smi = open('./Kd/SPS/GPCR_smile', 'w+')
	sps_kd_w3_smi = open('./Kd/SPS/ER_smile', 'w+')
	sps_kd_w4_smi = open('./Kd/SPS/channel_smile', 'w+')
	sps_kd_w5_smi = open('./Kd/SPS/kinase_smile', 'w+')
	sps_kd_w6_smi = open('./Kd/SPS/test_smile', 'w+')
	sps_kd_w1_sps = open('./Kd/SPS/train_sps', 'w+')
	sps_kd_w2_sps = open('./Kd/SPS/GPCR_sps', 'w+')
	sps_kd_w3_sps = open('./Kd/SPS/ER_sps', 'w+')
	sps_kd_w4_sps = open('./Kd/SPS/channel_sps', 'w+')
	sps_kd_w5_sps = open('./Kd/SPS/kinase_sps', 'w+')
	sps_kd_w6_sps = open('./Kd/SPS/test_sps', 'w+')


	#Loading dictionary
	seq_sps = {}
	seq_dom = {}
	cid_fea = {}
	uid_class = {}

	groupedDir = 'protein_grouped_finalPresentation' # protein_grouped directory
	proteinDom = 'Seq_Vector'                        # protein_feature directory
	compoundFea = 'CID_Smi_Feature'                  # compound_Feature directory
	uidClass = 'uniID_class'                         # UniprotID_Class directory


	with open(uidClass) as f:
		mark = 0
		for line in f:
			if mark == 0:
				uid = line.strip()
				mark = 1
			else:
				uinClass = line.strip()
				mark = 0
				uid_class[uid] = uinClass

	with open(groupedDir) as f:
		mark = 0
		for line in f:
			if mark == 0:
				protein = line.strip()
				mark = 1
			else:
				sps = line.strip()
				mark = 0
				seq_sps[protein] = sps

	with open(proteinDom) as f:
		mark = 0
		for line in f:
			if mark == 0:
				protein = line.strip()
				mark = 1
			else:
				dom = line.strip()
				mark = 0
				seq_dom[protein] = dom

	with open(compoundFea) as f:
		for line in f:
			line = line.strip()
			if line == '> <PUBCHEM_COMPOUND_CID>':
				cid = f.next().strip()
			elif line == '> <PUBCHEM_CACTVS_SUBSKEYS>':
				fea = f.next().strip()
			elif line == '$$$$':
				cid_fea[cid] = fea
	print('Dictionary loaded')

	with open(dataDir) as f, open('noUniprotID.tsv','w+') as tsvout:
		noUniprotID = csv.writer(tsvout, dialect='excel-tab')
		reader = csv.reader(f, delimiter="\t",quoting=csv.QUOTE_NONE)
		header = reader.next()
		header[8] = "pKi_[M]"
		header[9] = "pIC50_[M]"
		header[10] = "pKd_[M]"
		header[11] = "pEC50_[M]"
		noUniprotID.writerow(header)
		ic_o1_writer.writerow(header)
		ic_o2_writer.writerow(header)
		ic_o3_writer.writerow(header)
		ic_o4_writer.writerow(header)
		ic_o5_writer.writerow(header)
		ic_o6_writer.writerow(header)
		ec_o1_writer.writerow(header)
		ec_o2_writer.writerow(header)
		ec_o3_writer.writerow(header)
		ec_o4_writer.writerow(header)
		ec_o5_writer.writerow(header)
		ec_o6_writer.writerow(header)
		ki_o1_writer.writerow(header)
		ki_o2_writer.writerow(header)
		ki_o3_writer.writerow(header)
		ki_o4_writer.writerow(header)
		ki_o5_writer.writerow(header)
		ki_o6_writer.writerow(header)
		kd_o1_writer.writerow(header)
		kd_o2_writer.writerow(header)
		kd_o3_writer.writerow(header)
		kd_o4_writer.writerow(header)
		kd_o5_writer.writerow(header)
		kd_o6_writer.writerow(header)
		for row in reader:
			smi = row[1]
			bd_name = row[6]
			uni_name = row[39]
			ki = row[8]
			ic = row[9]
			kd = row[10]
			ec = row[11]
			cid = row[28]
			seq = row[37]
			uid = row[41]
			if seq not in seq_dom:
				print(row)
			domain = pos2vec(seq_dom[seq])
			#domain = seq_dom[seq]
			feature = cid_fea[cid]
			sps = seq_sps[seq]

	#filter IC50, Ki, Kd with corresponding lower and upper bound threshold and convert nM to µM
			if ic.strip() == '':
				dropic = 1
				noicC += 1
			elif '<' in ic:
				value = float(re.sub(r'[><=]+', '', ic))
				if value > 0.01:
					dropic = 1
					dropicC += 1
				else:
					ic = str(11)
			elif '>' in ic:
				value = float(re.sub(r'[><=]+', '', ic))
				if value < 1E7:
					dropic = 1
					dropicC += 1
				else:
					ic = str(2)
			else:
				if float(ic) < 0.01:
					ic = str(11)
				elif float(ic) > 1E7:
					ic = str(2)
				else:
					ic = str(-np.log10(float(ic))+9)


			if ec.strip() == '':
				dropec = 1
				noecC += 1
			elif '<' in ec:
				value = float(re.sub(r'[><=]+', '', ec))
				if value > 0.01:
					dropec = 1
					dropecC += 1
				else:
					ec = str(11)
			elif '>' in ec:
				value = float(re.sub(r'[><=]+', '', ec))
				if value < 1E7:
					dropec = 1
					dropecC += 1
				else:
					ec = str(2)
			else:
				if float(ec) < 0.01:
					ec = str(11)
				elif float(ec) > 1E7:
					ec = str(2)
				else:
					ec = str(-np.log10(float(ec))+9)

			if ki.strip() == '':
				dropki = 1
				nokiC += 1
			elif '<' in ki:
				value = float(re.sub(r'[><=]+', '', ki))
				if value > 0.01:
					dropki = 1
					dropkiC += 1
				else:
					ki = str(11)
			elif '>' in ki:
				value = float(re.sub(r'[><=]+', '', ki))
				if value < 1E7:
					dropki = 1
					dropkiC += 1
				else:
					ki = str(2)
			else:
				if float(ki) < 0.01:
					ki = str(11)
				elif float(ki) > 1E7:
					ki = str(2)
				else:
					ki = str(-np.log10(float(ki))+9)

			if kd.strip() == '':
				dropkd = 1
				nokdC += 1
			elif '<' in kd:
				value = float(re.sub(r'[><=]+', '', kd))
				if value > 0.01:
					dropkd = 1
					dropkdC += 1
				else:
					kd = str(11)
			elif '>' in kd:
				value = float(re.sub(r'[><=]+', '', kd))
				if value < 1E7:
					dropkd = 1
					dropkdC += 1
				else:
					kd = str(2)
			else:
				if float(kd) < 0.01:
					kd = str(11)
				elif float(kd) > 1E7:
					kd = str(2)
				else:
					kd = str(-np.log10(float(kd))+9)

		#split data by class
			if uid.strip() == '':
				nouid = 1
			else:
				uniClass = uid_class[uid].strip()
				if uniClass == 'GPCR':
					mark2 = 1
				   # mark6 = 1
				elif uniClass == 'ER':
					mark3 = 1
				   # mark6 = 1
				elif uniClass == 'channel':
					mark4 = 1
				   # mark6 = 1
				elif uniClass == 'kinase':
					mark5 = 1
				   # mark6 = 1
				else:
					randomNum = random.randint(1,100)
					if randomNum <= 30:
						mark6 = 1
					else:
						mark1 = 1
			row[8] = ki
			row[9] = ic
			row[10] = kd
			row[11] = ec
		#output to files
			if mark1 == 1:
				if dropic == 0:
					ic_o1_writer.writerow(row)
					bl_ic_w1_ic.write(ic+'\n')
					bl_ic_w1_com.write(feature+'\n')
					bl_ic_w1_pro.write(domain+'\n')
					bl_ic_w1_seq.write(seq+'\n')
					sps_ic_w1_ic.write(ic+'\n')
					sps_ic_w1_smi.write(smi+'\n')
					sps_ic_w1_sps.write(sps+'\n')
					countic += 1
					count1_ic += 1
				if dropec == 0:
					ec_o1_writer.writerow(row)
					bl_ec_w1_ec.write(ec+'\n')
					bl_ec_w1_com.write(feature+'\n')
					bl_ec_w1_pro.write(domain+'\n')
					bl_ec_w1_seq.write(seq+'\n')
					sps_ec_w1_ec.write(ec+'\n')
					sps_ec_w1_smi.write(smi+'\n')
					sps_ec_w1_sps.write(sps+'\n')
					countec += 1
					count1_ec += 1
				if dropki == 0:
					ki_o1_writer.writerow(row)
					bl_ki_w1_ki.write(ki+'\n')
					bl_ki_w1_com.write(feature+'\n')
					bl_ki_w1_pro.write(domain+'\n')
					bl_ki_w1_seq.write(seq+'\n')
					sps_ki_w1_ki.write(ki+'\n')
					sps_ki_w1_smi.write(smi+'\n')
					sps_ki_w1_sps.write(sps+'\n')
					countki += 1
					count1_ki += 1
				if dropkd == 0:
					kd_o1_writer.writerow(row)
					bl_kd_w1_kd.write(kd+'\n')
					bl_kd_w1_com.write(feature+'\n')
					bl_kd_w1_pro.write(domain+'\n')
					bl_kd_w1_seq.write(seq+'\n')
					sps_kd_w1_kd.write(kd+'\n')
					sps_kd_w1_smi.write(smi+'\n')
					sps_kd_w1_sps.write(sps+'\n')
					countkd += 1
					count1_kd += 1
			if mark2 == 1:
				if dropic == 0:
					ic_o2_writer.writerow(row)
					bl_ic_w2_ic.write(ic+'\n')
					bl_ic_w2_com.write(feature+'\n')
					bl_ic_w2_pro.write(domain+'\n')
					bl_ic_w2_seq.write(seq+'\n')
					sps_ic_w2_ic.write(ic+'\n')
					sps_ic_w2_smi.write(smi+'\n')
					sps_ic_w2_sps.write(sps+'\n')
					countic += 1
					count2_ic += 1
				if dropec == 0:
					ec_o2_writer.writerow(row)
					bl_ec_w2_ec.write(ec+'\n')
					bl_ec_w2_com.write(feature+'\n')
					bl_ec_w2_pro.write(domain+'\n')
					bl_ec_w2_seq.write(seq+'\n')
					sps_ec_w2_ec.write(ec+'\n')
					sps_ec_w2_smi.write(smi+'\n')
					sps_ec_w2_sps.write(sps+'\n')
					countec += 1
					count2_ec += 1
				if dropki == 0:
					ki_o2_writer.writerow(row)
					bl_ki_w2_ki.write(ki+'\n')
					bl_ki_w2_com.write(feature+'\n')
					bl_ki_w2_pro.write(domain+'\n')
					bl_ki_w2_seq.write(seq+'\n')
					sps_ki_w2_ki.write(ki+'\n')
					sps_ki_w2_smi.write(smi+'\n')
					sps_ki_w2_sps.write(sps+'\n')
					countki += 1
					count2_ki += 1
				if dropkd == 0:
					kd_o2_writer.writerow(row)
					bl_kd_w2_kd.write(kd+'\n')
					bl_kd_w2_com.write(feature+'\n')
					bl_kd_w2_pro.write(domain+'\n')
					bl_kd_w2_seq.write(seq+'\n')
					sps_kd_w2_kd.write(kd+'\n')
					sps_kd_w2_smi.write(smi+'\n')
					sps_kd_w2_sps.write(sps+'\n')
					countkd += 1
					count2_kd += 1
			if mark3 == 1:
				if dropic == 0:
					ic_o3_writer.writerow(row)
					bl_ic_w3_ic.write(ic+'\n')
					bl_ic_w3_com.write(feature+'\n')
					bl_ic_w3_pro.write(domain+'\n')
					bl_ic_w3_seq.write(seq+'\n')
					sps_ic_w3_ic.write(ic+'\n')
					sps_ic_w3_smi.write(smi+'\n')
					sps_ic_w3_sps.write(sps+'\n')
					countic += 1
					count3_ic += 1
				if dropec == 0:
					ec_o3_writer.writerow(row)
					bl_ec_w3_ec.write(ec+'\n')
					bl_ec_w3_com.write(feature+'\n')
					bl_ec_w3_pro.write(domain+'\n')
					bl_ec_w3_seq.write(seq+'\n')
					sps_ec_w3_ec.write(ec+'\n')
					sps_ec_w3_smi.write(smi+'\n')
					sps_ec_w3_sps.write(sps+'\n')
					countec += 1
					count3_ec += 1
				if dropki == 0:
					ki_o3_writer.writerow(row)
					bl_ki_w3_ki.write(ki+'\n')
					bl_ki_w3_com.write(feature+'\n')
					bl_ki_w3_pro.write(domain+'\n')
					bl_ki_w3_seq.write(seq+'\n')
					sps_ki_w3_ki.write(ki+'\n')
					sps_ki_w3_smi.write(smi+'\n')
					sps_ki_w3_sps.write(sps+'\n')
					countki += 1
					count3_ki += 1
				if dropkd == 0:
					kd_o3_writer.writerow(row)
					bl_kd_w3_kd.write(kd+'\n')
					bl_kd_w3_com.write(feature+'\n')
					bl_kd_w3_pro.write(domain+'\n')
					bl_kd_w3_seq.write(seq+'\n')
					sps_kd_w3_kd.write(kd+'\n')
					sps_kd_w3_smi.write(smi+'\n')
					sps_kd_w3_sps.write(sps+'\n')
					countkd += 1
					count3_kd += 1
			if mark4 == 1:
				if dropic == 0:
					ic_o4_writer.writerow(row)
					bl_ic_w4_ic.write(ic+'\n')
					bl_ic_w4_com.write(feature+'\n')
					bl_ic_w4_pro.write(domain+'\n')
					bl_ic_w4_seq.write(seq+'\n')
					sps_ic_w4_ic.write(ic+'\n')
					sps_ic_w4_smi.write(smi+'\n')
					sps_ic_w4_sps.write(sps+'\n')
					countic += 1
					count4_ic += 1
				if dropec == 0:
					ec_o4_writer.writerow(row)
					bl_ec_w4_ec.write(ec+'\n')
					bl_ec_w4_com.write(feature+'\n')
					bl_ec_w4_pro.write(domain+'\n')
					bl_ec_w4_seq.write(seq+'\n')
					sps_ec_w4_ec.write(ec+'\n')
					sps_ec_w4_smi.write(smi+'\n')
					sps_ec_w4_sps.write(sps+'\n')
					countec += 1
					count4_ec += 1
				if dropki == 0:
					ki_o4_writer.writerow(row)
					bl_ki_w4_ki.write(ki+'\n')
					bl_ki_w4_com.write(feature+'\n')
					bl_ki_w4_pro.write(domain+'\n')
					bl_ki_w4_seq.write(seq+'\n')
					sps_ki_w4_ki.write(ki+'\n')
					sps_ki_w4_smi.write(smi+'\n')
					sps_ki_w4_sps.write(sps+'\n')
					countki += 1
					count4_ki += 1
				if dropkd == 0:
					kd_o4_writer.writerow(row)
					bl_kd_w4_kd.write(kd+'\n')
					bl_kd_w4_com.write(feature+'\n')
					bl_kd_w4_pro.write(domain+'\n')
					bl_kd_w4_seq.write(seq+'\n')
					sps_kd_w4_kd.write(kd+'\n')
					sps_kd_w4_smi.write(smi+'\n')
					sps_kd_w4_sps.write(sps+'\n')
					countkd += 1
					count4_kd += 1
			if mark5 == 1:
				if dropic == 0:
					ic_o5_writer.writerow(row)
					bl_ic_w5_ic.write(ic+'\n')
					bl_ic_w5_com.write(feature+'\n')
					bl_ic_w5_pro.write(domain+'\n')
					bl_ic_w5_seq.write(seq+'\n')
					sps_ic_w5_ic.write(ic+'\n')
					sps_ic_w5_smi.write(smi+'\n')
					sps_ic_w5_sps.write(sps+'\n')
					countic += 1
					count5_ic += 1
				if dropec == 0:
					ec_o5_writer.writerow(row)
					bl_ec_w5_ec.write(ec+'\n')
					bl_ec_w5_com.write(feature+'\n')
					bl_ec_w5_pro.write(domain+'\n')
					bl_ec_w5_seq.write(seq+'\n')
					sps_ec_w5_ec.write(ec+'\n')
					sps_ec_w5_smi.write(smi+'\n')
					sps_ec_w5_sps.write(sps+'\n')
					countec += 1
					count5_ec += 1
				if dropki == 0:
					ki_o5_writer.writerow(row)
					bl_ki_w5_ki.write(ki+'\n')
					bl_ki_w5_com.write(feature+'\n')
					bl_ki_w5_pro.write(domain+'\n')
					bl_ki_w5_seq.write(seq+'\n')
					sps_ki_w5_ki.write(ki+'\n')
					sps_ki_w5_smi.write(smi+'\n')
					sps_ki_w5_sps.write(sps+'\n')
					countki += 1
					count5_ki += 1
				if dropkd == 0:
					kd_o5_writer.writerow(row)
					bl_kd_w5_kd.write(kd+'\n')
					bl_kd_w5_com.write(feature+'\n')
					bl_kd_w5_pro.write(domain+'\n')
					bl_kd_w5_seq.write(seq+'\n')
					sps_kd_w5_kd.write(kd+'\n')
					sps_kd_w5_smi.write(smi+'\n')
					sps_kd_w5_sps.write(sps+'\n')
					countkd += 1
					count5_kd += 1
			if mark6 == 1:
				if dropic == 0:
					ic_o6_writer.writerow(row)
					bl_ic_w6_ic.write(ic+'\n')
					bl_ic_w6_com.write(feature+'\n')
					bl_ic_w6_pro.write(domain+'\n')
					bl_ic_w6_seq.write(seq+'\n')
					sps_ic_w6_ic.write(ic+'\n')
					sps_ic_w6_smi.write(smi+'\n')
					sps_ic_w6_sps.write(sps+'\n')
					countic += 1
					count6_ic += 1
				if dropec == 0:
					ec_o6_writer.writerow(row)
					bl_ec_w6_ec.write(ec+'\n')
					bl_ec_w6_com.write(feature+'\n')
					bl_ec_w6_pro.write(domain+'\n')
					bl_ec_w6_seq.write(seq+'\n')
					sps_ec_w6_ec.write(ec+'\n')
					sps_ec_w6_smi.write(smi+'\n')
					sps_ec_w6_sps.write(sps+'\n')
					countec += 1
					count6_ec += 1
				if dropki == 0:
					ki_o6_writer.writerow(row)
					bl_ki_w6_ki.write(ki+'\n')
					bl_ki_w6_com.write(feature+'\n')
					bl_ki_w6_pro.write(domain+'\n')
					bl_ki_w6_seq.write(seq+'\n')
					sps_ki_w6_ki.write(ki+'\n')
					sps_ki_w6_smi.write(smi+'\n')
					sps_ki_w6_sps.write(sps+'\n')
					countki += 1
					count6_ki += 1
				if dropkd == 0:
					kd_o6_writer.writerow(row)
					bl_kd_w6_kd.write(kd+'\n')
					bl_kd_w6_com.write(feature+'\n')
					bl_kd_w6_pro.write(domain+'\n')
					bl_kd_w6_seq.write(seq+'\n')
					sps_kd_w6_kd.write(kd+'\n')
					sps_kd_w6_smi.write(smi+'\n')
					sps_kd_w6_sps.write(sps+'\n')
					countkd += 1
					count6_kd += 1
			if nouid == 1:
				noUniprotID.writerow(row)

			mark1 = 0
			mark2 = 0
			mark3 = 0
			mark4 = 0
			mark5 = 0
			mark6 = 0
			dropec = 0
			dropic = 0
			dropki = 0
			dropkd = 0
			nouid = 0

	ic_o1.close()
	ic_o2.close()
	ic_o3.close()
	ic_o4.close()
	ic_o5.close()
	ic_o6.close()
	ec_o1.close()
	ec_o2.close()
	ec_o3.close()
	ec_o4.close()
	ec_o5.close()
	ec_o6.close()
	ki_o1.close()
	ki_o2.close()
	ki_o3.close()
	ki_o4.close()
	ki_o5.close()
	ki_o6.close()
	kd_o1.close()
	kd_o2.close()
	kd_o3.close()
	kd_o4.close()
	kd_o5.close()
	kd_o6.close()

	# IC50
	bl_ic_w1_ic.close()
	bl_ic_w2_ic.close()
	bl_ic_w3_ic.close()
	bl_ic_w4_ic.close()
	bl_ic_w5_ic.close()
	bl_ic_w6_ic.close()
	bl_ic_w1_com.close()
	bl_ic_w2_com.close()
	bl_ic_w3_com.close()
	bl_ic_w4_com.close()
	bl_ic_w5_com.close()
	bl_ic_w6_com.close()
	bl_ic_w1_pro.close()
	bl_ic_w2_pro.close()
	bl_ic_w3_pro.close()
	bl_ic_w4_pro.close()
	bl_ic_w5_pro.close()
	bl_ic_w6_pro.close()

	sps_ic_w1_ic.close()
	sps_ic_w2_ic.close()
	sps_ic_w3_ic.close()
	sps_ic_w4_ic.close()
	sps_ic_w5_ic.close()
	sps_ic_w6_ic.close()
	sps_ic_w1_smi.close()
	sps_ic_w2_smi.close()
	sps_ic_w3_smi.close()
	sps_ic_w4_smi.close()
	sps_ic_w5_smi.close()
	sps_ic_w6_smi.close()
	sps_ic_w1_sps.close()
	sps_ic_w2_sps.close()
	sps_ic_w3_sps.close()
	sps_ic_w4_sps.close()
	sps_ic_w5_sps.close()
	sps_ic_w6_sps.close()

	# EC50
	bl_ec_w1_ec.close()
	bl_ec_w2_ec.close()
	bl_ec_w3_ec.close()
	bl_ec_w4_ec.close()
	bl_ec_w5_ec.close()
	bl_ec_w6_ec.close()
	bl_ec_w1_com.close()
	bl_ec_w2_com.close()
	bl_ec_w3_com.close()
	bl_ec_w4_com.close()
	bl_ec_w5_com.close()
	bl_ec_w6_com.close()
	bl_ec_w1_pro.close()
	bl_ec_w2_pro.close()
	bl_ec_w3_pro.close()
	bl_ec_w4_pro.close()
	bl_ec_w5_pro.close()
	bl_ec_w6_pro.close()

	sps_ec_w1_ec.close()
	sps_ec_w2_ec.close()
	sps_ec_w3_ec.close()
	sps_ec_w4_ec.close()
	sps_ec_w5_ec.close()
	sps_ec_w6_ec.close()
	sps_ec_w1_smi.close()
	sps_ec_w2_smi.close()
	sps_ec_w3_smi.close()
	sps_ec_w4_smi.close()
	sps_ec_w5_smi.close()
	sps_ec_w6_smi.close()
	sps_ec_w1_sps.close()
	sps_ec_w2_sps.close()
	sps_ec_w3_sps.close()
	sps_ec_w4_sps.close()
	sps_ec_w5_sps.close()
	sps_ec_w6_sps.close()

	# Ki
	bl_ki_w1_ki.close()
	bl_ki_w2_ki.close()
	bl_ki_w3_ki.close()
	bl_ki_w4_ki.close()
	bl_ki_w5_ki.close()
	bl_ki_w6_ki.close()
	bl_ki_w1_com.close()
	bl_ki_w2_com.close()
	bl_ki_w3_com.close()
	bl_ki_w4_com.close()
	bl_ki_w5_com.close()
	bl_ki_w6_com.close()
	bl_ki_w1_pro.close()
	bl_ki_w2_pro.close()
	bl_ki_w3_pro.close()
	bl_ki_w4_pro.close()
	bl_ki_w5_pro.close()
	bl_ki_w6_pro.close()

	sps_ki_w1_ki.close()
	sps_ki_w2_ki.close()
	sps_ki_w3_ki.close()
	sps_ki_w4_ki.close()
	sps_ki_w5_ki.close()
	sps_ki_w6_ki.close()
	sps_ki_w1_smi.close()
	sps_ki_w2_smi.close()
	sps_ki_w3_smi.close()
	sps_ki_w4_smi.close()
	sps_ki_w5_smi.close()
	sps_ki_w6_smi.close()
	sps_ki_w1_sps.close()
	sps_ki_w2_sps.close()
	sps_ki_w3_sps.close()
	sps_ki_w4_sps.close()
	sps_ki_w5_sps.close()
	sps_ki_w6_sps.close()

	# Kd
	bl_kd_w1_kd.close()
	bl_kd_w2_kd.close()
	bl_kd_w3_kd.close()
	bl_kd_w4_kd.close()
	bl_kd_w5_kd.close()
	bl_kd_w6_kd.close()
	bl_kd_w1_com.close()
	bl_kd_w2_com.close()
	bl_kd_w3_com.close()
	bl_kd_w4_com.close()
	bl_kd_w5_com.close()
	bl_kd_w6_com.close()
	bl_kd_w1_pro.close()
	bl_kd_w2_pro.close()
	bl_kd_w3_pro.close()
	bl_kd_w4_pro.close()
	bl_kd_w5_pro.close()
	bl_kd_w6_pro.close()

	sps_kd_w1_kd.close()
	sps_kd_w2_kd.close()
	sps_kd_w3_kd.close()
	sps_kd_w4_kd.close()
	sps_kd_w5_kd.close()
	sps_kd_w6_kd.close()
	sps_kd_w1_smi.close()
	sps_kd_w2_smi.close()
	sps_kd_w3_smi.close()
	sps_kd_w4_smi.close()
	sps_kd_w5_smi.close()
	sps_kd_w6_smi.close()
	sps_kd_w1_sps.close()
	sps_kd_w2_sps.close()
	sps_kd_w3_sps.close()
	sps_kd_w4_sps.close()
	sps_kd_w5_sps.close()
	sps_kd_w6_sps.close()
	print('split conmplete\n\n\n')

	print('---------------------IC50 Statistic----------------------')
	print('Total number: %d' %countic)
	print('Train: %d' %count1_ic)
	print('GPCR: %d' %count2_ic)
	print('ER: %d' %count3_ic)
	print('channel: %d' %count4_ic)
	print('kinase: %d' %count5_ic)
	print('Test: %d' %count6_ic)
	print('No IC50: %d' % noicC)
	print('Dropped cases: %d\n' %dropicC)

	print('---------------------EC50 Statistic----------------------')
	print('Total number: %d' %countec)
	print('Train: %d' %count1_ec)
	print('GPCR: %d' %count2_ec)
	print('ER: %d' %count3_ec)
	print('channel: %d' %count4_ec)
	print('kinase: %d' %count5_ec)
	print('Test: %d' %count6_ec)
	print('No EC50: %d' % noecC)
	print('Dropped cases: %d\n' %dropecC)

	print('---------------------Ki Statistic----------------------')
	print('Total number: %d' % countki)
	print('Train: %d' % count1_ki)
	print('GPCR: %d' % count2_ki)
	print('ER: %d' % count3_ki)
	print('channel: %d' % count4_ki)
	print('kinase: %d' % count5_ki)
	print('Test: %d' % count6_ki)
	print('No Ki: %d' % nokiC)
	print('Dropped cases: %d\n' % dropkiC)

	print('---------------------Kd Statistic----------------------')
	print('Total number: %d' % countkd)
	print('Train: %d' % count1_kd)
	print('GPCR: %d' % count2_kd)
	print('ER: %d' % count3_kd)
	print('channel: %d' % count4_kd)
	print('kinase: %d' % count5_kd)
	print('Test: %d' % count6_kd)
	print('No Kd: %d' % nokdC)
	print('Dropped cases: %d\n' % dropkdC)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("usage: split.py <data directory>")
	main(sys.argv[1])
