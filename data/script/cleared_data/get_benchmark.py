#coding:utf-8
import re
import random
import os
import sys
import csv
import string
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
	dropic = 0
	dropki = 0
	dropkd = 0
	dropec = 0
	noicC = 0
	nokiC = 0
	nokdC = 0
	noecC = 0
	dropicC = 0
	dropkiC = 0
	dropkdC = 0
	dropecC = 0
	random.seed(1234)
	# for folder in ['IC50', 'Ki', 'Kd', 'EC50']:
	# 	if not os.path.exists('./' + folder):
	# 		os.makedirs('./' + folder)
	# 	if not os.path.exists('./' + folder + '/baseline'):
	# 		os.makedirs('./' + folder + '/baseline')
	# 	if not os.path.exists('./' + folder + '/SPS'):
	# 		os.makedirs('./' + folder + '/SPS')
	
	ic_protein_compound = open('IC50_protein_compound_pair.tsv', 'w+')
	ec_protein_compound = open('EC50_protein_compound_pair.tsv', 'w+')
	ki_protein_compound = open('Ki_protein_compound_pair.tsv', 'w+')
	kd_protein_compound = open('Kd_protein_compound_pair.tsv', 'w+')
	dcid_fea = open('dcid_fingerprint.tsv', 'w+')
	dcid_smi = open('dcid_smi.tsv', 'w+')
	dpid_sps = open('dpid_sps.tsv', 'w+')
	dpid_seq = open('dpid_seq.tsv', 'w+')
	dpid_dom = open('dpid_dom.tsv', 'w+')

	ic_writer = csv.writer(ic_protein_compound, dialect='excel-tab')
	ec_writer = csv.writer(ec_protein_compound, dialect='excel-tab')
	ki_writer = csv.writer(ki_protein_compound, dialect='excel-tab')
	kd_writer = csv.writer(kd_protein_compound, dialect='excel-tab')
	dcid_fea_writer = csv.writer(dcid_fea, dialect='excel-tab')
	dcid_smi_writer = csv.writer(dcid_smi, dialect='excel-tab')
	dpid_sps_writer = csv.writer(dpid_sps, dialect='excel-tab')
	dpid_seq_writer = csv.writer(dpid_seq, dialect='excel-tab')
	dpid_dom_writer = csv.writer(dpid_dom, dialect='excel-tab')

	dcid_fea_header = ["DeepAffinity Compound ID","Fingerprint Feature"]
	dcid_smi_header = ["DeepAffinity Compound ID","Canonical SMILE"]
	dpid_fea_header = ["DeepAffinity Protein ID","SPS Format"]
	dpid_seq_header = ["DeepAffinity Protein ID","Sequence"]
	dpid_dom_header = ["DeepAffinity Protein ID","Domain Features"]
	dcid_fea_writer.writerow(dcid_fea_header)
	dcid_smi_writer.writerow(dcid_smi_header)
	dpid_sps_writer.writerow(dpid_fea_header)
	dpid_seq_writer.writerow(dpid_seq_header)
	dpid_dom_writer.writerow(dpid_dom_header)

	#Loading dictionary

	seq_id = {}
	dcid_id = {}

	groupedDir = './data/protein_grouped_finalPresentation' # protein_grouped directory
	proteinDom = './data/Seq_Vector'                        # protein_feature directory
	compoundFea = './data/CID_Smi_Feature'                  # compound_Feature directory

	with open(groupedDir) as f:
		mark = 0
		idSet = set()
		for line in f:
			if mark == 0:
				protein = line.strip()
				mark = 1
			else:
				sps = line.strip()
				mark = 0
				while 1:
					pro_id = ''.join(random.choice(string.digits+string.ascii_uppercase) for i in range(4))
					if pro_id not in idSet:
						idSet.add(pro_id)
						break
				dpid_sps_writer.writerow([pro_id,sps])
				dpid_seq_writer.writerow([pro_id,protein])
				seq_id[protein] = pro_id

	with open(proteinDom) as f:
		mark = 0
		for line in f:
			if mark == 0:
				protein = line.strip()
				mark = 1
			else:
				dom = line.strip()
				mark = 0
				dpid_dom_writer.writerow([seq_id[protein],pos2vec(dom)])
				# dpid_dom_writer.writerow([seq_id[protein],dom])

	with open(compoundFea) as f:
		idSet = set()
		for line in f:
			line = line.strip()
			if line == '> <PUBCHEM_COMPOUND_CID>':
				cid = f.next().strip()
			elif line == '> <PUBCHEM_CACTVS_SUBSKEYS>':
				fea = f.next().strip()
			elif line == '> <PUBCHEM_OPENEYE_CAN_SMILES>':
				smi = f.next().strip()
			elif line == '$$$$':
				while 1:
					smi_id = ''.join(random.choice(string.digits+string.ascii_lowercase) for i in range(4))
					if smi_id not in idSet:
						idSet.add(smi_id)
						break
				dcid_fea_writer.writerow([smi_id,fea])
				dcid_smi_writer.writerow([smi_id,smi])
				dcid_id[cid] = smi_id
	print 'Dictionary loaded'

	dirpath = "SDF"
	os.system("mkdir "+dirpath)
	for i in range(1,4):
		resultFile = './data/CID' + str(i) + '.sdf'
		with open(resultFile) as f:
			sdf = ''
			mark = 0
			for line in f:
				if mark == 0:
					while line.strip() != "M  END":
						sdf += line
						line = f.next()
					sdf += line.strip()
					mark = 1
				else:
					if line.strip() == '> <PUBCHEM_COMPOUND_CID>':
						cid = f.next().strip()
					elif line.strip() == '$$$$':
						smi_id = dcid_id[cid]
						w = open(dirpath+"/"+smi_id,'w+')
						w.write(sdf)
						w.close()
						sdf = ''
						mark = 0






	with open(dataDir) as f:
		reader = csv.reader(f, delimiter="\t",quoting=csv.QUOTE_NONE)
		header = reader.next()
		ic_header = ["DeepAffinity Protein ID", "Uniprot ID", "DeepAffinity Compound ID", "CID", "pIC50_[M]"]
		ec_header = ["DeepAffinity Protein ID", "Uniprot ID", "DeepAffinity Compound ID", "CID", "pEC50_[M]"]
		ki_header = ["DeepAffinity Protein ID", "Uniprot ID", "DeepAffinity Compound ID", "CID", "pKi_[M]"]
		kd_header = ["DeepAffinity Protein ID", "Uniprot ID", "DeepAffinity Compound ID", "CID", "pKd_[M]"]
		
		ic_writer.writerow(ic_header)
		ec_writer.writerow(ec_header)
		ki_writer.writerow(ki_header)
		kd_writer.writerow(kd_header)
		
		for row in reader:
			smi = row[1]
			# bd_name = row[6]
			# uni_name = row[39]
			ki = row[8]
			ic = row[9]
			kd = row[10]
			ec = row[11]
			cid = row[28]
			seq = row[37]
			uid = row[41]
			# if seq not in seq_dom:
			# 	print row
			# domain = pos2vec(seq_dom[seq])
			#domain = seq_dom[seq]
			# feature = cid_fea[cid]
			# sps = seq_sps[seq]
			dpid = seq_id[seq]
			smi_id = dcid_id[cid]
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

			# row[8] = ki
			# row[9] = ic
			# row[10] = kd
			# row[11] = ec
		#output to files
			if dropic == 0:
				ic_writer.writerow([dpid, uid, smi_id, cid, ic])
				countic += 1
			if dropec == 0:
				ec_writer.writerow([dpid, uid, smi_id, cid, ec])
				countec += 1
			if dropki == 0:
				ki_writer.writerow([dpid, uid, smi_id, cid, ki])
				countki += 1
			if dropkd == 0:
				kd_writer.writerow([dpid, uid, smi_id, cid, kd])
				countkd += 1

			dropec = 0
			dropic = 0
			dropki = 0
			dropkd = 0

	
	ic_protein_compound.close()
	ec_protein_compound.close()
	ki_protein_compound.close()
	kd_protein_compound.close()
	dcid_fea.close()
	dcid_smi.close()
	dpid_sps.close()
	dpid_seq.close()
	dpid_dom.close()

	print 'split conmplete\n\n\n'

	print '---------------------IC50 Statistic----------------------'
	print 'Total number: %d' %countic
	# print 'Train: %d' %count1_ic
	# print 'GPCR: %d' %count2_ic
	# print 'ER: %d' %count3_ic
	# print 'channel: %d' %count4_ic
	# print 'kinase: %d' %count5_ic
	# print 'Test: %d' %count6_ic
	print 'No IC50: %d' % noicC
	print 'Dropped cases: %d\n' %dropicC

	print '---------------------EC50 Statistic----------------------'
	print 'Total number: %d' %countec
	# print 'Train: %d' %count1_ec
	# print 'GPCR: %d' %count2_ec
	# print 'ER: %d' %count3_ec
	# print 'channel: %d' %count4_ec
	# print 'kinase: %d' %count5_ec
	# print 'Test: %d' %count6_ec
	print 'No EC50: %d' % noecC
	print 'Dropped cases: %d\n' %dropecC

	print '---------------------Ki Statistic----------------------'
	print 'Total number: %d' % countki
	# print 'Train: %d' % count1_ki
	# print 'GPCR: %d' % count2_ki
	# print 'ER: %d' % count3_ki
	# print 'channel: %d' % count4_ki
	# print 'kinase: %d' % count5_ki
	# print 'Test: %d' % count6_ki
	print 'No Ki: %d' % nokiC
	print 'Dropped cases: %d\n' % dropkiC

	print '---------------------Kd Statistic----------------------'
	print 'Total number: %d' % countkd
	# print 'Train: %d' % count1_kd
	# print 'GPCR: %d' % count2_kd
	# print 'ER: %d' % count3_kd
	# print 'channel: %d' % count4_kd
	# print 'kinase: %d' % count5_kd
	# print 'Test: %d' % count6_kd
	print 'No Kd: %d' % nokdC
	print 'Dropped cases: %d\n' % dropkdC

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "usage: split.py <data directory>"
	main(sys.argv[1])
