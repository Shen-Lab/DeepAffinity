import urllib.request as request
import os
import re

def test_url(het_list, pid,ligtype,ligno):
	base_url = 'https://www.ebi.ac.uk/thornton-srv/databases/cgi-bin/pdbsum/GetLigInt.pl?pdb={}&ligtype={}&ligno={}'
	if ligtype < 10:
		ligtype_str = '0' + str(ligtype)
	else:
		ligtype_str = str(ligtype)

	if ligno < 10:
		ligno_str = '0' + str(ligno)
	else:
		ligno_str = str(ligno)

	url = base_url.format(pid.lower(),ligtype_str,ligno_str)
	request.urlretrieve(url, "temp")
	f = open("temp")
	het = ''
	for line in f:
		m = re.search(r'PDB code:[0-9a-z ]+Ligand ([a-zA-Z0-9-]+)',line.strip())
		if m:
			het = m.group(1).strip()
			break
	if het == '':
		os.system('rm temp')
		return False
	elif het not in het_list:
		os.system('rm temp')
		return True
	if ligno == 1:
		os.system('mv temp contact/{}_{}'.format(pid,het))
	elif ligno == 2:
		os.system('mv contact/{}_{} contact/{}_{}_1'.format(pid,het,pid,het))
		os.system('mv temp contact/{}_{}_{}'.format(pid,het,ligno))
	else:
		os.system('mv temp contact/{}_{}_{}'.format(pid,het,ligno))
	return True

if "contact" in os.listdir():
	os.system('rm -r contact')
os.system('mkdir contact')
uid_het = {}
# pid_set = set()
pid_het = {}
with open("uid_het_cid_useq_smi_ki") as f:
	for line in f:
		line=line.strip()
		if line != '':
			line = line.split('\t')
			uid = line[0]
			het = line[1]
			if het != '':
				if uid in uid_het:
					uid_het[uid].append(het)
				else:
					uid_het[uid] = [het]

with open("pid_uid") as f:
	for line in f:
		s = line.strip().split('\t')
		pid = s[0]
		uid = s[1]
		if uid in uid_het:
			het_list = uid_het[uid]
			if pid in pid_het:
				pid_het[pid].update(het_list)
			else:
				pid_het[pid] = set(het_list)

print(len(pid_het))
print("Dictionary Loaded!")


for pid in pid_het:
	ligtype = 1
	ligno = 2
	while test_url(pid_het[pid],pid,ligtype,1):
		while test_url(pid_het[pid],pid,ligtype,ligno):
			ligno += 1
		ligtype += 1
		ligno = 2
