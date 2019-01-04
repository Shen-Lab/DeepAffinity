In this folder, we provide two sets of data and their script for your information.

## Dataset:
1. Split dataset for DeepAffinity model.
	Split dataset is included in "dataset" folder with 4 compressed subfolders and they are split based on their labels: EC50， IC50， Ki and Kd. In each folder, there are 6 files which have cleared original data. For details, you can refer to introduction of BindingDB at:
	https://www.bindingdb.org/bind/chemsearch/marvin/BindingDB-TSV-Format.pdf
	
	If you are trying to find the data we used for our model, you can directly look at other two folders.
	In "baseline" folder, there will be 4 types of classes completely withheld from training, which are channel, GPCR, ER and kinase, classified based on their biological functions. These four classes constitute the "generalization" set in our manuscript.  For other pairs, we put 70% in training set and 30% in test set. For each class, there are 4 files:
	*_compound_fingerprint: fingerprint features of compounds, each one of them is a 881 digit binary number. 
	*_ic50/ec50/ki/kd: -log10(M) of label value
	*_protein_domain: domain feature of proteins, each one of them is a 16712 digits binary number.
	*_protein_seq: protein sequences for each pair.
	
In "SPS" folder, there are 3 files for each class:
	*_smile: canonical SMILE format of compounds
	*_ic50/ec50/ki/kd: -log10(M) of label value
	*_sps: Structural Property Sequence(SPS) format of proteins

2. Cleared dataset without split.
	Since the result folder is too large, you can download it from:
	https://drive.google.com/open?id=1_msEbSh_YZr0NLSR_DJ_xWE9FlqBlMV9

You will get several files:
	1) *_protein_compound_pair.tsv
	* represents the measure method for protein compound pairs. It can be IC50, EC50, Ki and Kd. In this file, it contains DeepAffinity protein ID (4 digits random number and Uppercase letters), protein Uniprot ID, DeepAffinity compound ID (4 digits random number and Lowercase letters), compound CID and measure value. DeepAffinity ID can be used to retrieve corresponding representation format. Those measure value is calculated by -log10(M) and the detail can be found in the supplement of our paper.
	
	2) dcid_smi.tsv
	DeepAffinity compound ID and its corresponding canonical SMILE format of compounds.

	3) dcid_fingerprint.tsv
	DeepAffinity compound ID and its corresponding fingerprint feature of compound. Each one of them is a 881 digit binary number. 
	
	4) dpid_seq.tsv
	DeepAffinity protein ID and its corresponding protein sequence.

	5) dpid_sps.tsv
	DeepAffinity protein ID and its corresponding protein SPS representation format.

	6) dpid_dom.tsv
	DeepAffinity protein ID and its corresponding protein domain features. Each one of them is a 16712 digits binary number.

	7) SDF folder
	This folder contains the graphic data of compounds, which is SDF format. Each file of this folder is named by correpsonding DeepAffinity compound ID. 
--------------------------------------------------------------------------------------------------------------------------

## Data Script: 
	If you would like to generate dataset by yourself or generate based on your own dataset, you can refer to "script" folder for details.
