# DeepAffinity
Drug discovery demands rapid quantification of compound-protein interaction  (CPI). However, there is a lack of methods that can predict compound-protein affinity from sequences alone with high applicability, accuracy, and interpretability. We present a integration of domain knowledges and learning-based approaches. Under novel representations of structurally-annotated protein sequences, a semi-supervised deep learning model that unifies recurrent and convolutional neural networks has been proposed to exploit both unlabeled and labeled data, for jointly encoding molecular representations and predicting affinities. Performances for new protein classes with few labeled data are further improved by transfer learning. Furthermore, novel attention mechanisms are developed and embedded to our model to add to its interpretability. Lastly, alternative representations using protein sequences or compound graphs and a unified RNN/GCNN-CNN model using graph CNN (GCNN) are also explored to reveal algorithmic challenges ahead.

![Training-Process](/main_fig.png)

## Pre-requisite:
* Tensorflow-gpu v1.1
* Python 3.6
* [TFLearn](http://tflearn.org/) v0.3
* [Scikit-learn](https://scikit-learn.org/stable/) v0.19
* Anaconda 3/5.0.0.1
* We have already provided our environment list as environment.yml. You can create your own environment by:
```
conda env create -n envname -f environment.yml
```
## Table of contents:
* **data_script**: Contain the supervised learning datasets(pIC50, pKi, pEC50, and pKd)
* **Seq2seq_models**: Contain auto-encoder seq2seq models and their data for both SPS and SMILE representations
* **baseline_models**: Contain shallow models for both Pfam/pubchem features and features generated from the encoder part of seq2seq model
* **Separate_models**: Contain deep learning model for features generated from the encoder part of seq2seq model
* **Joint_models**: Contain all the joint models including:
	* Separate attention mechanism
	* Marginalized attention mechanism
	* Joint attention mechanism
	* Graph convolution neural network (GCNN) with separate attention mechanism

## Testing the model
To test DeepAffinity for new dataset, please follow the steps below:
* Download the checkpoints trained based on training set of IC50 from the following [link](https://drive.google.com/drive/folders/1Pwn8uTyHNig4G2JDy0TErzH9hVacSadt?usp=sharing)
* Download the checkpoints trained based on the whole dataset of IC50 from the following [link](https://drive.google.com/drive/folders/1XAnXHSRnrO8DGA1drW3YnmaBaCihdiP5?usp=sharing)
* Copy the checkpoints to the folder "Joint_models/joint_attention/testing/"
* Put your data in folder "Joint_models/joint_attention/testing/data"
* cd Joint_models/joint_attention/testing/
* Run the Python code joint-Model-test.py
 

## Note:
We recommend referring to PubChem for canonical SMILES for compounds.  

## Citation:
If you find the code useful for your research, please consider citing our paper:
```
@article{karimi2019deepaffinity,
  title={DeepAffinity: interpretable deep learning of compound--protein affinity through unified recurrent and convolutional neural networks},
  author={Karimi, Mostafa and Wu, Di and Wang, Zhangyang and Shen, Yang},
  journal={Bioinformatics},
  volume={35},
  number={18},
  pages={3329--3338},
  year={2019},
  publisher={Oxford University Press}
}
```

## Contacts:
Yang Shen: yshen@tamu.edu

Di Wu: wudi930325@gmail.com

Mostafa Karimi: mostafa_karimi@tamu.edu
