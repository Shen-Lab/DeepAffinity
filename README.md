# DeepAffinity
Drug discovery demands rapid quantification of compound-protein interaction  (CPI). However, there is a lack of methods that can predict compound-protein affinity from sequences alone with high applicability, accuracy, and interpretability. We present a integration of domain knowledges and learning-based approaches. Under novel representations of structurally-annotated protein sequences, a semi-supervised deep learning model that unifies recurrent and convolutional neural networks has been proposed to exploit both unlabeled and labeled data, for jointly encoding molecular representations and predicting affinities. Performances for new protein classes with few labeled data are further improved by transfer learning. Furthermore, novel attention mechanisms are developed and embedded to our model to add to its interpretability. Lastly, alternative representations using protein sequences or compound graphs and a unified RNN/GCNN-CNN model using graph CNN (GCNN) are also explored to reveal algorithmic challenges ahead.
## Pre-requisite:
* Tensorflow-gpu v1.1
* Python 3.6
* [TFLearn](http://tflearn.org/) v0.3
* [Scikit-learn](https://scikit-learn.org/stable/) v0.19

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

## Citation:
If you find the code useful for your research, please consider citing our paper:
```
@article{karimi2018deepaffinity,
  title={DeepAffinity: Interpretable Deep Learning of Compound-Protein Affinity through Unified Recurrent and Convolutional Neural Networks},
  author={Karimi, Mostafa and Wu, Di and Wang, Zhangyang and Shen, Yang},
  journal={arXiv preprint arXiv:1806.07537},
  year={2018}
}
```

## Contacts:
Yang Shen: yshen@tamu.edu

Mostafa Karimi: mostafa_karimi@tamu.edu
