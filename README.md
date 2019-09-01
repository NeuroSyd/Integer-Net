# Integer-Net

##### Code to reproduce results reported in our paper published as:
Truong, N. D., A. D. Nguyen, L. Kuhlmann, M. R. Bonyadi, J. Yang, S. Ippolito, and O. Kavehei (2018). Integer Convolutional Neural Network for Seizure Detection. *IEEE Journal on Emerging and Selected Topics in Circuits and Systems* 8.4, 849-857. DOI:10.1109/JETCAS.2018.2842761

#### Requirements
* h5py (2.7.1)
* hickle (2.1.0)
* Keras (2.0.6)
* matplotlib (1.3.1)
* mne (0.11.0)
* pandas (0.21.0)
* scikit-learn (0.19.1)
* scipy (1.0.0)
* tensorflow-gpu (1.4.1)

#### How to run the code
1. Set the paths in \*.json files. Copy files in folder "copy-to-CHBMIT-folder" to your CHBMIT dataset folder.

2. Run the code
```console
python main.py --dataset DATASET --mode cv --model MODEL --bits BITS
```
##### where: </br>
* MODE: cv, test
  * cv: leave-one-seizure-out cross-validation
  * test: ~1/3 of last seizures used for test, interictal signals are split accordingly
* DATASET: FB, CHBMIT, Kaggle2014Pred
* BITS: an integer value (only applied when MODEL==int) 
