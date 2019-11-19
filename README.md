# WaRCO

## Introduction

This is an implementation of a multi-class classifier and regressor based on a patch based model termed WARCO (Weighted ARray of COvariance matrices) able to deal with low resolution objects. WARCO is fully implemented in Matlab. The software was tested on different versions of Linux and Windows using Matlab versions R2010a/b (x64) and R2011a(x64). There may be compatibility issues with other versions of Matlab.

## Requirements

* Matlab 2010 or higher.
* Several algorithms require the Images toolbox
  by MathWorks.
* Libsvm <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>
* Liblinear <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>
* Piotr Dollar Toolbox <http://vision.ucsd.edu/~pdollar/toolbox/doc/>

## Installation  

* Create a directory called 'WARCO' and copy the code there.
* Put all the additional tooboxes (Libsvm, Liblinear, Piotr Dollar Toolbox) in './utils'.

# Datasets

* [CAVIARShoppingCenterFull](https://drive.google.com/open?id=0B0MZ5gr7K36SSWF6bmkwTzNXRm8)
* [CAVIARShoppingCenterFullOccl](https://drive.google.com/open?id=0B0MZ5gr7K36SUHh3c2VkOVJ3LVU)
* [HIIT6HeadPose](https://drive.google.com/open?id=0B0MZ5gr7K36SeEhyQ0I1QlVDVHM)
* [HOC](https://drive.google.com/open?id=0B0MZ5gr7K36SWHNjODB4bW5tZzQ)
* [HOCoffee](https://drive.google.com/open?id=0B0MZ5gr7K36SR0YxRTF6NGVKUjg)
* [IHDPHeadPose](https://drive.google.com/open?id=0B0MZ5gr7K36Sd3kzNUxUWndFakE)
* [QMUL4PoseHeads](https://drive.google.com/open?id=0B0MZ5gr7K36SVFVjYVBpaTFuRFU)
* [QMUL5PoseHeads](https://drive.google.com/open?id=0B0MZ5gr7K36Sb3dyUlN4d0hBa28)

## How to start

* Create a the directory 'WARCO\database' and unzip all the datasets in that folder.
* Add the paths of WARCO to Matlab with 'addpath(genpath(WARCO\path));'
* Type 'help WARCO' to know how WARCO is organized.
* Switch to the WARCO folder typing 'cd WARCO\path;'
* Use the 'test_*' scripts to test WARCO. Be sure to set the right data path into the testing scripts.

## Reference

* D. Tosato, M. Spera, M. Cristani, V. Murino, _Characterizing humans on Riemannian manifolds_, IEEE  Trans. PAMI, Preprint 2011.
