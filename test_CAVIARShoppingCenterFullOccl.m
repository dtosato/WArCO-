% Testing Session of the regression framework exploiting WARCO on the CAVIAR dataset
% containing OCCLUDED images.
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
%%
addpath(pathdef) 
ccc
addpath(pathdef) 
ccc

% parametres
root            = './database/CAVIARShoppingCenterFullOccl'; % dataset path
train_dir       = [root '/train']; % training path
test_dir        = [root '/test']; % testing path
ds_name         = 'CAVIARShoppingCenterFullOccl';
K               = 8;
d               = 13;
patch_dim       = 16;
patch_overlap   = .5;
n_row           = 50;
n_col           = 50;

% load orientation info
name   =  [root '/or_label'];
load(name);
n      = length(or_label_train);
cv_idx = crossvalind('Kfold', n, K);% build the cross-validation indexes

% per scale test
scale           = .5;% scale factor
% WARCO using the Frobenious distance
Z_WARCO_regression_caviar_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);
% WARCO using the CBH distance
Z_WARCO_regression_caviar_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

scale           = .75;
Z_WARCO_regression_caviar_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

Z_WARCO_regression_caviar_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

scale           = 1;
Z_WARCO_regression_caviar_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

Z_WARCO_regression_caviar_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);
disp('end.')
