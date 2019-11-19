% Testing Session of the regression framework exploiting WARCO on the IHDPHeadPose dataset.
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions
%%
addpath(pathdef) 
ccc

% parametres
root            = './database/IHDPHeadPose'; % dataset path
train_dir       = [root '/train']; % training path
test_dir        = [root '/test']; % testing path
ds_name         = 'IHDPHeadPose';
K               = 8;
d               = 13;
patch_dim       = 24;
patch_overlap   = .5;
n_row           = 75;
n_col           = 75;


% load orientation info
name   =  [root '/or_label'];
load(name);
n      = length(or_label_train);

% build the cross-validation indexes
if(exist([root '/cv_'  num2str(K)  '.mat'],'file') == 0)
cv_idx = crossvalind('Kfold', n, K);
save([root '/cv_'  num2str(K)  '.mat'],'cv_idx','-v7.3');
else
disp('load CV IDXs')
load([root '/cv_'  num2str(K)  '.mat']);
end

% per scale test
scale           = .5;% scale factor
% WARCO using the Frobenious distance
Z_WARCO_regression_idiap_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);
% WARCO using the CBH distance
Z_WARCO_regression_idiap_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

scale           = .75;
Z_WARCO_regression_idiap_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

Z_WARCO_regression_idiap_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

scale           = 1;
Z_WARCO_regression_idiap_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

Z_WARCO_regression_idiap_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);
disp('end.')


