% Testing Session of the classification framework exploiting WARCO on the HIIT6HeadPose dataset.
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
addpath(pathdef)
ccc

% parametres
train_dir       = './database/HIIT6HeadPose/train'; % training path
test_dir        = './database/HIIT6HeadPose/test'; % testing path
ds_name         = 'HIIT6HeadPose';
J               = 6;
K               = 8;% (K)cross-validation parameter
d               = 13; % number of features used to build a covariance matrix
patch_dim       = 16; % WARCO (single) patch size 
patch_overlap   = .5; % WARCO patch overlapping 
n_row           = 50; % (normalized) number of rows for the imgs in the db. 
n_col           = 50; % (normalized) number of colums for the imgs in the db.
% collect images in the db
classes_dir   =   dir(strcat(train_dir,'/Data_*'));
num                 =   zeros(J,1);
for j = 1:J
    img_dir =   [dir(strcat(train_dir,'/',classes_dir(j).name,'/*.jpg'));
        dir(strcat(train_dir,'/',classes_dir(j).name,'/*.bmp'));
        dir(strcat(train_dir,'/',classes_dir(j).name,'/*.png'));];
    num(j)  =   size(img_dir,1);
end
n = sum(num);
cv_idx = crossvalind('Kfold', n, K); % build the cross-validation indexes

% per scale test
scale           = .5;% scale factor
% WARCO using the Frobenious distance
Z_WARCO_classification_svm_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,...
   patch_overlap,n_row,n_col,scale,d,cv_idx);
% WARCO using the CBH distance
Z_WARCO_classification_svm_cbh_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,...
   patch_overlap,n_row,n_col,scale,d,cv_idx);

scale           = .75;
Z_WARCO_classification_svm_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,...
   patch_overlap,n_row,n_col,scale,d,cv_idx);

Z_WARCO_classification_svm_cbh_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,...
   patch_overlap,n_row,n_col,scale,d,cv_idx);

scale           = 1;
Z_WARCO_classification_svm_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,...
   patch_overlap,n_row,n_col,scale,d,cv_idx);

Z_WARCO_classification_svm_cbh_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,...
    patch_overlap,n_row,n_col,scale,d,cv_idx);

disp('end.')



