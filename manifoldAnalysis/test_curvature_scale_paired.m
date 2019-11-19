% Test the curvature of a SPD dataset made by the covariance (SPD) matrices.
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

% db_name = 'VIPER4PoseHuman3';
% train_dir = ['../../database/' db_name '/train'];
% n_row           = 128; %(normalized) number of rows for the imgs in the db. 
% n_col           = 48; % (normalized) number of colums for the imgs in the db.
% patch_overlap   = .5; % WARCO patch overlapping 
% patch_dim       = 16; % WARCO (single) patch size 
% sample          = 128; % (img) sampling step
% % per scale test
% scale           = 1;% scale factor
% % curvature analysis
% Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
% scale           = .75;
% Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
% scale           = .50;
% Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'ETHZ4HumanPose';
train_dir = ['../../database/' db_name '/train'];
patch_dim       = 24;
patch_overlap   = .5;
n_row           = 132;
n_col           = 62;
sample          = 128;
scale           = 1;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'QML4PoseHeads';
train_dir = ['../../database/' db_name '/train'];
n_row           = 50;
n_col           = 50;
patch_overlap   = .5;
patch_dim       = 16;
sample          = 128;
scale           = 1;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'QML5PoseHeads';
train_dir = ['../../database/' db_name '/train'];
n_row           = 50;
n_col           = 50;
patch_overlap   = .5;
patch_dim       = 16;
sample          = 128;
scale           = 1;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'IIT6HeadPose';
train_dir = ['../../database/' db_name '/train'];
n_row           = 50;
n_col           = 50;
patch_overlap   = .5;
patch_dim       = 16;
sample          = 128;
scale           = 1;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'Sestri6HeadPose';
train_dir = ['../../database/' db_name '/train'];
n_row           = 50;
n_col           = 50;
patch_overlap   = .5;
patch_dim       = 16;
sample          = 128;
scale           = 1;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'CAVIARShoppingCenterFull_1';
train_dir = ['../../database/' db_name '/train'];
n_row           = 50;
n_col           = 50;
patch_overlap   = .5;
patch_dim       = 16;
sample          = 128;
scale           = 1;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'CAVIARShoppingCenterFullOccl_1';
train_dir = ['../../database/' db_name '/train'];
n_row           = 50;
n_col           = 50;
patch_overlap   = .5;
patch_dim       = 16;
sample          = 128;
scale           = 1;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

db_name = 'IHDP_10smp_flip_alpha_allSeq_pose_reg';
train_dir = ['../../database/' db_name '/train'];
n_row           = 75;
n_col           = 75;
patch_overlap   = .5;
patch_dim       = 24;
sample          = 128;
scale           = 1;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .75;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
scale           = .50;
Z_Curvature_paired_reg(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)

disp('end!')



