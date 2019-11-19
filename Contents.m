% WARCO (Weighted ARray of COvariance matrices)
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
%
% classification.
%   X_build_testset_class               - Build the classification testing set according to the patch-based model WARCO.
%   X_build_trainset_class              - Build the classification training set according to the patch-based model WARCO.
%   Y_ESVM_Train_deterministic          - Learn a multiclass kernel SVM in the space of SPD matrices using the Frobenius distance.
%   Y_GSVM_Train_deterministic          - Learn a multiclass kernel SVM in the space of SPD matrices using the closed form Riemanninan distance.
%   Y_MSVM_Train_deterministic          - Learn a multiclass kernel SVM in the space of SPD matrices using the CBH distance.
%   Y_patch_learning_cbh_deterministic  - Learn a patch model of WARCO with the CBH distance.
%   Y_patch_learning_deterministic      - Learn a patch model of WARCO with the Frobenius distance.
%   Y_patch_testing                     - Test a patch model of WARCO with the Frobenius distance.
%   Y_patch_testing                     - Test a patch model of WARCO with the CBH distance.
%   Z_WARCO_classification_svm_cbh_scale_deterministic - Classifier training and testing procedure exploiting WARCO with the CBH distance.
%   Z_WARCO_classification_svm_scale_deterministic     - Classifier training and testing procedure exploiting WARCO with the Frobenius distance.
%
% common.
%   X_CovCal_d                          - Compute a covariance descriptors.
%   X_expp                              - Compute the exponential map associated to the Riemaniann metric of SPD matrices.
%   X_logp                              - Compute the logarithmic map associated to the Riemaniann metric of SPD matrices.
%   X_patch                             - Compute the WARCO regular grid of patches.
%   Y_BuildFf                           - Compute a set of image features to build WARCO.
%
% manifoldAnalysis.
%   test_curvature_scale_paired         - Test the curvature of a SPD dataset made by the covariance (SPD) matrices.
%   Z_Curvature_paired                  - Curvature analysis of a SPD dataset. 
%
% regression.
%   X_build_trainset_idiap_reg_svm      - Build the regression training set according to the patch-based model WARCO for the IDIAP data.
%   X_build_trainset_caviar_reg_svm     - Build the regression training set according to the patch-based model WARCO for the CAVIAR data.
%   X_build_testset_idiap_reg_svm       - Build the regression testing set according to the patch-based model WARCO for the IDIAP data.
%   X_build_testset_caviar_reg_svm      - Build the regression testing set according to the patch-based model WARCO for the CAVIAR data.
%   Y_ESVR_Train_deterministic          - Learn a kernel SVR in the space of SPD matrices using the Frobenius distance.
%   Y_MSVR_Train_deterministic          - Learn a kernel SVR in the space of SPD matrices using the CBH distance.
%   Z_WARCO_regression_caviar_svm_cbh_scale_deterministic - Regressor training and testing procedure exploiting WARCO with the CBH distance for the CAVIAR data.
%   Z_WARCO_regression_caviar_svm_scale_deterministic     - Regressor training and testing procedure exploiting WARCO with the Frobenius distance for the CAVIAR data.
%   Z_WARCO_regression_idiap_svm_cbh_scale_deterministic  - Regressor training and testing procedure exploiting WARCO with the CBH distance for the IDIAP data.
%   Z_WARCO_regression_idiap_svm_scale_deterministic      - Regressor training and testing procedure exploiting WARCO with the Frobenius distance for the CAVIAR data.
%
% testing.
% test_*                                 - Testing scripts for a dataset at different image scales. Be sure to set the right data path into the testing scripts.
% 

% Modifications:
% 22-11-2014

