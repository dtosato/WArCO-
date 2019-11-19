function Z_WARCO_classification_svm_cbh_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,patch_overlap,n_row,n_col,scale,d,cv_idx)
% Classifier training and testing procedure exploiting WARCO with the CBH distance.
%
% USAGE
%  Z_WARCO_classification_svm_cbh_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,patch_overlap,n_row,n_col,scale,d,cv_idx)
%
% INPUTS
%  train_dir     - array of training paths
%  test_dir      - array of testing paths
%  ds_name       - dataset (or db) name
%  J             - number of classes
%  K             - (K)cross-validation parameter
%  patch_dim     - WARCO (single) patch size
%  patch_overlap - WARCO patch overlapping
%  n_row         - (normalized) number of rows for the imgs in the db.
%  n_col         - (normalized) number of colums for the imgs in the db.
%  scale         - scale factor
%  d             - number of features used to build a covariance matrix
%  cv_idx        - cross-validation indexes
%
%
% EXAMPLE
% train_dir       = '../database/QML5PoseHeads/train'; % training path
% test_dir        = '../database/QML5PoseHeads/test';% testing path
% ds_name         = 'QML5PoseHeads';
% J               = 5;% number of classes
% K               = 8;% (K)cross-validation parameter
% d               = 13; % number of features used to build a covariance matrix
% patch_dim       = 16; % WARCO (single) patch size
% patch_overlap   = .5; % WARCO patch overlapping
% n_row           = 50; % (normalized) number of rows for the imgs in the db.
% n_col           = 50; % (normalized) number of colums for the imgs in the db.
%
% % collect images in the db
% classes_dir   =   dir(strcat(train_dir,'/Data_*'));
% num                 =   zeros(J,1);
% for j = 1:J
%     img_dir =   [dir(strcat(train_dir,'/',classes_dir(j).name,'/*.jpg'));
%         dir(strcat(train_dir,'/',classes_dir(j).name,'/*.bmp'));
%         dir(strcat(train_dir,'/',classes_dir(j).name,'/*.png'));];
%     num(j)  =   size(img_dir,1);
% end
% n = sum(num);
% cv_idx = crossvalind('Kfold', n, K); % build the cross-validation indexes
%
% % per scale test
% scale           = .5;% scale factor
% % WARCO using the Frobenious distance
% Z_ARCO_classification_svm_scale_deterministic(train_dir,test_dir,ds_name,J,K,patch_dim,...
%    patch_overlap,n_row,n_col,scale,d,cv_idx);
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% See also Z_WARCO_classification_svm_cbh_scale_deterministic
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.


% compute the actual scaled image size (to avoid rounding errors)
img_model     = imResample(zeros(n_row,n_col),scale);
[n_row,n_col] = size(img_model);
patch_dim     = round(patch_dim*scale);

% image Sym. DooG filtres computation
FB        = FbMake(2,6,0);
offset    = 0;
if scale ~= 1
    FB         = FbCrop(FB,round(1/scale)); % filters are scale dependent
end

% Id matrix (used for the log projection)
Id              = eye(d);

% output settings
common          = [ds_name '_r' num2str(n_row) '_c' num2str(n_col)...
    '_d' num2str(d) '_po' num2str(patch_overlap*100)...
    '_pd' num2str(patch_dim) '_po' num2str(scale*100) '_liblinear' ];
data_dir        = ['./dataset/' common];% working data directory (IT HAS TO HAVE SPACE to save the II representations)
common          = [common '_cbh' '_K' num2str(K) '_deterministic'];
experiment      = ['test/' date '/' common];% experiment output directory
disp(experiment)
mkdir(experiment);
mkdir(data_dir);


%% patches extraction
patches = X_patch(zeros(n_row,n_col),scale,patch_dim,patch_overlap);
wins    =   patches(1).wins; % number of WARCO patches
n_wins  =   size(wins,1);

%% training set building
n_scale = length(scale); % check for multiple resolution WARCO (typically is 1) if(exist([data_dir '/train_covariance_1.mat'],'file') == 0)
if(exist([data_dir '/train_covariance_1.mat'],'file') == 0)
    disp(' create train sets...')
    X_build_trainset_class(train_dir,data_dir,patches,Id,d,FB,offset,scale,n_scale);
end

%% WARCO computation
if(exist([experiment '/model.mat'],'file') > 0)
    load([experiment '/model']);
else
    disp('model building...')
    F_arr         = struct([]);
    %for l = 1:n_scale % multi-scale model
    %input/output data
    training_set = [];
    class_num_train = [];
    name              =   [data_dir '/' 'train_covariance_' num2str(1)];
    load(name);
    
    %output structures
    F             = cell(n_wins,1);  % classifiers
    w             = zeros(n_wins,J); % weights
    train_kernel  = cell(n_wins,1);  % kernels' elements (for testing purposes)
    kernel_reg    = cell(n_wins,1);  % regularization terms
    
    % patch/part classifier learning (CAN BE PARALLELIZED using matlabpool)
    %matlabpool(4)
    for t = 1:n_wins%par
        disp(['patch: ' num2str(t) '/' num2str(n_wins)])
        [ F{t}, train_kernel{t},kernel_reg{t},w(t,:)] = ...
            Y_patch_learning_cbh_deterministic(training_set,class_num_train,K,J,t,d,cv_idx);
    end
    %matlabpool close
    
    % save
    F_arr(1).U = U;
    F_arr(1).F = F;
    F_arr(1).w = w;
    F_arr(1).kernel = train_kernel;
    F_arr(1).reg = kernel_reg;
    F_arr(1).n_wins = n_wins;
    %end
    name              =   [experiment '/' 'model'];
    save(name,'F_arr','-v7.3');
end

%% testing set building
disp('classification...');
[testing_set,class_num_test] = X_build_testset_class...
    (test_dir,data_dir,patches,F_arr(1).U,Id,d,FB,offset,scale);

% general parameters
w             = F_arr(1).w;
n_wins        = F_arr(1).n_wins;
F             = F_arr(1).F;
train_kernel  = F_arr(1).kernel;
kernel_reg    = F_arr(1).reg;
RM            = [];
GT            = [];

for j = 1:J % for each class
    % per class parameters
    N                 = class_num_test(j); % number of testing examples
    testing_set_class = testing_set{j};
    log_posterior                = zeros(N,J); % multidimensional array of log-posterior
    log_posterior_tmp            = cell(n_wins,1);
    % patch/part classifier testing (CAN BE PARALLELIZED using matlabpool)
    %matlabpool(4)
    for t = 1:n_wins%par
        log_posterior_tmp{t} = Y_patch_testing_cbh(zeros(N,J),testing_set_class,kernel_reg{t},...
            train_kernel{t},F{t},w,N,j,J,t,d);
    end
    %matlabpool close
    for t = 1:n_wins
        log_posterior =   log_posterior + log_posterior_tmp{t};
    end
    % store results
    [~,y_out]    = max(log_posterior,[],2);
    RM           = [RM;y_out];
    GT           = [GT;ones(length(y_out),1)*j];
end

% confusion matrix computation
CM = confMatrix( GT, RM, J );
% display
figure(1)
confMatrixShow(CM)
confMatrixShow(CM,[], {'FontSize',16},[], 0 )
avg_acc = mean(diag(CM)./sum(CM,2));
title(['confusion matrix (avg accuracy' num2str(avg_acc) ')'],'FontSize',16 );
% store results
name              =   [experiment '/' common];
saveas(1,[name '.fig'], 'fig');
name         =   [experiment '/' common];
save(name,'CM','avg_acc','-v7.3');


end

