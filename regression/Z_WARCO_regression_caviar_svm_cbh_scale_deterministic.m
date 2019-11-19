function Z_WARCO_regression_caviar_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,patch_overlap,n_row,n_col,scale,d,cv_idx)
% Regressor training and testing procedure exploiting WARCO with the CBH distance for the CAVIAR data.
%
% USAGE
%  Z_WARCO_regression_caviar_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,patch_overlap,n_row,n_col,scale,d,cv_idx)
%
% INPUTS
%  root          - root path
%  train_dir     - training path
%  test_dir      - testing path
%  ds_name       - dataset (or db) name
%  K             - (K)cross-validation parameter
%  patch_dim     - WARCO (single) patch size
%  patch_overlap - WARCO patch overlapping
%  n_row         - (normalized) number of rows for the imgs in the db.
%  n_col         - (normalized) number of colums for the imgs in the db.
%  scale         - scale factor
%  d             - number of features used to build a covariance matrix
%  cv_idx        - cross-validation indexes
%
% EXAMPLE
% root            = './database/CAVIARShoppingCenterFull_1'; % dataset path
% train_dir       = [root '/train']; % training path
% test_dir        = [root '/test']; % testing path
% ds_name         = 'CAVIARShoppingCenterFull_1';
% K               = 8;
% d               = 13;
% patch_dim       = 16;
% patch_overlap   = .5;
% n_row           = 50;
% n_col           = 50;
%
% % load orientation info
% name   =  [root '/or_label'];
% load(name);
% n      = length(or_label_train);
% cv_idx = crossvalind('Kfold', n, K);% build the cross-validation indexes
%
% % per scale test
% scale           = .5;% scale factor
% Z_WARCO_regression_caviar_svm_cbh_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
%     patch_overlap,n_row,n_col,scale,d,cv_idx);
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
%%
% compute the actual scaled image size (to avoid rounding errors)
img_model = imResample(zeros(n_row,n_col),scale);
[n_row,n_col] =  size(img_model);
patch_dim = round(patch_dim*scale);

% image Sym. DooG filtres computation
FB        = FbMake(2,6,0);
offset    = 0;
crop_cost = 0;
if scale ~= 1
    FB         = FbCrop(FB,round(1/scale));
end

%load regression labels
name              =  [root '/or_label'];
load(name);

% Id matrix (used for the log projection
Id           = eye(d);


common          = [ds_name '_r' num2str(n_row) '_c' num2str(n_col)...
    '_d' num2str(d) '_po' num2str(patch_overlap*100)...
    '_pd' num2str(patch_dim) '_fc' num2str(crop_cost)...
    '_po' num2str(scale*100) '_reg' ];
data_dir        = ['./dataset/' common]; % working data directory (IT HAS TO HAVE SPACE to save the II representations)
common          = [common '_libsvm'  '_reg'  '_K' num2str(K) '_cbh' '_deterministic'];
experiment      = ['test/' date '/' common];% experiment output directory
disp(experiment)
mkdir(experiment);
mkdir(data_dir);


%% patches extraction
n_scale = length(scale);
patches = X_patch(zeros(n_row,n_col),scale,patch_dim,patch_overlap);




%% training set building
if(exist([data_dir '/train_covariance_1.mat'],'file') == 0)
    disp(' create train sets...')
    X_build_trainset_caviar_reg_svm(train_dir,data_dir,or_label_train,patches,Id,...
        d,FB,offset,scale,n_scale);
end

%% WARCO computation
if(exist([experiment '/model.mat'],'file') > 0)
    load([experiment '/model']);
else
    disp('model building...')
    F_arr         = struct([]);
    training_set = [];
    U            = [];
    n_wins = 0;
    %for l = 1:n_scale % multi-scale model
    %load data
    name              =  [data_dir '/' 'train_covariance_' num2str(1)];
    load(name);
    n_train           =  length(training_set);
    
    %output structures
    F              = cell(n_wins,1); % regressors
    train_kernel   = cell(n_wins,1); % kernels' elements (for testing purposes)
    kernel_reg     = cell(n_wins,1); % regularization terms
    ts_data        = zeros(d,d,n_train);

    %matlabpool(4)
    for t = 1:n_wins
        disp(['patch: ' num2str(t) '/' num2str(n_wins)])
        n = 1;
        for i = 1:n_train
            ts_data(:,:,n)  = training_set{i}(:,:,t);
            label_data(n) = or_label_train(n).gaze;
            n               = n + 1;
        end
        [train_kernel{t},F{t},kernel_reg{t},~] = ...
            Y_MSVR_Train_deterministic(ts_data, sin(deg2rad(label_data)),0,K,cv_idx);
        
    end
    %matlabpool close
    
    % save
    F_arr(1).U = U;
    F_arr(1).F = F;
    F_arr(1).kernel = train_kernel;
    F_arr(1).reg = kernel_reg;
    F_arr(1).n_wins = n_wins;
    %end
    name              =   [experiment '/' 'model'];
    save(name,'F_arr','-v7.3');
end

%% testing set building
disp('regression...');
U             = F_arr(1).U;
[testing_set] = X_build_testset_caviar_reg_svm(test_dir,...
    data_dir,or_label_test,patches,U,Id,d,FB,offset,scale);

% general parameters
n_wins       = F_arr(1).n_wins;
F            = F_arr(1).F;
train_kernel = F_arr(1).kernel;
kernel_reg   = F_arr(1).reg;
N            = length(testing_set);
GT           = zeros(N,1);
ts_data2              = zeros(d,d,N);
output_orientation         = zeros(N,n_wins);

% orientations normalization
for i = 1:N,GT(i) =  sin(deg2rad(or_label_test(i).gaze));end 

for t = 1:n_wins% for each patch of WARCO
    % extraxting testing data
    for i = 1:N
        ts_data2(:,:,i)  = testing_set{i}(:,:,t);
    end
    
    % compute the dual representation 
    M = size(train_kernel{t},3);
    test = zeros(N,M+1);
    ticId     = ticStatus('test',.2,1);
    for i = 1:N
        %test
        D = zeros(1,M);
        A = ts_data2(:,:,i);
        for z = 1:M
            B = train_kernel{t}(:,:,z);
            %CBH distance
            E  =  trace((B-A)^2);
            D(z) = sqrt(E - 1/12*(trace((A*B)^2) - ...
                trace(A^2*B^2)));
        end
        D = exp(-(1/kernel_reg{t})*D);
        test(i,:) = [1 D];
        tocStatus( ticId, i/N);
    end
    
    % prediction
    [y_out, ~, ~] = svmpredict(GT, sparse(test), F{t});
    output_orientation(:,t)  = y_out;
    
end
% store results
RM           =  median(output_orientation,2);

% evaluation
err_pan(1)  = rad2deg(asin((mean(abs(GT(:,1)-RM(:,1))))));
err_pan(2)  = rad2deg(asin(std(abs(GT(:,1)-RM(:,1)))));
err_pan(3)  = rad2deg(asin(median(abs(GT(:,1)-RM(:,1)))));
disp('pan error:')
disp(err_pan)
save([experiment '/ ' common ],'err_pan','GT','RM')
end




