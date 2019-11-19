function Z_WARCO_regression_idiap_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,patch_overlap,n_row,n_col,scale,d,cv_idx)
% Regressor training and testing procedure exploiting WARCO with the Frobenius distance for the CAVIAR data.
%
% USAGE
%  Z_WARCO_regression_idiap_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,patch_overlap,n_row,n_col,scale,d,cv_idx)
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
% root            = './database/IHDP_10smp_flip_alpha_allSeq_pose_reg'; % dataset path
% train_dir       = [root '/train']; % training path
% test_dir        = [root '/test']; % testing path
% ds_name         = 'IHDP_10smp_flip_alpha_allSeq_pose_reg';
% K               = 8;
% d               = 13;
% patch_dim       = 24;
% patch_overlap   = .5;
% n_row           = 75;
% n_col           = 75;
% %
% % % load orientation info
% name   =  [root '/or_label'];
% load(name);
% n      = length(or_label_train);
%
% % build the cross-validation indexes
% if(exist([root '/cv_'  num2str(K)  '.mat'],'file') == 0)
% cv_idx = crossvalind('Kfold', n, K);
% save([root '/cv_'  num2str(K)  '.mat'],'cv_idx','-v7.3');
% else
% disp('load CV IDXs')
% load([root '/cv_'  num2str(K)  '.mat']);
% end
% % % per scale test
% % scale           = .5;% scale factor
% Z_WARCO_regression_idiap_svm_scale_deterministic(root,train_dir,test_dir,ds_name,K,patch_dim,...
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
common          = [common '_libsvm'  '_reg'  '_K' num2str(K) '_deterministic'];
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
    X_build_trainset_idiap_reg_svm(train_dir,data_dir,or_label_train,patches,Id,...
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
    label_data     = zeros(n_train,1);    
    % structures
    F             = cell(n_wins,3); % regressors
    train_kernel  = cell(n_wins,3); % kernels' elements (for testing purposes)
    kernel_reg    = cell(n_wins,3); % regularization terms
    ts_data       = zeros(d,d,n_train);

    
    
    for t = 1:n_wins
        disp(['patch: ' num2str(t) '/' num2str(n_wins)])
        for a = 1:3
            disp(['angle: ' num2str(a) '/' num2str(3)])
            for i = 1:n_train
                ts_data(:,:,i)  = training_set{i}(:,:,t);
                if a  == 1
                    label_data(i) =  or_label_train(i).pan;
                elseif a == 2
                    label_data(i) =  or_label_train(i).tilt;
                else
                    label_data(i) =  or_label_train(i).roll;
                end
            end
            [train_kernel{t,a},F{t,a},kernel_reg{t,a},~] = ...
                Y_ESVR_Train_deterministic(ts_data,label_data,0,K,cv_idx);
        end
        
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
U            = F_arr(1).U;
[testing_set] = X_build_testset_idiap_reg_svm(test_dir,...
    data_dir,or_label_test,patches,U,Id,d,FB,offset,scale);

% general parameters
n_wins       = F_arr(1).n_wins;
F            = F_arr(1).F;
train_kernel = F_arr(1).kernel;
kernel_reg   = F_arr(1).reg;
N          = length(testing_set);
GT         = zeros(N,3);
RM         = zeros(N,3);
% extract the testing regression labels
for i = 1:N
    for a  = 1:3
        if a  == 1
            GT(i,1) =  or_label_test(i).pan;
        elseif a == 2
            GT(i,2) =  or_label_test(i).tilt;
        else
            GT(i,3) =  or_label_test(i).roll;
        end
        
    end
end
ts_data2   = zeros(d,d,N);
output_orientations         = zeros(N,n_wins,3);
for t = 1:n_wins
    for i = 1:N
        ts_data2(:,:,i)  = testing_set{i}(:,:,t);
    end
    for a = 1:3
        M = size(train_kernel{t,a},3);
        test = zeros(N,M+1);
        ticId     = ticStatus('test',.2,1);
        for i = 1:N
            D = zeros(1,M);
            A = ts_data2(:,:,i);
            for z = 1:M
                B = train_kernel{t,a}(:,:,z);
                % Frobenius distance
                D(z)  =  sqrt(trace((B-A)^2));
            end
            D = exp(-(1/kernel_reg{t,a})*D);
            test(i,:) = [1 D];
            tocStatus( ticId, i/N);
        end
        
        % prediction
        [y_out, ~, ~] = svmpredict(GT(:,a), sparse(test), F{t,a});
        output_orientations(:,t,a)  = y_out;
    end
end
% store results
for a = 1:3
    RM(:,a)           =  median(output_orientations(:,:,a),2);
end

% evaluation
err_pan(1)  = rad2deg(mean(abs(GT(:,1)-RM(:,1))));
err_tilt(1) = rad2deg(mean(abs(GT(:,2)-RM(:,2))));
err_roll(1) = rad2deg(mean(abs(GT(:,3)-RM(:,3))));
err_pan(2)  = rad2deg(std(abs(GT(:,1)-RM(:,1))));
err_tilt(2) = rad2deg(std(abs(GT(:,2)-RM(:,2))));
err_roll(2) = rad2deg(std(abs(GT(:,3)-RM(:,3))));
err_pan(3)  = rad2deg(median(abs(GT(:,1)-RM(:,1))));
err_tilt(3) = rad2deg(median(abs(GT(:,2)-RM(:,2))));
err_roll(3) = rad2deg(median(abs(GT(:,3)-RM(:,3))));
disp('pan error:')
disp(err_pan)
disp('tilt error:')
disp(err_tilt)
disp('roll error:')
disp(roll_tilt)
save([experiment '/ ' common ],'err_pan','err_tilt','err_roll','GT','RM')
end




