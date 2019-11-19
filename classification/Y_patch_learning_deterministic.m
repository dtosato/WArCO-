function  [model,train_kernel,kernel_reg,w] = Y_patch_learning_deterministic(training_set,class_num_train,K,J,t,d,cv_idx)
% Learn a patch model of WARCO with the Frobenius distance.
%
% USAGE
%  [model,train_kernel,kernel_reg,w] = Y_patch_learning_deterministic(training_set,class_num_train,K,J,t,d,cv_idx)
%
% INPUTS
%  training_set      - (single patch) training set
%  class_num_train   - array of classes #
%  K                 - (K)cross-validation parameter
%  J                 - number of classes
%  t                 - patch index
%  d                 - number of features used to build a covariance matrix
%  cv_idx            - cross-validation indexes
%
% OUTPUT
% model              - model learned by the kernel SVM
% train_kernel       - kernels' elements (for testing purposes)
% kernel_reg         - regularization term
% w                  - patch weight
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.

% general parameters
ts_data        = zeros(d,d,sum(class_num_train));
label_data     = zeros(sum(class_num_train),1);

% labelling training data
n = 1;
for j = 1:J
    for i = 1:class_num_train(j)
        label_data(n) =  j;
        n             = n + 1;
    end
end

% organizing training data
n = 1;
for j = 1:J
    for i = 1:class_num_train(j)
        ts_data(:,:,n)  = training_set{j}{i}(:,:,t);
        n             = n + 1;
    end
end

%display
%figure(1); bar(label_data);
%figure(2); imagesc(reshape(ts_data,sum(class_num_train),d*d));

% learning Kernal SVM 
[train_kernel,model,kernel_reg,~, w] =...
    Y_ESVM_Train_deterministic(ts_data,label_data,100,K,J,class_num_train,cv_idx);

end