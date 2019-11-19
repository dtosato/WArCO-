function [train_out,F,R,accuracy,w] = Y_ESVM_Train_deterministic(train,y,c1_rate,K,J,class_num_train,cv_idx)
% Learn a multiclass kernel SVM in the space of SPD matrices using the Frobenius distance.
%
% USAGE
%  [train_out,F,R,accuracy,w] = Y_ESVM_Train_deterministic(train,y,c1_rate,K,J,class_num_train,cv_idx)
%
% INPUTS
%  training_set      - (single patch) training set
%  y                 - array of training labels
%  c1_rate           - target classification accuracy
%  K                 - (K)cross-validation parameter
%  J                 - number of classes
%  class_num_train   - array of classes #
%  cv_idx            - cross-validation indexes
%
% OUTPUT
% F                  - model learned by the kernel SVM
% train_out          - kernels' elements (for testing purposes)
% R                  - regularization term
% accuracy           - overall classification accuracy
% w                  - patch weight
%
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.

% Copyright @ Diego Tosato, 2011/01.
% Version @ 1.0.
%% general settings 
accuracy = 0;
R = [];
idx = cv_idx;
for k = 1:K
    %% distance matrix building
    y_tmp = y(idx == k);
    train_tmp = train(:,:,idx == k);
    M = length(y_tmp);
    D  = zeros(M);
    ticId     = ticStatus('kernel',.2,1);
    for i = 1:M
        A = train_tmp(:,:,i);
        for j = i:M
            if i ~= j
                B = train_tmp(:,:,j);
                %Frobenius distance
                D(i,j)  =  sqrt(trace((B-A)^2));
                D(j,i) = D(i,j);
            end
        end
        tocStatus( ticId, i/M);
    end
    %% regularization and trasforming the distance matrix in a kernel (symilarity matrix)
    T = mean(D(:));
    D = exp(-(1/T)*D);
    % format the current training set
    train_data = [ones(size(D,2),1) D];
    
    %% grid-search
    C = -3:1:0; C = 2.^C;
    for c = 1:length(C)
        if accuracy < c1_rate
            svm_cmd_str = ['-s ' num2str(1) ' -c ' num2str(C(c)) ' -q']; 
            F_tmp = liblinear_train(y_tmp,sparse(train_data),svm_cmd_str);
            [cv_out_tmp, ~, ~] = liblinear_predict(y_tmp,sparse(train_data), F_tmp);
            a =  (sum((y_tmp == cv_out_tmp))/numel(y_tmp))*100;
            if a > accuracy
                accuracy = a;
                F = F_tmp;
                cv_out = cv_out_tmp;
                w = diag(confMatrix( y_tmp, cv_out, J))./class_num_train;
                R = T;
                train_out = train(:,:,idx == k);
            end
        end
    end
end
end
