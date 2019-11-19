function [train_out,F,R,error] = Y_ESVR_Train_deterministic(train,y,c1_rate,K,cv_idx)
% Learn a kernel SVR in the space of SPD matrices using the Frobenius distance.
%
% USAGE
%  [train_out,F,R,error] = Y_ESVR_Train_deterministic(train,y,c1_rate,K,cv_idx)
% INPUTS
%  training_set      - (single patch) training set
%  y                 - array of training labels
%  c1_rate           - target error rate
%  K                 - (K)cross-validation parameter
%  cv_idx            - cross-validation indexes
%
% OUTPUT
% F                  - model learned by the kernel SVM
% train_out          - kernels' elements (for testing purposes)
% R                  - regularization term
% error              - overall error 
%
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
%% general settings 
error = inf;
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
        if error > c1_rate
            svm_cmd_str = ['-s ' num2str(3) ' -c ' num2str(C(c)) ' -t ' ' -q']; 
            F_tmp = svmtrain(y_tmp,train_data,svm_cmd_str);
            [cv_out_tmp, ~, ~] = svmpredict(y_tmp,sparse(train_data), F_tmp);
            e = mean(abs(cv_out_tmp-y_tmp));
            if e <= error
                error = e;
                F = F_tmp;
                R = T;
                train_out = train(:,:,idx == k);
            end
        end
    end  
end
end
