function  [log_posterior] = Y_patch_testing(log_posterior,testing_set,kernel_reg,train_kernel,F,w,N,j,J,t,d)
% Test a patch model of WARCO with the Frobenius distance.
%
% USAGE
%  [log_posterior] = Y_patch_testing(log_posterior,testing_set,kernel_reg,train_kernel,F,w,N,j,J,t,d)
%
% INPUTS
%  log_posterior     - input log-posterior
%  testing_set       - (single patch) testing set
%  kernel_reg        - regularization terms
%  train_kernel      - kernels' elements (for testing purposes)
%  F                 - model learned by the kernel SVM
%  w                 - patch weight
%  J                 - number of classes
%  t                 - patch index
%  d                 - number of features used to build a covariance matrix
%
% OUTPUT
%  log_posterior     - output log-posterior
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.

% labelling testing data
ts_data       = zeros(d,d,N);
label         = zeros(N,1); % array of labels
for i = 1:N
    ts_data(:,:,i)  = testing_set{i}(:,:,t);
    label(i)        =  j;
end

% compute the dual representation
M = size(train_kernel,3);
test = zeros(N,M+1);
ticId     = ticStatus('test',.2,1);
for i = 1:N
    D = zeros(1,M);
    A = ts_data(:,:,i);
    for z = 1:M
        B = train_kernel(:,:,z);
        % Frobenius distance
        D(z)  =  sqrt(trace((B-A)^2));
    end
    D = exp(-(1/kernel_reg)*D);
    test(i,:) = [1 D];
    tocStatus( ticId, i/N);
end

% prediction
[~, ~, vote] = liblinear_predict(label, sparse(test), F);
vote = exp(vote); % fit the exponential model to get a probability distr.
vote = vote./repmat(sum(vote,2),1,J);
vote(isnan(vote)) = 0;
vote(isinf(vote)) = 0;
vote(vote < 0.001)  = 0.001;
vote(vote > 0.999)  = 0.999;

% log-posterior computing
log_posterior(1:N,:)  = log_posterior(1:N,:) + log(vote) + log(repmat(w(t,:),N,1));
end