function logXY = X_logp(X,Y) 
% Compute the logarithmic map associated to the Riemaniann metric of SPD matrices.
%
% USAGE
% logXY = X_logp(X,Y) 
%
% INPUTS
%  X      - projection point (SPD matrix)
%  Y      - point to be projected (SPD matrix)
%
% OUTPUT
% logXY - SSPD matrix projected to the tangent space in X
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
exppos = X^(1/2);
expneg = X^(-1/2);
inner = expneg*Y*expneg;
[U,S] = eig(inner);
log_inner = U*diag(log(diag(S)))*U';
logXY = exppos * log_inner * exppos;
end