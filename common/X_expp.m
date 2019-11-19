function expXY = X_expp(X,Y)
%   Compute the exponential map associated to the Riemaniann metric of SPD matrices.
%
% USAGE
% expXY = X_expp(X,Y) 
%
% INPUTS
%  X      - projection point (SPD matrix)
%  Y      - point to be projected (SSPD matrix)
%
% OUTPUT
% expXY - SPD matrix projected to the tangent space in X
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
exppos = X^(+0.5);
expneg = X^(-0.5);
inner = expneg*Y*expneg;
[U,S] = eig(inner);
exp_inner = U*diag(exp(diag(S)))*U';
expXY = exppos * exp_inner * exppos;
end