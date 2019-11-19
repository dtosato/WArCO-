function C = X_CovCal_d(pxy,Qxy,R,d)
% Compute a covariance descriptors using the integral representation as 
% described in [2].
%
% USAGE
% C = Y_CovCal_d(pxy,Qxy,R,d)
%
% INPUTS
%  pxy             - first order tensor of integral images WxHxd
%  Qxy             - second order tensor of integral images WxHxd
%  R               - 4-tuple of the region 
%  d               - matrix dimension
%
% OUTPUT
% C                - covariance descriptors
%
% EXAMPLE
% patch_dim       = 16; % WARCO (single) patch size 
% patch_overlap   = .5; % WARCO patch overlapping 
% n_row           = 50; % (normalized) number of rows for the imgs in the db. 
% n_col           = 50; % (normalized) number of colums for the imgs in the db.
% scale           = .5;% scale factor
% patches         = X_patch(zeros(n_row,n_col),scale,patch_dim,patch_overlap);
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.

% [2]O. Tuzel, F. Porikli, P. Meer, Pedestrian Detection via
%      Classification on Riemannian Manifolds, IEEE Trans. PAMI, 2008.
%
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
if(R(1) == 1 && R(2) == 1)
    S    = R(3)*R(4); % normalization term
    to_transp = reshape(pxy(R(3),R(4),:),d,1);
    C = 1/(S-1).*(reshape(Qxy(R(3),R(4),:,:),d,d) -...
        1/S.*(to_transp*to_transp'));
    
elseif R(1) == 1
    S    = (R(3) - R(1) + 1)*(R(4) - R(2) + 1);% normalization term
    first_fact = reshape(Qxy(R(3),R(4),:,:) - Qxy(R(3),R(2)-1,:,:),d,d);
    to_transp = reshape((pxy(R(3),R(4),:) -  pxy(R(3),R(2)-1,:)),d,1);
    C = (1/(S-1)).*(first_fact-(1/S).*(to_transp*to_transp'));
elseif R(2) == 1
    S    = (R(3) - R(1) + 1)*(R(4) - R(2) + 1);% normalization term
    first_fact = reshape(Qxy(R(3),R(4),:,:) -  Qxy(R(1)-1,R(4),:,:),d,d);
    to_transp = reshape((pxy(R(3),R(4),:) - pxy(R(1)-1,R(4),:)),d,1);
    C = (1/(S-1)).*(first_fact-(1/S).*(to_transp*to_transp'));
else
    S    = (R(3) - R(1) + 1)*(R(4) - R(2) + 1);% normalization term
    first_fact = reshape(Qxy(R(3),R(4),:,:)    +    Qxy(R(1)-1,R(2)-1,:,:)   -  Qxy(R(3),R(2)-1,:,:)  -  Qxy(R(1)-1,R(4),:,:),d,d);
    to_transp = reshape((pxy(R(3),R(4),:)    +    pxy(R(1)-1,R(2)-1,:)     -  pxy(R(1)-1,R(4),:)    -  pxy(R(3),R(2)-1,:)),d,1);
    C = (1/(S-1)).*(first_fact-(1/S).*(to_transp*to_transp'));
end
end

