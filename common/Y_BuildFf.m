function [F] = Y_BuildFf(data,FB,offset)
% Compute a set of image features to build WARCO.
%
% USAGE
%  [F] = Y_BuildFf(data,FB,offset)
%
% INPUTS
%  data      - input WxH image
%  FB        - bank of filters
%  offset    - FB's offset (pixels) 
%
% OUTPUT
% F          - multidimensional array (WxHxd) of features
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.


% filters
data_lum          = rgb2ycbcr(data);
N                 = size(FB,3);
if ~offset
    A                 = double(FbApply2d(data_lum(:,:,1), FB(:,:,1:N), 'same', 0));
else
    A                 = double(FbApply2d(data_lum(:,:,1), FB(:,:,1:N), 'valid', 0));
end
F(:,:,1:N)        = double(A);


% color
if ~offset
    data_lum           =   double(data_lum);
else
    data_lum            = double(data_lum(offset:end-offset+1,offset:end-offset+1,:));
end
F(:,:,N+1)          =   data_lum(:,:,1); %L
F(:,:,N+2)          =   data_lum(:,:,2); %U
F(:,:,N+3)          =   data_lum(:,:,3); %V

% gradient
[gh,gv]           =   gradient(data_lum(:,:,1));
F(:,:,N+4)         =   sqrt(gh.^2+gv.^2);
F(:,:,N+5)         =   atan2(gh,gv);
end