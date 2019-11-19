function [patches]=X_patch(img,layers,patch_dim_input,patch_overlap)
% Compute the WARCO regular grid of patches.
%
% USAGE
% [patches]=X_patch(img,layers,patch_dim_input,patch_overlap)
%
% INPUTS
%  img             - input image 
%  layers          - number of WARCO's layers
%  patch_dim_input - dataset (or db) name
%  patch_overlap   - overlap among patches
%
% OUTPUT
% patches          - structure containing {
%                      wins  - array of patches' coordinates [r c sizeOfPatch(r) sizeOfPatch(w)]
%                      l_row - number of WARCO's rows  
%                      l_col - number of WARCO's colums
%                      scale - array of scale factors
%                   }
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
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.

% general settings
n_layers  = length(layers);
img_or    = img;
patches    = struct([]);
patch_dim  = zeros(n_layers,1);

for l = 1: n_layers % for each WARCO layer
    % image resizing
    img      = imResample(img_or,layers(l),'bilinear');
    si_im    = size(img);
    x        = si_im(1);
    y        = si_im(2);
    
    % check the img size
    patch_dim(l)   = round(patch_dim_input*layers(l));
    stepPatch      = round(patch_dim(l)*(1 - patch_overlap));
    
    % regular grid computation
    x_arr          = 1:stepPatch:x-patch_dim(l)+1;
    y_arr          = 1:stepPatch:y-patch_dim(l)+1;
    l_row          = length(x_arr);
    l_col          = length(y_arr);
    wins           = zeros(l_row*l_col,4);
    
    % collect all the patches
    n = 1;
    for i = 1:l_row
       for j = 1:l_col 
           wins(n,:) = [x_arr(i) y_arr(j) patch_dim(l) patch_dim(l)];
           n = n + 1;
       end
    end

    % centering the grid
    row_bound = wins(end,1) + wins(end,3) - 1;
    col_bound = wins(end,2) + wins(end,4) - 1;
    if row_bound < x
       diff =  ceil((x - row_bound)/2);
       wins(:,1) = wins(:,1) + diff;
    end
    if col_bound < y
       diff =  ceil((y - col_bound)/2);
       wins(:,2) = wins(:,2) + diff;
    end
    
    % saving
    patches(l).wins    = wins;
    patches(l).nrow    = l_row;
    patches(l).ncol    = l_col;
    patches(l).scale   = layers(l);
end
end