function [test_set,class_num] = X_build_testset_class(data_dir,store_dir,patches,U,Id,d,FB,offset,scale)
% Build the testing set according to the patch-based model WARCO.
%
% USAGE
%  [test_set,class_num] = X_build_testset_class(data_dir,store_dir,patches,U,Id,d,FB,offset,scale)
%
% INPUTS
%  data_dir      - testing data path
%  store_dir     - working data directory (IT REQUIRES SPACE to save the II representations)
%  patches       - structure containing {
%                      wins  - array of patches' coordinates [r c sizeOfPatch(r) sizeOfPatch(w)]
%                      l_row - number of WARCO's rows  
%                      l_col - number of WARCO's colums
%                      scale - array of scale factors
%                   }
%  U             - normalization matrix (used to normalize covariance matrices)
%  Id            - identity matrix
%  d             - number of features used to build a covariance matrix
%  FB            - bank of filters
%  offset        - FB's offset (pixels) 
%  scale         - scale factor
%
% OUTPUT
% testing_set    - testing set
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
%% general settings
wins                    =   patches.wins; % array of patches' coordinates [r c sizeOfPatch(r) sizeOfPatch(w)]
n_wins                  =   size(wins,1); 
coord                   =   zeros(4,1);
data_dir_categories     =   dir(strcat(data_dir,'/Test*'));
J                       =   size(data_dir_categories,1); % class number
test_set                =   cell(J,1); % array of covariances
class_num               =   zeros(J,1); % classes' number
C_single                =   zeros(d,d,n_wins);
ticId = ticStatus('load data',.2,1);
%% main
for j = 1:J % for all classes
    % % collect images the j-th class
    fg                  =   [dir([data_dir '/' data_dir_categories(j).name '/*.png']);...
        dir([data_dir '/' data_dir_categories(j).name '/*.bmp']);...
        dir([data_dir '/' data_dir_categories(j).name '/*.jpg']);...
        dir([data_dir '/' data_dir_categories(j).name '/*.pgm']);];
    n_img               =   size(fg,1);
    class_num(j)        =   n_img;
    class    =   cell(n_img,1);
    for i   =   1:n_img % for all images in the j-th class
        % load image
        img            =   imread([data_dir '/' data_dir_categories(j).name ...
            '/' fg(i).name]);
        % image rescale
        img    = imResample(img,scale(1),'bilinear');
        
        %feature extraction
        img_feat        =   Y_BuildFf(img,FB,offset);
        %integal tensor representation computation
        pxy =  cumsum(cumsum(img_feat,2));
        [r,c] = meshgrid(1:d,1:d);
        img_feat_prod = cumsum(cumsum(img_feat(:,:,r).*img_feat(:,:,c)),2);
        Qxy           = reshape(img_feat_prod,size(img_feat,1),size(img_feat,2),d,d);
        
        for t = 1:n_wins % for each patch
            coord(1)   =  wins(t,1);
            coord(2)   =  wins(t,2);
            coord(3)   =  (wins(t,1) + wins(t,3))-1;
            coord(4)   =  (wins(t,2) + wins(t,4))-1;
            
            % window covariance estimation
            C = X_CovCal_d(pxy,Qxy,coord,d);
            if any(isnan(C(:))), error('NaN into the covariance'); end
            
            % check SPD prop.
            [V,D] = eig(C);
            if any(diag(D) < 10^(-4))
                diagD = diag(D);
                diagD(diagD < 10^(-4)) = 10^(-4);
                D = diag(diagD);
                C = V * D * V';
            end
            
            % projection of a globally normalized covariance matrix
            C_single(:,:,t) = X_logp(Id,U*C*U);
        end
        class{i} = C_single;
    end
    % store
    test_set{j} = class;
    tocStatus( ticId, j/J );
end
%% store
name              =   [store_dir '/' 'test_covariance'];
save(name,'test_set','class_num');
end
