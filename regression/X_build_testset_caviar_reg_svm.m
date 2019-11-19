function [testing_set] = X_build_testset_caviar_reg_svm(data_dir,~,or_label_test,patches,U,Id,d,FB,offset,scale)
% Build the regression testing set according to the patch-based model WARCO for the CAVIAR data.O.
%
% USAGE
%  [testing_set] =  X_build_testset_caviar_reg_svm(data_dir,store_dir,patches,Id,d,FB,offset,scale)
%
% INPUTS
%  data_dir      - training data path
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
wins                    =   patches(1).wins; % array of patches' coordinates [r c sizeOfPatch(r) sizeOfPatch(w)]
n_wins                  =   size(wins,1);
coord                   =   zeros(4,1);
C_single                =   zeros(d,d,n_wins); % array of covariances
ticId = ticStatus('test data',.2,1);
n_img               =   length(or_label_test);
testing_set                =   cell(n_img,1);
%% main
for i   =   1:n_img % for all images
    % load image
    img            =   imread([data_dir  ...
        '/' or_label_test(i).name '.jpg']);
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
        %disp(['coord: ' num2str(coord)])
        
        % window covariance estimation
        C = X_CovCal_d(pxy,Qxy,coord,d);
        if any(isnan(C(:))), error('NaN into the covariance'); end


       % check SPD prop.
        [V,D] = eig(C);
        if any(diag(D) < 10^(-4))
            %disp('* eig shift C(:,:,ii,t,j)')
            diagD = diag(D);
            diagD(diagD < 10^(-4)) = 10^(-4);
            D = diag(diagD);
            C = V * D * V';
        end
        
        % normalization
        C = U*C*U;
        
        % mapping on Id
        C_single(:,:,t) = real(X_logp(Id,C));
    end
    % store
    testing_set{i} = C_single;
    tocStatus( ticId, i/n_img );
end
%% store
name              =   [store_dir '/' 'test_covariance'];
save(name,'testing_set','-v7.3');
end
