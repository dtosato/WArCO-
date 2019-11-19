function  X_build_trainset_class(data_dir,store_dir,patches,Id,d,FB,offset,scale,~)
% Build the training set according to the patch-based model WARCO.
%
% USAGE
%  X_build_trainset_class(data_dir,store_dir,patches,Id,d,FB,offset,scale,~)
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
%  d             - number of features used to build a covariance matrix
%  FB            - bank of filters
%  offset        - FB's offset (pixels) 
%  scale         - scale factor
%
%
% EXAMPLE
% data_dir       = './database/QML5PoseHeads/train'; % training path
% store_dir       = './tmp';makedir(store_dir); % training path            
% d               = 13; % number of features used to build a covariance matrix
% Id              = eye(d);
% patch_dim       = 16; % WARCO (single) patch size 
% patch_overlap   = .5; % WARCO patch overlapping 
% n_row           = 50; % (normalized) number of rows for the imgs in the db. 
% n_col           = 50; % (normalized) number of colums for the imgs in the db.
% scale           = .5;% scale factor
% patches         = X_patch(zeros(n_row,n_col),scale,patch_dim,patch_overlap);
% FB              = FbMake(2,6,0);
% offset          = 0;
% if scale ~= 1
%     FB          = FbCrop(FB,round(1/scale)); % filters are scale dependent
% end
% X_build_trainset_class(train_dir,data_dir,patches,Id,d,FB,offset,scale,1);
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
feature_max             =   zeros(d,1); % meatures' max value (for normalization puroposes)
C_single                =   zeros(d,d,n_wins);% array of covariances
coord                   =   zeros(4,1); % coordiantes of a window
data_dir_categories     =   dir(strcat(data_dir,'/Data*'));
J                       =   size(data_dir_categories,1); % classes' number
class_num_train         =   zeros(J,1);
training_set            =   cell(J,1);
ticId = ticStatus('load data',.2,1);

%% main
for j = 1:J % for all classes
    % % collect images the j-th class
    fg                  =   [dir([data_dir '/' data_dir_categories(j).name '/*.png']);...
        dir([data_dir '/' data_dir_categories(j).name '/*.bmp']);...
        dir([data_dir '/' data_dir_categories(j).name '/*.jpg']);...
        dir([data_dir '/' data_dir_categories(j).name '/*.pgm']);];
    n_img               =   size(fg,1);
    class_num_train(j)        =   n_img;
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
            
            % collecting the max val for each feature
            var_curr = diag(C);
            idx = var_curr > feature_max;
            if any(idx),feature_max(idx) = var_curr(idx); end
            % store
            C_single(:,:,t) = C;
        end
        % store
        class{i} = C_single;
    end
    training_set{j} = class;
    tocStatus( ticId, j/J );
end

% global normalization factor
U = diag(feature_max.^(-1/2));

%% projection
ticId = ticStatus('project data',.2,1);
for j = 1:J
    for i = 1:length(training_set{j})
        for t = 1:n_wins
            X = training_set{j}{i}(:,:,t);
			% projection of a globally normalized covariance matrix
            training_set{j}{i}(:,:,t) = X_logp(Id,U*X*U);
        end
    end
    tocStatus( ticId, j/J );
end

%% store
name              =   [store_dir '/' 'train_covariance_1'];
save(name,'training_set','U','class_num_train','n_wins','-v7.3');
end