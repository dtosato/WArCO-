function Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
% Curvature analysis of a SPD dataset. 
%
% USAGE
%  Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
%
% INPUTS
%  train_dir     - array of training paths
%  ds_name       - dataset (or db) name
%  patch_dim     - WARCO (single) patch size
%  patch_overlap - WARCO patch overlapping
%  n_row         - (normalized) number of rows for the imgs in the db
%  n_col         - (normalized) number of colums for the imgs in the db
%  scale         - scale factor
%  sample        - sampling step

%
% EXAMPLE
% db_name = 'VIPER4PoseHuman3';
% train_dir = ['../database/' db_name '/train'];
% n_row           = 128; %(normalized) number of rows for the imgs in the db. 
% n_col           = 48; % (normalized) number of colums for the imgs in the db.
% patch_overlap   = .5; % WARCO patch overlapping 
% patch_dim       = 16; % WARCO (single) patch size 
% sample          = 8; % (img) sampling step
% 
% % per scale test
% scale           = 1;% scale factor
% Z_Curvature_paired(train_dir,db_name,n_row,n_col,patch_dim,patch_overlap,scale,sample)
%
% REFERENCES
% [1] D. Tosato, M. Spera, M. Cristani, V. Murino, Characterizing humans on Riemannian manifolds,
% IEEE  Trans. PAMI, Preprint 2011.
%
% Diego Tosato      Version 1.00
% Copyright 2012 Diego Tosato.  [diego.tosato-at-univr.it]
% Please email me if you have questions.
%%
% compute the actual scaled image size (to avoid rounding errors)
img_model = imResample(zeros(n_row,n_col),scale);
[n_row,n_col] =  size(img_model);
patch_dim = round(patch_dim*scale);
patches = X_patch(zeros(n_row,n_col),scale,patch_dim,patch_overlap);

% image Sym. DooG filtres computation
FB        = FbMake(2,6,0);
offset    = 0;
if scale ~= 1
    FB         = FbCrop(FB,round(1/scale)); % filters are scale dependent
end

% output settings
store_dir = ['./test/' date '/' db_name '_r' num2str(n_row) '_c' num2str(n_col)...
    '_po' num2str(patch_overlap*100) '_pd' num2str(patch_dim) '_s' num2str(scale*100) ...
     '_smp' num2str(sample)  '_curvaturePaired' ];
mkdir(store_dir)
%% load all the covariance matrices
disp('load covariances')

%% count imgs
class_dir           = dir(strcat(train_dir,'/Data_*'));
J                   = length(class_dir); % number of classes
num                 = zeros(J,1);
wins                = patches(1).wins; % patches 
n_wins              = size(wins,1);
d                   = 13; % number of feature
Id                  = eye(d); % identity matrix
X                   = cell(J,1); % store the covariances
for j = 1:J
    img_dir =   [dir(strcat(train_dir,'/',class_dir(j).name,'/*.jpg'));
        dir(strcat(train_dir,'/',class_dir(j).name,'/*.bmp'));
        dir(strcat(train_dir,'/',class_dir(j).name,'/*.png'));];
    img_dir   = img_dir(1:sample:end);
    num(j)    =    size(img_dir,1);
    X{j}      =   zeros(d,d,n_wins,num(j));
end
for j = 1:J
    disp(['-> class ' class_dir(j).name])
    ticId     = ticStatus('load',.2,1);
    img_dir =   [dir(strcat(train_dir,'/',class_dir(j).name,'/*.jpg'));
        dir(strcat(train_dir,'/',class_dir(j).name,'/*.bmp'));
        dir(strcat(train_dir,'/',class_dir(j).name,'/*.png'));];
    for i=1:num(j) 
        % load img
        img = imread(strcat(train_dir,'/',...
            class_dir(j).name,'/',img_dir(i).name));
        % image rescale
        img    = imResample(img,scale,'bilinear');
        si_img = size(img);
        %feature extraction
        img_feat  =   Y_BuildFf(img,FB,offset);
        %integal tensor representation computation
        pxy           = zeros(si_img(1),si_img(2),d);
        Qxy           = zeros(si_img(1),si_img(2),d,d);
        for ii = 1:d
            pxy(:,:,ii) =  squeeze(cumsum(cumsum(img_feat(:,:,ii),2)));
            for jj = ii:d
                Qxy(:,:,ii,jj)    =  squeeze(cumsum(cumsum(img_feat...
                    (:,:,ii).*img_feat(:,:,jj)),2));
                Qxy(:,:,jj,ii)    =   Qxy(:,:,ii,jj) ;
            end
        end
        
        for p = 1:n_wins % for each patch
            wins_curr = wins(p,:);
            r1 = wins_curr(1);
            c1 = wins_curr(2);
            r2 = wins_curr(1) + wins_curr(3) -1;
            c2 = wins_curr(2) + wins_curr(4) -1;
            
            % window covariance estimation
            C = X_CovCal_d(pxy,Qxy,[r1 c1 r2 c2],d);      
            if any(isnan(C(:))), error('NaN into the covariance'); end
            
            % check SPD prop.
            [V,D] = eig(C);
            if any(diag(D) < 10^(-4))
                diagD = diag(D);
                diagD(diagD < 10^(-4)) = 10^(-4);
                D = diag(diagD);
                C = V * D * V';
            end
            % store
            X{j}(:,:,p,i) = C;
        end
        tocStatus( ticId, i/num(j));
    end
end

%% curvature analysis
disp('curvature analysis')
N = sum(num);
sca  = (N*(N+1)/2)*n_wins; % sectional curvature
e    = (N*(N+1)/2)*n_wins; % Frobenius distaice
gC   = (N*(N+1)/2)*n_wins; % CBH distance
gM   = (N*(N+1)/2)*n_wins; % Closed form Reimannian distance

ticId = ticStatus(' ',.2,1);
n = 1;
for p = 1:n_wins % for each patch
    ts_data        = zeros(d,d,sum(num));
    m = 1;
    for j = 1:J % for each class
        for i = 1:num(j) % for each image
            ts_data(:,:,m)  = X{j}(:,:,p,i);
            m             = m + 1;
        end
    end
    for i = 1:N
        for j = 1:N%par
            
            X_id = real(X_logp(Id,ts_data(:,:,i)));
            Y_id = real(X_logp(Id,ts_data(:,:,j)));
            
            % SCA
            K = 2*(trace((X_id*Y_id)^2) - trace(X_id^2*Y_id^2))/...
                (trace(X_id^2)*trace(Y_id^2) - trace((X_id*Y_id))^2+eps);
            sca(n) =  K;
            
            % EUCLIDEAN
            e(n)  =  sqrt(trace((Y_id-X_id)^2));
            
            % GEODETIC -  CAMPBELL-HAUSDORFF            
            gC(n) =  sqrt(e(n)^2 - 1/12*(trace((X_id*Y_id)^2) - ...
                trace(X_id^2*Y_id^2)));
            
            % GEODETC - RIEMANNIAN
            X_M = ts_data(:,:,i)^(-0.5);
            Y_M = ts_data(:,:,j);
            gM(n) = real(sqrt(real(trace(X_logp(Id,X_M*Y_M*X_M)^2))));
            
            n = n + 1;  
        end
    end
    tocStatus( ticId, p/n_wins );
end

%% output statistics
mean_gM  = (mean(gM));
mean_gC  = (mean(gC));
mean_e   = (mean(e));
mean_sca   = (mean(sca));
median_gM  = (median(gM));
median_gC  = (median(gC));
median_e   = (median(e));
median_sca   = (median(sca));
std_gM  = (std(gM));
std_gC  = (std(gC));
std_e   = (std(e));
std_sca   = (std(sca));
err_e   = (mean(abs(e - gM)));
err_gC   = (mean(abs(gC - gM)));

%% save
try
name      =   [store_dir '/' 'MA'];
save(name,'mean_sca','median_sca','std_sca','mean_e','median_e','std_e',...
    'mean_gC','median_gC','std_gC','mean_gM','median_gM','std_gM',...
    'err_e', 'err_gC','e','gC','gM');

catch err 
    save(name);
    
end

%matlabpool close
end
