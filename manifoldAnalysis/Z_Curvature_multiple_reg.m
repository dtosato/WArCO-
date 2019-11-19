%   Riemannian Distance Approximation Analysis.
%   Copyright 2010 Diego Tosato

function Z_Curvature_multiple_reg(train_dir,db_name,n_samples,n_row,n_col,patch_dim,patch_overlap,scale,multiple)


% patches
img_model = imResample(zeros(n_row,n_col),scale);
[n_row,n_col] =  size(img_model);
patch_dim = round(patch_dim*scale);
patches = X_patch(zeros(n_row,n_col),scale,patch_dim,patch_overlap);

% filtres
FB        = FbMake(2,6,0);%(2,5,0);
offset    = 0;
if scale ~= 1
    FB         = FbCrop(FB,round(1/scale));
end


store_dir = ['./test/' date '/' db_name '_r' num2str(n_row) '_c' num2str(n_col)...
    '_po' num2str(patch_overlap*100) '_pd' num2str(patch_dim) '_s' num2str(scale*100) ...
    '_nsampl' num2str(n_samples) '_m' num2str(multiple) '_curvatureD' ];
mkdir(store_dir)
disp('load covariances')

% count imgs
J                   =   1;
num                 =   zeros(J,1);
idx                 =   zeros(n_samples,J);
wins                =   patches(1).wins;
n_wins              =   size(wins,1);
d                   = 13;
Id                  = eye(d);

for j = 1:J
    img_dir =   [dir(strcat(train_dir,'/*.jpg'));
        dir(strcat(train_dir,'/*.bmp'));
        dir(strcat(train_dir,'/*.png'));];
    num(j)    =   n_samples;
    idx(:,j)  =   randSample( 1:size(img_dir,1), n_samples);
    X         =   zeros(d,d,n_wins*num(j));   
end
t = 1;
for j = 1:J
    ticId     = ticStatus('load',.2,1);
    img_dir =   [dir(strcat(train_dir,'/*.jpg'));
        dir(strcat(train_dir,'/*.bmp'));
        dir(strcat(train_dir,'/*.png'));];
    for i=1:n_samples
        % load images
        img = imread(strcat(train_dir,'/',img_dir(idx(i,j)).name));
        
        
        img    = imResample(img,scale,'bilinear');
        si_img = size(img);
        
        % compute the features
        img_feat  =   Y_BuildFf(img,FB,offset);
 
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
        
        for p = 1:n_wins
            wins_curr = wins(p,:);
            r1 = wins_curr(1);
            c1 = wins_curr(2);
            r2 = wins_curr(1) + wins_curr(3) -1;
            c2 = wins_curr(2) + wins_curr(4) -1;
            C = X_CovCal_d(pxy,Qxy,[r1 c1 r2 c2],d);
            C = C-diag(diag(C))+eye(d)*multiple;
            X(:,:,t) = C;
            %X(:,:,t) = real(X_logp(Id,C));
            t = t + 1;
        end
        tocStatus( ticId, i/num(j));
    end
end

disp('curvature analysis')
N    = size(X,3);
%gM   = zeros(N^2,1);
%gC   = zeros(N^2,1);
%gC2  = zeros(N^2,1);
%e    = zeros(N^2,1);
sca  = zeros(N^2,1);

n  = 1;
ticId = ticStatus(' ',.2,1);
for i = 1:N
    for j = 1:N%par
        if j ~= i
            
            % Projection
            
            X_curr = X(:,:,i);
            Y_curr = X(:,:,j);
            
            % SCA
            K = 2*(trace((X_curr*Y_curr)^2) - trace(X_curr^2*Y_curr^2))/...
                (trace(X_curr^2)*trace(Y_curr^2) - trace((X_curr*Y_curr))^2+eps);
            sca(n) =  K;
            
            % GEODETC - LOB
            %k = sqrt(-K);
            %a      = sqrt(trace(X_curr^2));%norm(X_curr,'fro');
            %b      = sqrt(trace(Y_curr^2));
            %prd    = trace((X_curr*Y_curr));
            %gL(n) =  k^-1*acosh(cosh(a*k)*cosh(b*k)-sinh(a*k)*sinh(b*k)*...
            %    cos(prd/(a*b)));
            % EUCLIDEAN
            %e(n)  =  sqrt(trace((Y_curr-X_curr)^2));
            % GEODETIC -  CAMPBELL-HAUSDORFF
            %gC(n) =  sqrt(e(n)^2 - 1/12*(trace((X_curr*Y_curr)^2) - ...
            %    trace(X_curr^2*Y_curr^2)));
            % GEODETIC - CAMPBELL-HAUSDORFF 2
            %R0 = X_curr*Y_curr - Y_curr*X_curr;
            %R1 = R0*X_curr - X_curr*R0;
            %R2 = R0*Y_curr - Y_curr*R0;
            %gC2(n) =  sqrt(e(n)^2 - 1/12*(trace((X_curr*Y_curr)^2) - ...
            %    trace(X_curr^2*Y_curr^2)) + (sqrt(trace(R1^2))^2+...
            %    sqrt(trace(R2^2))^2 + 4*sqrt(trace((R1-R2)^2))^2)/1152);
            % GEODETC - MEER
            %X_curr = X(:,:,i)^(-0.5);
            %Y_curr = X(:,:,j);
            %gM(n) = sqrt(real(trace(X_logp(Id,X_curr*Y_curr*X_curr)^2)));
            
            n = n + 1;
        end
    end
    tocStatus( ticId, i/N );
end
%gM   = gM(1:n-1);
%gC2  = gC2(1:n-1);
%gC   = gC(1:n-1);
%e    = e(1:n-1);
sca  = sca(1:n-1);
%mean_gM  = (mean(gM));
%mean_gC  = (mean(gC));
%mean_gC2 = (mean(gC2));
%mean_e   = (mean(e));
mean_sca   = (mean(sca));
%median_gM  = (median(gM));
%median_gC  = (median(gC));
%median_gC2 = (median(gC2));
%median_e   = (median(e));
median_sca   = (median(sca));
%std_gM  = (std(gM));
%std_gC  = (std(gC));
%std_gC2 = (std(gC2));
%std_e   = (std(e));
std_sca   = (std(sca));

%% save
name      =   [store_dir '/' 'SCA'];
save(name,'sca','mean_sca','median_sca','std_sca');

%matlabpool close
end
