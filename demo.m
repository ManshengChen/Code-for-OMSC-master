clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = {'Caltech101-20'};

resPath = './res-lmd0/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

for dsi = 1
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dataName));
    k = length(unique(Y));
    
    matpath = strcat(resPath,dataName);
    txtpath = strcat(resPath,strcat(dataName,'.txt'));
    if (~exist(matpath,'file'))
        mkdir(matpath);
        addpath(genpath(matpath));
    end
    dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
    
    %% para setting
    anchor = [k 2*k 3*k];
    d = k;
    lambda = [1 0.1 0.01 0.001];
    
    %%
    for ichor = 1:length(anchor)
        for id = 1:length(d)
            for j = 1:length(lambda)
            tic;
            [A,W,Z,G,F,iter,obj,alpha] = algo_OMSC(X,Y,lambda(j),d(id),anchor(ichor)); % X,Y,lambda,d,numanchor
            [~,idx]=max(F);
            res = Clustering8Measure(Y,idx); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
            timer(ichor,id)  = toc;
            fprintf('Lambda:%f \t Anchor:%d \t Dimension:%d\t Res:%12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[lambda anchor(ichor) d(id) res(1) res(2) res(3) res(4) timer(ichor,id)]);
            
            resall{ichor,id} = res;
            objall{ichor,id} = obj;
            
            dlmwrite(txtpath, [lambda(j) anchor(ichor) d(id) res timer(ichor,id)],'-append','delimiter','\t','newline','pc');
            matname = ['_Anch_',num2str(anchor(ichor)),'_Dim_',num2str(d(id)),'.mat'];

            save([matpath,'/',matname],'Z','A','W','alpha');
            end
        end
    end
    clear resall objall X Y k
end


