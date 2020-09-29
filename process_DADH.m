function result = process_DADH(dataset, param)
XTrain = dataset.IAll(:, :, :, param.indexTrain);
trainLabel = dataset.LAll(param.indexTrain,:);
XRetrieval = dataset.IAll(:,:,:,param.indexRetrieval);
retrievalLabel = dataset.LAll(param.indexRetrieval,:);

XQuery = dataset.IAll(:,:,:,param.indexQuery);
queryLabel = dataset.LAll(param.indexQuery,:);
clear dataset

S = trainLabel * trainLabel' > 0;
s1 = 1;
s0 = s1 * 0.11;
sampleS = S*(s1 + s0) - s0;
    
bit = param.bit;
gamma = param.gamma;
eta = param.eta;
tau = param.tau;

lr_img = param.lr;
lr_txt = param.lr;

num_train = size(XTrain,4);

maxIter = param.maxIter;
F = zeros(bit, num_train);
G = zeros(bit, num_train);
load('./data/vgg_net.mat');
txt_net = net_structure_img(net, bit);
img_net = net_structure_img(net, bit);
B = zeros(bit, num_train);
loss = zeros(1, maxIter);

batch_size = param.batch_size;
for epoch = 1: maxIter
for i = 1:1
    FV = F';
    GV = G';
    V = B';
    for ii = 1: bit
        V_ = V;
        V_(:, ii) = [];
        Q = -2 * bit * sampleS' * (FV+GV) - 2 * gamma * (FV+GV);
        q = Q(:, ii);
        Hf_ = FV;
        hf = Hf_(:, ii);
        Hf_(:, ii) = [];
        Hg_ = GV;
        hg = Hg_(:, ii);
        Hg_(:, ii) = [];
        V(:, ii) = sign(-2 * V_ * Hf_' * hf -2 * V_ * Hg_' * hg - q);
    end
    B = V';
end   
    for ii = 1: ceil(num_train/ batch_size)
        R = randperm(num_train);
        index = R(1: batch_size);
        img = single(XTrain(:,:,:,index));
        img = imresize(img, net.meta.normalization.imageSize(1: 2));
        im_ = img - repmat(net.meta.normalization.averageImage,1,1,1,size(img,4));
        im_ = gpuArray(im_);
        
        res = vl_simplenn(txt_net,im_);
        n_layers = numel(txt_net.layers);
        output = gather(squeeze(res(end).x));
        G(:,index) = tanh(output);
        
        dJdLogloss = 0.5*F*(1 ./ (1+exp(-0.5*F'*G(:,index)))-S(:,index));        
        G1 = G*ones(num_train,1);
        dJdGB = 2*gamma *(G(:,index)-B(:,index))+2*eta*repmat(G1,1,numel(index));
        dgdAS = 2*B*(B'*G(:,index) - bit*S(:,index));
        dJdGb = tau*dJdLogloss + dJdGB + dgdAS;
        dJdGb = dJdGb.*(1 - (G(:,index)).^2);
        
        dJdGb = single(gpuArray(reshape(dJdGb,[1,1,size(dJdGb,1),size(dJdGb,2)])));
        res = vl_simplenn(txt_net,im_,dJdGb);            
        txt_net = update_net2(txt_net,res,lr_txt(epoch),num_train,n_layers,batch_size);
    end
    
    for ii = 1:ceil(num_train/batch_size)
        R = randperm(num_train);
        index = R(1:batch_size);
        img = single(XTrain(:,:,:,index));
        img = imresize(img, net.meta.normalization.imageSize(1: 2));
        im_ = img - repmat(net.meta.normalization.averageImage,1,1,1,size(img,4));
        im_ = gpuArray(im_);
        
        res = vl_simplenn(img_net,im_);
        n_layers = numel(img_net.layers);
        output = gather(squeeze(res(end).x));
        F(:,index) = tanh(output);
        
        F1 = F*ones(num_train,1);
        dJdB = 2*gamma*(F(:,index)-B(:,index)) + 2*eta*repmat(F1,1,numel(index));
        dJdLogloss = 0.5*G*(1 ./ (1+exp(-0.5*G'*F(:,index))) - S(:,index));
        
        dfdAS = 2*B*(B'*F(:,index) - bit*S(:,index));
        
        dJdFb = tau*dJdLogloss + dJdB + dfdAS;
        
        dJdFb = dJdFb.*(1 - (F(:,index)).^2);
        
        dJdFb = reshape(dJdFb,[1,1,size(dJdFb,1),size(dJdFb,2)]);
        dJdFb = gpuArray(single(dJdFb));
        
        res = vl_simplenn(img_net,im_, dJdFb);
        img_net = update_net2(img_net,res,lr_img(epoch),num_train,n_layers,batch_size);
    end
    
    l = calc_loss(S,F,G,B,gamma,eta,num_train,tau,bit);
    fprintf('...epoch: %3d/%d\tloss:%3.3f\n',epoch,maxIter,...
        l);
    loss(epoch) = l;    
if mod(epoch,10)==0
fprintf('...training finishes\n');
[rBS] = generateImgCode2(img_net,txt_net,XRetrieval,bit);
[qBS] = generateImgCode2(img_net,txt_net,XQuery,bit);
rBS = compactbit(rBS > 0);
qBS = compactbit(qBS > 0);
%%
fprintf('...encoding finishes\n');
% hamming ranking
result.hri2i_S = calcMapTopkMapTopkPreTopkRecLabel(queryLabel, retrievalLabel, qBS, rBS,500);
% hash lookup
result.hli2i_S = calcPreRecRadiusLabel(queryLabel, retrievalLabel, qBS, rBS);
fprintf('...epoch:%d/ I2I:%.4f\n',epoch,result.hri2i_S.map);
fprintf('top@500...epoch:%d/  XS:%.4f/  PreS:%.4f\n',epoch,result.hri2i_S.topkMap,result.hri2i_S.topkPre);

end
end

result.loss = loss;
end

function l = calc_loss(S,F,G,B,gamma,eta,num_train,tau,bit)
theta = 0.5*F'*G;
Logloss = log(1+exp(theta))-theta.*S;
l = tau*sum(Logloss(:))+gamma*(norm(F-B,'fro')^2+norm(G-B,'fro')^2)+...
            eta*(norm(F*ones(num_train,1),'fro')^2+norm(G*ones(num_train,1),'fro')^2)+...
            (norm(B'*F - bit*S,'fro')^2+norm(B'*G - bit*S,'fro')^2);
end

function B = generateImgCode2(img_net,txt_net,images,bit)
batch_size = 256;
num = size(images,4);
B = zeros(num,bit);
for i = 1:ceil(num/batch_size)
    index = (i-1)*batch_size+1:min(i*batch_size,num);
    image = single(images(:,:,:,index));
    im_ = imresize(image,img_net.meta.normalization.imageSize(1:2));
    im_ = im_ - repmat(img_net.meta.normalization.averageImage,1,1,1,size(im_,4));        
    res = vl_simplenn(img_net,gpuArray(im_));
    output = gather(squeeze(res(end).x));
    
    im_ = imresize(image,txt_net.meta.normalization.imageSize(1:2));
    im_ = im_ - repmat(txt_net.meta.normalization.averageImage,1,1,1,size(im_,4));        
    res = vl_simplenn(txt_net,gpuArray(im_));
    output2 = gather(squeeze(res(end).x));
    
    B(index,:) = sign(0.5*(output'+output2'));
end
end
