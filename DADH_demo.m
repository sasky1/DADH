function DADH_demo()
%% before you running this code, please change direct to matconvnet and run setup.m to setup MatConvNet.
% As this is an old version of MatConvNet (and image-net-vgg-f.mat), if you
% complie a new version, please download correpsonding pretrained
% image-net-vgg-f.mat, and maybe you need rewrite update_net function by
% yourself.
addpath(fullfile('utils'));
dataname = 'IAPR-TC12';
%% load dataset
[dataset param] = load_data(dataname);
% numData = size(dataset.LAll, 1);
% param = gen_index(numData);
%% basic parameters
bits = [48];
param.dataname = dataname;
param.method = 'DADH';
param.batch_size = 64;
param.maxIter = 150;
%% hypper-parameters, please cross-validate the following params if you use
% these code for new datasets.
param.gamma = 100;
param.eta = 10;
param.tau = 10;
% param.lr = logspace(-3, -6, param.maxIter);
param.lr = logspace(-4, -6, param.maxIter);
%% training and evaluation
for i = 1: length(bits)
    param.bit = bits(i);
    result = process_DADH(dataset, param);
end
end


function [dataset,param] = load_data(dataname)
switch dataname
    case 'IAPR-TC12'
        load './data/IAPR-TC12.mat'
    case 'FLICKR-25K'
        load ./data/FLICKR-25K;
end
numData = size(LAll, 1);
param = gen_index(numData);
dataset.IAll = IAll;
dataset.LAll = LAll;
end
function param = gen_index(numData)
R = randperm(numData);
numQuery = 2000; % 2000

param.indexQuery = R(1: numQuery);
R(1: numQuery) = [];
param.indexRetrieval = R;
param.indexTrain = R(1: 5000);
end


