clc;
clear;
trainset=csvread('trainset.csv');
testset=csvread('testset1.csv');
trainx=trainset(:,1:59);
trainy=trainset(:,60);
testx=testset(:,1:59);
testy=testset(:,60);
[numsamp,numfea]=size(trainx);
train_y=zeros(size(trainy));
test_y=zeros(size(testy));
regtrain_y=zeros(size(trainy));
regtest_y=zeros(size(testy));
regtestreg_y2=zeros(size(testy))
T=1400;
trainypopsite=find(trainy>T);
train_y(trainypopsite)=1;
testypopsite=find(testy>T);
test_y(testypopsite)=1;
%%%normalize
meantrain=mean(trainx);
stdtrain=std(trainx);
meantrainy=mean(trainy);
stdtrainy=std(trainy)
meantest=mean(testx);
stdtest=std(testx);
%meantesty=mean(testy);
%stdtesty=std(testy);

for i=1:numfea
train_x(:,i)=(trainx(:,i)-meantrain(i))/stdtrain(i);
test_x(:,i)=(testx(:,i)-meantest(i))/stdtest(i);
end
ytrain=(trainy-meantrainy)/stdtrainy;
%ytest=(testy-meantesty)/stdtesty;
w=(train_x'*train_x)^-1*train_x'*ytrain;
regtrainy=train_x*w;
regtesty=test_x*w;
regtraysite=find(regtrainy>T);
regtrain_y(regtraysite)=1;
regtestysite=find(regtesty>T);
regtest_y(regtestysite)=1;
msetrain=mean(regtrain_y~=train_y);
Mse=mean(regtest_y~=test_y);
%%%%%%%%%%55%%%5
Nfolds=10;
params=(0:0.01:1);
[model1, bestParam1, mu1, se1] =fitCv(params,@(X, y, l)linregFit(X, y, 'lambda', l, 'regType', 'L2'),@linregPredict,@zeroOneLossFn,train_x,ytrain,Nfolds);
yhatTest = linregPredict(model1, test_x);
regtestpopsite2=find(yhatTest>T);
regtestreg_y2(regtestpopsite2)=1;
errRateTest = mean( (regtestreg_y2 ~= test_y) );
