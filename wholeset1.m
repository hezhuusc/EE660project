clc;
clear;
trainset=csvread('wholeset.csv');
trainx=trainset(:,1:59);
trainy=trainset(:,60);
[numsamp,numfea]=size(trainx);
%%%%
train_y=zeros(size(trainy));
regtrain_y1=zeros(size(trainy));
regtrain_y2=zeros(size(trainy));

%%%%
T=1400;
trainypopsite=find(trainy>T);
train_y(trainypopsite)=1;

%%%normalize
meantrain=mean(trainx);
stdtrain=std(trainx);
meantrainy=mean(trainy);
stdtrainy=std(trainy)
for i=1:numfea
train_x(:,i)=(trainx(:,i)-meantrain(i))/stdtrain(i);
end
ytrain=(trainy-meantrainy)/stdtrainy;
%%%%%%%%bayesian %%%%%%%%%%%
[model, logev, postSummary] = linregFitBayes(train_x, ytrain,'prior','uninf');
regtrainy1=linregPredictBayes(model,train_x);
regtrainysite1=find(regtrainy1>T);
regtrain_y1(regtrainysite1)=1;
errRatetrain(1)= mean(regtrain_y1~= train_y);
%%%%%%%linear regression%%%%%%%%
w=(train_x'*train_x)^-1*train_x'*ytrain;
regtrainy2=train_x*w;
regtraysite2=find(regtrainy2>T);
regtrain_y2(regtraysite2)=1;
errRatetrain(2)=mean(regtrain_y2~=train_y);
%%%%%%%logistic regression%%%%%%%
nLambda = 20;
lambdas = logspace(-6,1,nLambda);
Nfolds = 10;
[model, Lstar, mu, se] = fitCv(lambdas, ...
@logregFit, @logregPredict, @zeroOneLossFn, ...
train_x, train_y, Nfolds);
yhatTrain = logregPredict(model, train_x);
errRatetrain(3) = mean( (yhatTrain ~= train_y) );