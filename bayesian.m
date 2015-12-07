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
[model, logev, postSummary] = linregFitBayes(train_x, ytrain);
regtesty = linregPredictBayes(model, test_x);
regtrainy=linregPredictBayes(model,train_x);
regtrainysite=find(regtrainy>T);
regtrain_y(regtrainysite)=1;
regtestysite=find(regtesty>T);
regtest_y(regtestysite)=1;
errRateTest = mean( (regtest_y ~= test_y) );
errRatetrain= mean(regtrain_y~= train_y);
