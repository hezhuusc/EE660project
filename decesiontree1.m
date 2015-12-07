clc;
clear;
%%load data%%%%%%
trainset=csvread('trainset.csv');
testset=csvread('testset1.csv');
trainx=trainset(:,1:59);
trainy=trainset(:,60);
testx=testset(:,1:59);
testy=testset(:,60);
%%%%
train_y=zeros(size(trainy));
test_y=zeros(size(testy));
treetest=zeros(size(testy));
T=1400;
trainypopsite=find(trainy>T);
train_y(trainypopsite)=1;
testypopsite=find(testy>T);
test_y(testypopsite)=1;
%%%%
[numsamp,numfea]=size(trainx);
%%%normalize
meantrain=mean(trainx);
stdtrain=std(trainx);
meantest=mean(testx);
stdtest=std(testx);
for i=1:numfea
train_x(:,i)=(trainx(:,i)-meantrain(i))/stdtrain(i);
test_x(:,i)=(testx(:,i)-meantest(i))/stdtest(i);
end
model = dtfit(train_x,train_y,'maxdepth', 10);
ytree = dtpredict(model,test_x);
classtest=find(ytree>T);
treetest(classtest)=1;
errRateTest = mean(treetest ~= test_y );