clc;
clear;
%%load data%%%%%%
trainset=csvread('Validation1.csv');
testset=csvread('testset1.csv');
trainx=trainset(:,1:59);
trainy=trainset(:,60);
testx=testset(:,1:59);
testy=testset(:,60);
%%%%
train_y=zeros(size(trainy));
test_y=zeros(size(testy));
T=1400;
trainypopsite=find(trainy>T);
train_y(trainypopsite)=1;
trc2=trainx(trainypopsite,:);
ytrc2=train_y(trainypopsite,:);
trainonpop=find(trainy<=T);
trc1=trainx(trainonpop,:);
ytrc1=train_y(trainonpop,:);
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
%nLambda = 20;
%lambdas = logspace(-6,1,nLambda);
Nfolds = 10;
%[model, Lstar, mu, se] = fitCv(lambdas, ...
%@logregFit, @logregPredict, @zeroOneLossFn, ...
%train_x, train_y, Nfolds);
params=(0:0.01:1);
%[mod1]= logregFit(x_train1,ytrain,'regType','L2');
%[ypre_tra,pyy]= logregPredict(mod1,x_train1);
%mr=sum(ypre_tra ~= ytrain)/m1;
[model1, bestParam1, mu1, se1] =fitCv(params,@(X, y, l)logregFit(X, y, 'lambda', l, 'regType', 'L2'),@logregPredict,@zeroOneLossFn,train_x,train_y,Nfolds);
[ypred_train,py1]= logregPredict(model1,train_x);
[ypred_test,py1_test]= logregPredict(model1,test_x);
yhatTrain = logregPredict(model1, train_x);
errRateTrain = mean( (yhatTrain ~= train_y) );
yhatTest = logregPredict(model1, test_x);
errRateTest = mean( (yhatTest ~= test_y) );
site1 = find(yhatTest(trainypopsite)==1);
[size1,lie] = size(site1);
[sizeall1,lie]= size(trainypopsite);
tp=size1/sizeall1;
figure(1)
a1=trc1(:,26);
a2=trc2(:,26);
b1=trc1(:,27);
b2=trc2(:,27);
plot(a1,b1,'*');
hold on
plot(a2,b2,'.');