clc;
clear;
trainset=csvread('trainset.csv');
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
%testypopsite=find(testy>T);
%test_y(testypopsite)=1;
%%%%
[numsamp,numfea]=size(trainx);
%%%normalize
meantrain=mean(trainx);
stdtrain=std(trainx);
meantest=mean(testx);
stdtest=std(testx);
S2=1:1:numsamp
randomfeatures=[2,3,4,5];
bag = 1:1:5;
Bag=1./bag;
for i=1:numfea
train_x(:,i)=(trainx(:,i)-meantrain(i))/stdtrain(i);
%test_x(:,i)=(testx(:,i)-meantest(i))/stdtest(i);
end
nitr = 5;
ntree = 15;
errs_test = zeros(nitr,1);
errs_train = zeros(nitr,1);
%for ntree = 1:1:nntree
	%ntree
	for itr = 1:nitr
        %S = size(a,1);
        SampleRows = randsample(numsamp,800,true);
        xtrain = train_x(SampleRows,:);
        ytrain =train_y(SampleRows,:);
		forest = fitForest(xtrain,ytrain,'randomFeatures',5,'bagSize',1/3,'ntrees',ntree);
		yhat_test = predictForest(forest,test_x);
		errs_test(itr) = mean(test_y ~= yhat_test);
		%yhat_train = predictForest(forest,Xtrain);
		%errs_train(itr,ntree) = mean(Ytrain ~= yhat_train);
	end
%end
disp('finished');
%%