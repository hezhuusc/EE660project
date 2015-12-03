clc;
clear;
trainset=csvread('trainset.csv');
%testset=csvread('testset1.csv');
trainx=trainset(:,1:59);
trainy=trainset(:,60);
%testx=testset(:,1:59);
%testy=testset(:,60);
%%%%
train_y=zeros(size(trainy));
%test_y=zeros(size(testy));
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
%meantest=mean(testx);
%stdtest=std(testx);
S2=1:1:numsamp;
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
for itr = 1:nitr       
        SampleRows = randperm(numsamp,2000);
        SampleRows2(itr,:)=SampleRows;
        ValiRows = setdiff (S2,SampleRows);
        VA(itr,:)=ValiRows;
        xtrain = train_x(SampleRows,:);
        ytrain=train_y(SampleRows,:);
        Valix = train_x(ValiRows,:);
        Valiy=train_y(ValiRows,:);
        rfsize=size(randomfeatures',1);
        RFrows = randperm(rfsize);
        RFrows= RFrows(1);
        RF(itr,:)=RFrows;
        bagsz=size(Bag',1);
        Brows= randperm(bagsz);
        Brows1= Bag(Brows);
        Browsn=Brows1(1);
        BS(itr,:)=Browsn;
		forest = fitForest(xtrain,ytrain,'randomFeatures',RFrows,'bagSize',Browsn,'ntrees',ntree);
		yhat_test = predictForest(forest,Valix);
		errs_test(itr) = mean(Valiy ~= yhat_test);
 end
%end
disp('finished');