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
trc2=trainx(trainypopsite,:);
ptrmean=mean(trainx(trainypopsite,:));
trainonpop=find(trainy<=T);
trc1=trainx(trainonpop,:);
ntrmean=mean(trainx(trainonpop,:));
testypopsite=find(testy>T);
test_y(testypopsite)=1;
%%%%
[numsamp,numfea]=size(trainx);
%%%
error1=0;
error2=0;
for i=1:length(trc1)
    dt1(i)=sqrt(sum((trc1(i,:)-ntrmean(1,:)).^2));
    dt2(i)=sqrt(sum((trc1(i,:)-ptrmean(1,:)).^2));
    if dt2(i)<dt1(i)
        error1=error1+1;
    end
end
for j=1:length(trc2)
    dt3(j)=sqrt(sum((trc2(j,:)-ptrmean(1,:)).^2));
    dt4(j)=sqrt(sum((trc2(j,:)-ntrmean(1,:)).^2));  
    if dt4(j)<dt3(j)
        error2=error2+1;
    end
end
erate1=error1./length(trc1);
erate2=error2./length(trc2);