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
%%%%


for i=1:numfea
train_x(:,i)=(trainx(:,i)-meantrain(i))/stdtrain(i);
test_x(:,i)=(testx(:,i)-meantest(i))/stdtest(i);
end
%%
Xcolu=ones(numsamp,1);
X26=train_x(:,26);
X27=train_x(:,27);
xmerge=[Xcolu,X26,X27];
%%
figure,
positive = find(train_y==1);
negtive = find(train_y==0);
%subplot(1,2,1);
hold on
plot(train_x(positive,26), train_x(positive,27), 'k+', 'LineWidth',2, 'MarkerSize', 7);
plot(train_x(negtive,26), train_x(negtive,27), 'bo', 'LineWidth',2, 'MarkerSize', 7);
%%
%[m,n] = size(x);
x = mapFeature(xmerge(:,2:3));
theta = zeros(size(x,2), 1);
lambda = 1;
[cost, grad] = cost_func(theta, x, train_y, lambda);
threshold = 0.53;
alpha = 10^(-1);
costs = [];
while cost > threshold
    theta = theta + alpha * grad;
    [cost, grad] = cost_func(theta, x, train_y, lambda);
    costs = [costs cost];
end

%


% Plot Decision Boundary 
hold on
plotDecisionBoundary(theta, x, train_y);
legend('Positive', 'Negtive', 'Decision Boundary')
xlabel('Feature Dim1');
ylabel('Feature Dim2');
%title('Classifaction Using Logistic Regression');

% Plot Costs Iteration
%hold on
%figure,
%subplot(1,2,2);
%plot(costs, '*');
%title('Cost Function Iteration');
%xlabel('Iterations');
%ylabel('Cost Fucntion Value');


