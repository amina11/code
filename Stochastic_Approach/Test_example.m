function   [test_misclassification,labels]=Test_example(data,output_path,total,lambda2,lambda_2)
%load('/user/ai4/amina/classification/8dataset/20news/XYS_dictionary/Reduction_5S10.mat');
setenv('LC_ALL','C')
load(data);

featureD=size(train.x,2);
outputDim=size(train.y,2);
n=size(train.x,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=floor(n*0.2);
val_x=train.x(1:m,:);
val_y=train.y(1:m,:);
train_x=train.x(m+1:end,:);
train_y=train.y(m+1:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %rand('state',0)
 nn = nnsetup([featureD 1000 500 100  outputDim],lambda_2);
 opts.numepochs =2500;   %  Number of full s weeps through data
 opts.minibatchszie=20; 
 opts.batchnum=5;
 opts.plot = 0;
 [best,nn, L,loss] = nntrain(nn, train_x, train_y, opts, val_x, val_y,SS,total,lambda2);
[er, bad, labels] = nntest(nn, test.x, test.y);
[best.er, best.bad, best.labels] = nntest(best.net, test.x, test.y);
 test_misclassification=er;
 train_er=loss.train.e;   
 val_er=loss.val.e;
param.lambda=lambda2;
param.c=total;
param.L2=lambda_2;
% assert(er < 0.08, 'Too big error');
mkdir(output_path);
save([ output_path '/error.mat'],'nn','train_er','best','val_er','test_misclassification','labels','param')


