function   [test_misclassification]=Test_example_cv(data,output_path,lambda2)

setenv('LC_ALL','C')
%load('/user/ai4/amina/classification/8dataset/20news/XYS_dictionary/Reduction_5S10.mat');
load(data);


featureD=size(data.trainx,2);
outputDim=size(data.trainy,2);
n=size(data.trainx,1);


%data22=data11(randperm(n),:);
m=floor(n*0.2);
test_x=data.validationx;
test_y=data.validationy;
val_x=data.trainx(1:m,:);
val_y=data.trainy(1:m,:);
train_x=data.trainx(m+1:end,:);
train_y=data.trainy(m+1:end,:);

%%
 %rand('state',0)
 nn = nnsetup([featureD 500 100  outputDim],lambda2);
 opts.numepochs =500;   %  Number of full s weeps through data
 opts.minibatchszie=20; 
 opts.batchnum=5;
 opts.plot = 0;
 [nn, L,loss] = nntrain(nn, train_x, train_y, opts, val_x, val_y);
 [er_test, bad, labels_test] = nntest(nn, test_x, test_y);
 test_misclassification=er_test;
 train_er=loss.train.e; 
 val_er=loss.val.e;
% assert(er < 0.08, 'Too big error');
 mkdir(output_path);
save(['./' output_path '/error.mat'],'nn','train_er','val_er','test_misclassification','labels_test','test_y','loss')


