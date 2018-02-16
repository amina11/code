%%%%%%%%%%%%%%This file is for  final training for outer cross validation data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function   [test_misclassification,labels]=Test_example2(data_name,lambda2)
setenv('LC_ALL','C')
for  i=1:5
load([ '/user/ai1/amina/WMD_datasets/originaldata/' data_name '/XYS_dictionary/datapartition/' num2str(i) '/1/data.mat']);
featureD=size(data.trainx,2);
outputDim=size(data.trainy,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %rand('state',0)
 nn = nnsetup([featureD 1000 500 100  outputDim],lambda2);
 opts.numepochs =3000;   %  Number of full s weeps through data
 opts.minibatchszie=20; 
 opts.batchnum=5;
 opts.plot = 0;
 [nn, L,loss] = nntrain(nn, data.trainx, data.trainy, opts, data.validationx, data.validationy);
 [er, bad, labels] = nntest(nn, data.testx, data.testy);
 test_misclassification=er;
 train_er=loss.train.e;   
 val_er=loss.val.e;
truelabel=data.testy;
% assert(er < 0.08, 'Too big error');
mkdir(['/user/ai2/amina/ICML2017/NN/NN_ADAM_Sigmoid/output_L2/' data_name '/finaltrain/3layer' num2str(i)]);
save(['/user/ai2/amina/ICML2017/NN/NN_ADAM_Sigmoid/output_L2/' data_name '/finaltrain/3layer' num2str(i) '/error_1000h_500h_100h.mat'],'nn','train_er','truelabel','val_er','test_misclassification','labels')
end 


