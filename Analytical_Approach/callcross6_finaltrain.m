 function callcross6_finaltrain(dataname,lambda_1,lambda_2,pathname)

setenv('LC_ALL','C')
data= load(dataname);
%pathname='debug2';
% lambda_1=0.001;
% lambda_2=0;
 
%load('/user/ai1/amina/WMD_datasets/originaldata/reuter/XYS_dictionary/datapartition/1/data.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %instancenum=size(data.trainx,1);
 num_test=size(data.test.x,1);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %config.threshholdparameter=10;
 config.learningrate=1;
 config.numepochs=1000;
 config.numbatch=5;
 %config.batchsize=floor((instancenum/5)*0.8);
 config.batchsize=5;
 config.hiddensize=100;
 config.momentum=0.5;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 model=mainfunction4_finaltrain(data, lambda_1, lambda_2, config);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 prednet=feedforward(data.test.x,model.W1,model.W2,model.b1,model.b2);

prednet_Best=feedforward(data.test.x,model.optimalmodel.W1,model.optimalmodel.W2,model.optimalmodel.b1,model.optimalmodel.b2);
predY=prednet.pred_Y;
 [fff,classy]=max(predY');
[fff_Best,classy_Best]=max(prednet_Best.pred_Y');
for i=1:num_test
 [fff,truecalssy(i)]=find(data.test.y(i,:)==1);
end 
 errornum=size(find(truecalssy~=classy),2);
errornum_Best=size(find(truecalssy~=classy_Best),2); 
misclassification=errornum/num_test;
misclassification_Best=errornum_Best/num_test;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%test error cross entropy
%calculate the validation error to determine stopping crrteria
testerror=crossentropyloss(data.test.x,data.test.y,model.W1,model.W2,model.b1,model.b2);
fprintf(1,'testerror %d\n',testerror)
mkdir(pathname);
param=lambda_1;
save(['./' pathname '/error.mat'],'testerror','misclassification_Best','classy_Best','param','misclassification','truecalssy','classy','model');

