 function callcross_CV(dataname,lambda_1,lambda_2,pathname)
 load(dataname);
 %lambda_1=0.001;
 %lambda_2=0;
 
%load('/user/ai1/amina/WMD_datasets/originaldata/20news/XYS_dictionary/datapartition/1/data.mat');
data.trainx=data.trainx(:,1:100);
data.SS=data.SS(1:100,1:100);
data.validationx=data.validationx(:,1:100);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 instancenum=size(data.trainx,1);
 num_test=size(data.testx,1);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %config.threshholdparameter=10;
 config.learningrate=1;
 config.numepochs=100;
 config.numbatch=5;
 %config.batchsize=floor((instancenum/5)*0.8);
 config.batchsize=5;
 config.hiddensize=100;
 config.momentum=0.5;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 model=mainfunction4_gradientclip(data, lambda_1, lambda_2, config);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 prednet=feedforward(data.validationx,model.W1,model.W2,model.b1,model.b2);
 predY=prednet.pred_Y;
 [fff,classy]=max(predY');
for i=1:num_test
 [fff,truecalssy(i)]=find(data.validationy(i,:)==1);
end 
 errornum=size(find(truecalssy~=classy));
 misclassification=errornum/num_test;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%test error cross entropy
%calculate the validation error to determine stopping crrteria
testerror=crossentropyloss(data.testx,data.testy,model.W1,model.W2,model.b1,model.b2);
fprintf(1,'testerror %d\n',testerror)
mkdir(pathname);
save(['./' pathname '/error.mat'],'testerror','model','misclassification','truecalssy','classy')

