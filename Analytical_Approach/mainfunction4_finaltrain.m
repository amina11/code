%%%this function is for derivative model with one layer network%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=mainfunction4_finaltrain(data, lambda_1, lambda_2, config)
tstart2=tic;

datasize=size(data.train.x,1);
earlystopingnum=round(datasize*0.2);
validation_x=data.train.x(1:earlystopingnum,:);
validation_y=data.train.y(1:earlystopingnum,:);
S=data.S10;
trainingdata=data.train.x(earlystopingnum+1:datasize,:);
traininglabel=data.train.y(earlystopingnum+1:datasize,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=size(trainingdata,1);   % sample size 
assert(size(trainingdata,1)==size(traininglabel,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % key parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hidden_size=config.hiddensize;
max_epochs=config.numepochs;                 %epoch size
batchsize=config.batchsize;
num_batch=config.numbatch;
featureDim=size(trainingdata,2); %feature dimention
outputDim=size(traininglabel,2); %output dimention%batchsize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intialization of weight matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%rng(1);
rand('state',0);
W1=-sqrt(6 / (hidden_size + featureDim))+2*sqrt(6 / (hidden_size + featureDim))*rand(hidden_size,featureDim);
b1=-sqrt(6 / (hidden_size + 1))+2*sqrt(6 / (hidden_size +1))*rand(hidden_size,1);
W2=-sqrt(6 / (hidden_size + outputDim))+sqrt(6 / (hidden_size + outputDim))*2*rand(outputDim,hidden_size);
b2=-sqrt(6 / (outputDim+1))+rand(outputDim,1)*2* sqrt(6/ (outputDim+1));
batch_gradW1=0;
batch_gradW2=0;
batch_gradb1=0;
batch_gradb2=0;

mm1=zeros(hidden_size,featureDim);
vv1=zeros(hidden_size,featureDim);
aa=0.001;
beta1=0.9;
beta2=0.999;
mm2=zeros(outputDim,hidden_size);  %for W1
vv2=zeros(outputDim,hidden_size);   %%%for W2
mm3=zeros(hidden_size,1);  %for b2
vv3=zeros(hidden_size,1);  %for b2
mm4=zeros(outputDim,1);   %for  b1
vv4=zeros(outputDim,1);

mincounter=0;
minimal_validationerror=inf;
bestModel.W1=zeros(hidden_size,featureDim);
bestModel.W2=zeros(outputDim,hidden_size);
bestModel.b1=zeros(hidden_size,1);
bestModel.b2=zeros(outputDim,1);
validationerror=zeros(max_epochs,1);
trainingerror=zeros(max_epochs,1);
for epoch=1:max_epochs   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here begins one epoch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP1 : we shuffle the data
startbig=tic;
data1 = [traininglabel, trainingdata];
data1 = data1(randperm(size(data1,1)),:);
y = data1(:,1:outputDim);
X = data1(:,outputDim+1:end);

fprintf(1,'epoch %d\n',epoch);
    r=0;
for b=1:num_batch 
 batch_x=X((num_batch-1)*batchsize+1:num_batch*batchsize,:);
 batch_y=y((num_batch-1)*batchsize+1:num_batch*batchsize,:);
net=feedforward(batch_x,W1,W2,b1,b2);


for q=1:batchsize % batch size
    tstartsmall=tic;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step2 delta updatate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 def_z1=sigmoidderivative(net.z1(q,:)'); %delta1=W2'*delta2.* z1';the multiplication is column wise
 delta2=diag(sigmoidderivative(net.z2(q,:)'));
def_z2=sigmoidderivative(net.z2(q,:)'); 
 WW2=W2'*delta2;
 sizew=size(WW2,2);
 delta1=def_z1(:, ones(1,sizew)).*WW2; %(:, ones(1,sizew)) will duplicate the vector to sizew columns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step3 gradient calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gradient for the first layer
gradW1_part1=(W2'.*def_z1(:, ones(1,size(W2,1))))*(net.pred_Y(q,:)'-batch_y(q,:)')*net.z0(q,:)+2*lambda_2*W1;
 ttt=0;
 W14=0;
rrr2=0;
rrr4=0;
gradW1_part2=zeros(size(W1));
for j=1:featureDim
         bb=W1(:,j);
         nonzerosindex=find(S(:,j)~=0);
         f=size(nonzerosindex,1);
        gradW1_part2(:,j)=4*lambda_1*(delta1*(delta1'*(-W1(:,nonzerosindex)+bb(:,ones(1,f)))))*S(nonzerosindex,j);
     SS1=(((W2.*def_z2(:,ones(hidden_size,1)))'*(delta1'*(W1(:,nonzerosindex)-bb(:,ones(1,f))))).*(W1(:,nonzerosindex)-bb(:,ones(1,f))))*S(nonzerosindex,j);
  ttt=ttt+SS1;
  %part4
  z2l=((1-2*net.z2(q,:)').*def_z2);
  part444=((W2.*z2l(:,ones(hidden_size,1)))'* (((W2.*(def_z1(:,ones(outputDim,1)))')*(W1(:,nonzerosindex)-bb(:,ones(1,f)))).*((delta1)'*(W1(:,nonzerosindex)-bb(:,ones(1,f))))))*S(nonzerosindex,j);
  W14=W14+part444;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  W11=(W1(:,nonzerosindex)-bb(:,ones(1,f))).*(def_z1(:,ones(1,f)));
  rrr=((W2*W11).^2)*S(nonzerosindex,j);
  rrr2=rrr+rrr2;
 %%%%%%%%%%%%%%%%%%%%%%part 2
  ss=S(nonzerosindex,j)';
  rrr3=((W2.*(def_z2(:,ones(1,hidden_size))).^2)*W11)*(W11.*ss(ones(hidden_size,1),:))';
  rrr4=rrr3+rrr4;
end  
 commonterm2=ttt.*def_z1.*(1-2*net.z1(q,:)');
 gradW1_part3=2*lambda_1*commonterm2*net.z0(q,:);
 gradW1_part4=2*lambda_1*(W14.*def_z1)*net.z0(q,:);
 gradW1=gradW1_part1+ gradW1_part2+gradW1_part3+gradW1_part4;
 gradb1=(W2'.*def_z1(:, ones(1,size(W2,1))))*(net.pred_Y(q,:)'-batch_y(q,:)')+2*lambda_1*(W14.*def_z1)+2*lambda_1*commonterm2; 

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 gradW2_part3=2*lambda_1*(rrr2.*(1-2*(net.z2(q,:)')).*(def_z2).^2)*net.z1(q,:);
 gradW2_part2=2*lambda_1*rrr4;
 gradW2_part1=(net.pred_Y(q,:)'-batch_y(q,:)')*net.z1(q,:)+2*lambda_2*W2;
 gradW2=gradW2_part1+gradW2_part2+gradW2_part3;
 gradb2=(net.pred_Y(q,:)'-batch_y(q,:)')+2*lambda_1*(rrr2.*(1-2*(net.z2(q,:)')).*(def_z2).^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% should add something like if gradW1 is nan then break ?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test gradient with finite difference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%diffW=testgradietn(W1,W2,b1,b2,lambda_1,lambda_2,S,1,3,y,eta,r,1,0,0,0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%accumulate batch gradient
batch_gradW1=batch_gradW1+gradW1;
batch_gradW2=batch_gradW2+gradW2;
batch_gradb1=batch_gradb1+gradb1;
batch_gradb2=batch_gradb2+gradb2;
telapsedsmall=toc(tstartsmall);
end 
%%%%%%%%%%%%%%%%%%%%%%%%
% step4 update gradient for each epoch  ADAM
%%%clipping the gradient
norm_batchgradientW1=norm(batch_gradW1);
norm_batchgradientW2=norm(batch_gradW2);
norm_batchgradientb1=norm(batch_gradb1);
norm_batchgradientb2=norm(batch_gradb2);


if   norm_batchgradientW1>10
     batch_gradW1=10*batch_gradW1/norm_batchgradientW1;
end 

if   norm_batchgradientW2>10
     batch_gradW2=10*batch_gradW2/norm_batchgradientW2;
end
if   norm_batchgradientb1>10
     batch_gradb1=10*batch_gradb1/norm_batchgradientb1;
end

if   norm_batchgradientb2>10
     batch_gradb2=10*batch_gradb2/norm_batchgradientb2;
end
 

mm1=mm1*beta1+(1-beta1)*batch_gradW1;
vv1=beta2*vv1+(1-beta2)*(batch_gradW1).^2;
aa1=aa*sqrt(1-beta2^epoch)/(1-beta1^epoch);
W1=W1-aa1*mm1./(sqrt(vv1)+1e-8);
%%%%%%%
mm2=mm2*beta1+(1-beta1)*(batch_gradW2);
vv2=beta2*vv2+(1-beta2)*(batch_gradW2).^2;
W2=W2-aa1*mm2./(sqrt(vv2)+1e-8);
%%%%bias %%
mm3=mm3*beta1+(1-beta1)*(batch_gradb1);
vv3=beta2*vv3+(1-beta2)*(batch_gradb1).^2;
b1=b1-aa1*mm3./(sqrt(vv3)+1e-8);
%%%%%%%%%%
mm4=mm4*beta1+(1-beta1)*(batch_gradb2);
vv4=beta2*vv4+(1-beta2)*(batch_gradb2).^2;
b2=b2-aa1*mm4./(sqrt(vv4)+1e-8);
  
%initialzie batch gradient for each epoch
batch_gradW1=0;
batch_gradW2=0;
batch_gradb1=0;
batch_gradb2=0;
end 
telapsedbig=toc(startbig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ends one epoch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calculate the validation error to determine stopping crrteria  
validationerror(epoch)=crossentropyloss(validation_x,validation_y,W1,W2,b1,b2);
fprintf(1,'validationerror %d\n',validationerror(epoch))

%%%%%%%%%%%%%%%%break if the validation error increase continouesly
 if epoch>1
   previouserror=validationerror(epoch-1);
else 
  previouserror=inf;
end 

 if validationerror(epoch)-previouserror>0
 mincounter=mincounter+1
 else  mincounter=0;
 end 
 
  if mincounter>12
     break 
  end 

%%%%%%%%% keep trackingt he minimal error and  saving corresponding model

       if  validationerror(epoch)<minimal_validationerror
          minimal_validationerror=validationerror(epoch);
          bestiteration=epoch;
          bestModel.W1=W1;
          bestModel.W2=W2;
          bestModel.b1=b1;
          bestModel.b2=b2;
        else
         minimal_validationerror=minimal_validationerror;
         bestModel=bestModel;
         bestiteration=bestiteration;
         end

trainingerror(epoch)=crossentropyloss(trainingdata,traininglabel,W1,W2,b1,b2);
fprintf(1,'trainingerror %d\n',trainingerror(epoch))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end 


convergetime=toc(tstart2);
%%%%%%%%%%%%%%%%%
model.W1=W1;
model.W2=W2;
model.b1=b1;
model.b2=b2;
model.trainingerror=trainingerror;
model.validationerror=validationerror;
model.optimaiteration=bestiteration;
model.optimalValidationerror=minimal_validationerror;
model.optimalmodel=bestModel;
model.time=convergetime;

