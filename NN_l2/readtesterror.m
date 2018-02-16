%%%%%%%%%%%%%%%%%%%% data bbc. 
%function     readtesterror(dataname)

hidden_num=[10,100,200,300,400,500,600,700,800,900,1000];
lambda2=[0,0.01,0.1,10,100,1000];
num_lambda1=size(hidden_num,2);
num_lambda2=size(lambda2,2);
for i=1:num_lambda1
for j=1:num_lambda2
if exist(['/user/ai4/amina/classification/NN2/NN/8dataset/twitter/XYS_dictionary/output_1L_400_5b_gradclip/H' num2str(hidden_num(i)) '/lambda2' num2str( lambda2(j))  '/error.mat'], 'file')
    load(['/user/ai4/amina/classification/NN2/NN/8dataset/twitter/XYS_dictionary/output_1L_400_5b_gradclip/H' num2str(hidden_num(i)) '/lambda2' num2str( lambda2(j))  '/error.mat'],'mean_testerror');
%testingerrorfolder=testingerrorfolder+mean_testerror;
Testerroroture(i,j)=mean_testerror;
else Testerroroture(i,j)=inf;
end
end
end
save('/user/ai4/amina/classification/NN2/NN/8dataset/twitter/XYS_dictionary/result.mat','Testerroroture')
twitter=[lambda2; Testerroroture];
lambda1=[0;hidden_num'];
format shortG
twitter=[lambda1,twitter]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

