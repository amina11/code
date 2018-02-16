
function testingerror=readtesterror(lambda_1, lambda_2)
num_lambda_1=size(lambda_1,2);
num_lambda_2=size(lambda_2,2);
for i=1:num_lambda_1
for j=1:num_lambda_2
load(['./output/lambda1' num2str(lambda_1(i)) '/lambda2' num2str( lambda_2(j)) '/error.mat'],'testerror');
testingerror(i,j)=testerror;
end
end

