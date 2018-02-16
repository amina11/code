%%test error
validation_x=rand(10,15);
validation_y=rand(10,20);
earlystopingnum=10;
W1=rand(25,15);
W2=rand(20,25);
b1=rand(25,1);
b2=rand(20,1);
outputDim=20;

validationerror=0;
for i=1:earlystopingnum
    eta=validation_x(i,:)';
validation_net=feedforward(eta,W1,W2,b1,b2); 
error1=sum(validation_y(i,:)'.*log(validation_net.pred_Y(i,:))+(ones(outputDim,1)-validation_y(i,:)').*log(ones(outputDim,1)-validation_net.pred_Y(i,:)));
validationerror=validationerror+error1;
end 