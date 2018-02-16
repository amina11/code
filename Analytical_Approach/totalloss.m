function err=totalloss(y,pred_Y,W1,W2,delta1,S,lambda_1,lambda_2)
err=sum(-y'.*log(pred_Y)-(1-y').*log(1-pred_Y))+lambda_1*regularizer(W1,delta1,S)+lambda_2*(norm(W2,'fro'))^2+lambda_2*(norm(W1,'fro'))^2;
