%%test error
function error=crossentropyloss(x,y,W1,W2,b1,b2)
Val_n=size(x,1);
net=feedforward(x,W1,W2,b1,b2);
%%%% clip the value of predicted y to avoid nan value
format long
net.pred_Y(net.pred_Y<0.0001)=0.0001;
 net.pred_Y(net.pred_Y>0.9999)=0.9999;
error=sum(sum(-y.*log(net.pred_Y)-(1-y).*log(1-net.pred_Y)))/Val_n;
