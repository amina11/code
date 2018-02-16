function net=feedforward(x,W1,W2,b1,b2)
net.z0=x;
b1=b1';
 net.a1=net.z0*W1'+b1(ones(size(net.z0,1),1),:);
 net.z1=sigmoid(net.a1);
b2=b2';
 net.a2=net.z1*W2'+b2(ones(size(net.z1,1),1),:);
 net.z2=sigmoid(net.a2);
 net.pred_Y=net.z2;
 net.pred_Y(net.pred_Y<0.0001)=0.0001;
 net.pred_Y(net.pred_Y>0.9999)=0.9999;
 
