%testing the derivative with finite difference mathod
function R1=regularizer(W1,delta1,S)
k=size(S,1);
R1=0;
for i=1:k
    for j=1:k
       ff=(W1(:,i)-W1(:,j))'*delta1;
       ss=(norm(ff)^2)*S(i,j);
       R1=R1+ss;
    end 
end 