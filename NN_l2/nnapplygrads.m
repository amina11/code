function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
         
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        else
            dW = nn.dW{i};
        end
        
        %%%ADAM%%%%%%%%%%%%%%
         if (nn.ADAM>0)
            nn.mm{i}=nn.beta1*nn.mm{i}+(1-nn.beta1)*dW;
            nn.vv{i}=nn.beta2*nn.vv{i}+(1-nn.beta2)*dW.^2;
            nn.aa=nn.aa0*sqrt(1-nn.beta2)/(1-nn.beta1);
             nn.W{i} = nn.W{i} - nn.aa*nn.mm{i}./(sqrt(nn.vv{i})+1e-8);
         end 
        
         
        %%momentum%%%%%%%%%%%%%%%%%%%%%
        if(nn.momentum>0)
             dW = nn.learningRate * dW;
             nn.vW{i} = nn.momentum*nn.vW{i} + dW;
             dW = nn.vW{i};
             ndw=norm(dW);
        if ndw>10
            dW=10*dW/ndw;
        end 
        nn.W{i} = nn.W{i} - dW;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    end
end
