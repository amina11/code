function nn = nnbp(nn,S,lambda2)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    output_dim=size(nn.a{n},2);
    minibatch_num=size(nn.a{n},1); 
    sparsityError = 0;
   
    switch nn.output
        case 'sigm'
           % d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
             d{n} = - nn.e;   % for all instances in mini batch, it is the standard propagation term, definition is different from the d1, 
             for j=1:minibatch_num 
                 d_actn1 = nn.a1{n}(j,:) .* (1 - nn.a1{n}(j,:));
                 d_actn2 = nn.a2{n}(j,:) .* (1 - nn.a2{n}(j,:));
                 d1{n}{j}=diag(d_actn1 );    % the propagation term on the output layer
                 d2{n}{j}=diag(d_actn2);
             end 
           
    %     case {'softmax','linear'}  
    %       d{n} = - nn.e;
            %d1{n}=eye(output_dim);  
            %d2{n}=eye(output_dim);
    end
    for i = (n - 1) : -1 : 2  
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
                d_act1 = nn.a1{i} .* (1 - nn.a1{i});
                d_act2 = nn.a2{i} .* (1 - nn.a2{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
            for j=1:minibatch_num
                tt=d_act1(j,:)';
                cc=d_act2(j,:)';
            d1{i}{j} =(nn.W{i}'*d1{i + 1}{j}).* tt(:,ones(1,size(d1{i + 1}{j},2)));
            d2{i}{j} =(nn.W{i}'*d2{i + 1}{j}).* cc(:,ones(1,size(d1{i + 1}{j},2)));
            end 
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
            for j=1:minibatch_num
                ff=d_act1(j,:)';
                dd=d_act2(j,:)';
            d1{i}{j} =(nn.W{i}'*d1{i + 1}{j}(2:end,:)).* ff(:,ones(1,size(d1{i + 1}{j},2)));
            d2{i}{j} =(nn.W{i}'*d2{i + 1}{j}(2:end,:)).* dd(:,ones(1,size(d1{i + 1}{j},2)));
            end 
        end
        
        if(nn.dropoutFraction>0)   %apply drop out on the hidden layers
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
            for j=1:minibatch_num
                gg=[1;nn.dropOutMask{i}(j,:)'];
                d1{i}{j}=d1{i}{j}.*gg(:,ones(1,output_dim));
                d2{i}{j}=d2{i}{j}.*gg(:,ones(1,output_dim));
          end 
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW0{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);  %derivative for the standard loss
            
            nn.dW1{i} =0;   %derivative for the data augmented part
            for j=1: minibatch_num
            nn.dW1{i} = nn.dW1{i}+2*lambda2*S(nn.target1(j),nn.target2(j))*((d1{i+1}{j}*nn.e1(j,:)')*nn.a1{i}(j,:)-(d2{i+1}{j}*nn.e1(j,:)')*nn.a2{i}(j,:));
            end
             %nn.dW{i}=nn.dW0{i}+(nn.dW1{i} / size(d{i + 1}, 1));
             nn.dW1{i}=nn.dW1{i} / size(d{i+1}, 1);
        else
            nn.dW0{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);
             nn.dW1{i} =0;
            for j=1: minibatch_num
            nn.dW1{i} = nn.dW1{i}+2*lambda2*S(nn.target1(j),nn.target2(j))*((d1{i+1}{j}(2:end,:)*nn.e1(j,:)')*nn.a1{i}(j,:)- (d2{i+1}{j}(2:end,:)*nn.e1(j,:)')*nn.a2{i}(j,:));
            end
           %nn.dW{i}=nn.dW0{i}+nn.dW1{i} / size(d{i+1}, 1);
            nn.dW1{i}=nn.dW1{i} / size(d{i+1}, 1);
        end
    end
end
