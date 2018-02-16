function nn = nnff(nn, x, y,S,total)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);  %mini batch size, instance num
  
    %%%%%%%%%%%augment the data x to x1, x2 according to the similarity
 
    if nargin==5;
    %get the feature indexes we want to augment
    fd=size(x,2);  %feature dimension
    S=S-eye(fd);   %eleminate the diagonals which is all 1
    [feature_1, feature_2]=find(S~=0);  % pick the indexex corresponding to nozeros
    nnS=size(feature_1,1);  %number of pairs which are similar
    se=randi([1, nnS],m,1);
    nn.target1=feature_1(se);
    nn.target2=feature_2(se);
    %% first aumentation on x
    xx11=zeros(m,fd);
    ffff=1:m;
    linearInd1 = sub2ind(size(xx11), ffff', nn.target1);  %gives the linear index of corresponding chosen pair features
    rrr=randi([0,total],m,1);
    xx11(linearInd1)=rrr;
    xx22=zeros(m,fd);
    linearInd2 = sub2ind(size(xx22), ffff', nn.target2);  %gives the linear index of corresponding chosen pair features
    xx22(linearInd2)=total-rrr;
    x1=x+xx11+xx22;
    %% second augmentation on x
    x2_xx1=zeros(m,fd);
    x2_xx2=zeros(m,fd);
    sss=randi([0,total],m,1);
    x2_xx1(linearInd1)=sss;
    x2_xx2(linearInd2)=total-sss;
    x2=x+x2_xx1+x2_xx2;
    end 
    
    %%%%adding the baies
    x = [ones(m,1) x];   %original instances
    nn.a{1} = x;    %activations corresponding to x
  
    
    if nargin == 5
    x1= [ones(m,1) x1];  %pertubed x1 from x
    x2= [ones(m,1) x2];  %pertubed x2 from x
    nn.a1{1}=sparse(x1);    %activations corresponding to x1
    nn.a2{1}=sparse(x2);    %activations corresponding to x2
    end 

    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
                if nargin==5
                nn.a1{i} = sigm(nn.a1{i - 1} * nn.W{i - 1}');
                nn.a2{i} = sigm(nn.a2{i - 1} * nn.W{i - 1}');
                end 
                %nn.a{i}(nn.a{i}< 0.001)=0.0001;
                %nn.a{i}(nn.a{i}>0.99)=0.999;
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
                if nargin==5
                nn.a1{i} = tanh_opt(nn.a1{i - 1} * nn.W{i - 1}');
                nn.a2{i} = tanh_opt(nn.a2{i - 1} * nn.W{i - 1}');
                end 
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
                if  nargin==5;
                    nn.a1{i} = nn.a1{i}.*nn.dropOutMask{i};
                    nn.a2{i} = nn.a2{i}.*nn.dropOutMask{i};
                end 
            end
        end
        
   
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
 
        if nargin==5
        nn.a1{i} =[ones(m,1) nn.a1{i}];
        nn.a2{i} =[ones(m,1) nn.a2{i}];
        end 
        
    end
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
            nn.a{n}(nn.a{n}< 0.001)=0.001;
            nn.a{n}(nn.a{n}>0.99)=0.99;  
            if nargin ==5
            nn.a1{n} = sigm(nn.a1{n - 1} * nn.W{n - 1}');
            %nn.a1{n}(nn.a1{n}< 0.001)=0.001;
            %nn.a1{n}(nn.a1{n}>0.999)=0.999;

            nn.a2{n} = sigm(nn.a2{n - 1} * nn.W{n - 1}');
            %nn.a2{n}(nn.a2{n}< 0.001)=0.001;
            %nn.a2{n}(nn.a2{n}>0.999)=0.999;
            end 
         case 'linear'
             nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';            
             nn.a1{n} =nn.a1{n - 1} * nn.W{n - 1}';
             nn.a2{n} =nn.a2{n - 1} * nn.W{n - 1}';
% 
%           case 'softmax'   %%%%%%% didnt modify for the augmented data yet!!!!!!!!!!!!!! pay attention
%               nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
%               nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
%               nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
            


    end

    %error and loss
    nn.e = y - nn.a{n};
    if nargin==5
    nn.e1 =nn.a1{n} - nn.a2{n};   %% for the derivative y1-y2
    end 
    
    switch nn.output
        case {'sigm', 'linear'}
           % nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
           nn.L = -sum(sum(y .* log(nn.a{n}))+sum((1-y) .*log(1-nn.a{n})))/ m;
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
