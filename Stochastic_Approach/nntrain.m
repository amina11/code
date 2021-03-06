function [best,nn, L,loss]  = nntrain(nn, train_x, train_y, opts, val_x, val_y, S, total,lambda2)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 9 || nargin == 7,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
miscounter=0;
%opts.validation = 1;
if nargin == 9
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);
numbatches= opts.batchnum;
batchsize =opts.minibatchszie;
%batchsize=20;
numepochs = opts.numepochs;
%numbatches =floor(m / batchsize);


%%%%%%%%%%%%%%%%
minimal_validationerror=inf;
best_nn=nn;
bestiteration=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    tic;
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        
%         %%%%%%%%%%%%%%%reapet the augmentation for same batch b times
        if lambda2==0
             nn = nnff(nn, batch_x, batch_y,S,total);
             nn = nnbp_standard(nn);
        else 
            
         for h=1:nn.n-1
             GradW{h}=zeros(size(nn.W{h}));
         end
        for b=1:5
        nn = nnff(nn, batch_x, batch_y,S,total);
        nn = nnbp(nn,S,lambda2);
        for c=1:nn.n-1
            GradW{c}=GradW{c}+nn.dW1{c};
        end 
        end
        for hh=1:nn.n-1
            nn.dW{hh}=GradW{hh}+nn.dW0{hh};
        end 
        end 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        nn = nnapplygrads(nn);
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    t = toc;

    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train cross entropy = %f, val cross entropy = %f', loss.train.e(end), loss.val.e(end));
        if i>2
            previous_val_missclassification= loss.val.e(i-1);
              if loss.val.e(i)>previous_val_missclassification;
                  miscounter=miscounter+1;
              else miscounter=0;
              end 
        end 
        if miscounter>10
            break
        end 
        miscounter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if loss.val.e(i)<minimal_validationerror
             minimal_validationerror=loss.val.e(i);
               bestiteration=i;
              best_nn=nn;
       else 
               minimal_validationerror=minimal_validationerror;
                best_nn=best_nn;
                 bestiteration=bestiteration;
     end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    else
        loss = nneval(nn, loss, train_x, train_y,S,total);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean crossentropy error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    
end
%%%%%%%%%%%%%%%%%
best.net=best_nn;
best.iteration=bestiteration;
best.val_loss=minimal_validationerror;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
end

