function parameters = TrainClassifierX(input, label)

parameters = {};
[row, col] = size(input);

%randomize order of dataset
idx = randperm(row);
input = input(idx,:);
label = label(idx,:);

%use num_training training examples
num_training = 4800;
if (row < 4800) num_training = row; end
input = input(1:num_training,:);
label = label(1:num_training,:);
%train_data = input(1:num_training,:);
%train_label = label(1:num_training,:);
%test_data = input(num_training:end,:);
%test_label = label(num_training:end,:);
[n,d] = size(input);

% K Cross Validation
totalK = 4;
frame = n/totalK-1;
x=1;
confuse = zeros(5,5);

%Hyperparameters
alphak = [0.5,2,5,15];
lambdak = [0.01,0.1,0.25,0.5];
Bk = [0.0075,0.01,0.02,0.03];
bs = [100,480,1200,4800];
hidden_layer = [5,10,25,50];
epochs = [50,100,250,500];


figure
hold on
for k=1:totalK;
    
    test_data = input(x:x+frame,:);
    test_label = label(x:x+frame,:);
    
    train_data = input;
    train_data(x:x+frame,:) = [];
    train_label = label;
    train_label(x:x+frame) = [];
    x=x+frame;
    
    
    %feed parameters to test and do training
    hyperparam = [alphak(k),lambdak(k),Bk(k),bs(k),hidden_layer(k),epochs(k)];
    [parameters,loss] = doTraining(train_data,train_label,hyperparam);
    parameter_list{k} = parameters;
    
    %train data
    loss_data{k} = loss(2:end);
    train_output = ClassifyX(train_data, parameters);
    train_correct = length(find(train_output == train_label));
    train_accuracy(k) = 100 * (train_correct/length(train_data));
    
    %validation data
    output = ClassifyX(test_data, parameters);
    correct = length(find(output == test_label));
    validation_accuracy(k) = 100 * (correct/length(test_data));
    
    %Plot loss for each epoch
    plot(loss_data{k}, 'LineWidth', 2);
    
    
end
xlabel('epoch')
ylabel('Cross Entropy Loss')  
title('Loss for each epoch')
legend('cross-val 1','cross-val 2','cross-val 3','cross-val 4')
hold off

%obtain parameters that maximize accuracy
max_val = find(validation_accuracy == max(validation_accuracy))
parameters = parameter_list{max_val};

%Plot train & test accuracy
kvals = [1:totalK];
hl = [5,10,25,50];
figure
hold on
plot(kvals,train_accuracy,'*')
plot(kvals,validation_accuracy,'o')
legend('train accuracy','validation accuracy');
xlabel('k cross val');
ylabel('accuracy (%)')
title('accuracy for number of epochs')
    


%% Nested function to actually do training on specific K cross-val data
    function [param,loss_data] = doTraining(inputk, labelk,hyperparam)
        inputk(:,end+1) = 1;
        [num_examples,dim] = size(inputk);
        
        %{
        %use following code to test and tune hyperparameters
        alpha = hyperparam(1);
        lambda = hyperparam(2);
        B = hyperparam(3);
        batch_size = hyperparam(4);
        h1 = hyperparam(5);
        %}
        
        %set optimal hyperparamter values
        alpha = 2;
        lambda = 0.25; 
        B = 0.0075;
        batch_size = 1200;
        %check that batch size is divisible by total examples
        if (mod(num_examples,batch_size) ~= 0)
            batch_size = num_examples; 
        end
        e = 250;
    
        
        %Neural Network wth hidden layer n=25
        h1 = 25;
        h2 = 5;

        Theta1 = rand(dim,h1);
        Theta2 = rand(h1+1,h2);

        Theta1_grad = zeros(size(Theta1));
        Theta2_grad = zeros(size(Theta2));
        epoch = 0;
         
        
        %train parameters for number of epochs
        while epoch < e
            if epoch == 0 CEloss = 0; end
            epoch = epoch+1;
            CEloss = 0;
            
            %Mini Batch Gradient Descent
            num_batches = num_examples/batch_size;
            for b=1:num_batches
                bloss = 0;
                next = (b-1)*batch_size+1;
                inputbatch = inputk(next:next+batch_size-1,:);
                labelbatch = labelk(next:next+batch_size-1,:);
        
            %Feedforward (vectorize)
            for t=1:batch_size
                a1 = inputbatch(t,:); %1x65
    
                z2 = a1 * Theta1; %1x25
                a2 = sigmoid(B, z2); %1x25
                a2(:,end+1) = 1; %1x26
    
                z3 = a2*Theta2; %1x5
                a3 = softmax(z3); %1x5
                h = a3;
    
                ynew = zeros(1,h2); %1x5
                ynew(labelbatch(t))=1;
        
                %Cross Entropy Loss
                bloss = bloss - dot(ynew,log(h));
    
                %Backpropagation
                S3 = h-ynew;
                z2 = [z2 1];
    
                S2 = S3.*softmaxGradient(z3);
                Theta2_grad = Theta2_grad + a2'*S2;
    
                S1 = (S2*Theta2').* sigmoidGradient(B,z2);
                S1 = S1(1:end-1);
                Theta1_grad = Theta1_grad + a1'*S1;
    
            end

            Theta2_grad = (1/batch_size)*Theta2_grad;
            Theta1_grad = (1/batch_size)*Theta1_grad;

            
            %update parameters
            Theta1 = Theta1 - alpha*Theta1_grad;
            Theta2 = Theta2 - alpha*Theta2_grad;

            %Loss per each batch
            %bloss = bloss/batch_size;
            regularization = (lambda/(2*batch_size))*(sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2)));
            CEloss = bloss + regularization;
            end
        %store loss data for each epoch
        loss_data(epoch+1) = CEloss;
        precision = abs(loss_data(epoch) - CEloss);
        end

        param{1} = Theta1;
        param{2} = Theta2;

    end


    %% Activation Functions
    function g = sigmoid(b,z)
        g = 1.0 ./ (1.0 + exp(-b*z));
    end

    function h = sigmoidGradient(b,z)
        h = zeros(size(z));
        h = sigmoid(b,z).*(1-sigmoid(b,z));
    end

    function g = softmax(z)
        g = exp(z)/ (sum(exp(z)));
    end

    function h = softmaxGradient(z)
        h = zeros(size(z));
        h = softmax(z).*(1-softmax(z));
    end

end