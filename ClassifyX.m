function label = ClassifyX(input, parameters)

%% All implementation should be inside the function.

%Initialize label to random value
label = randi([1,5],1);

%Set sigmoid coefficient hyperparamter B
B = 0.0075;

%Obtain weights from trained parameters
Theta1 = parameters{1};
Theta2 = parameters{2};

a0 = input; %examples x 64
a0(:,end+1) = 1; %for bias term

z1 = a0 * Theta1;
a1 = sigmoid(B,z1); %examples x 25
a1(:,end+1) = 1;  

z2 = a1 * Theta2; %examples x 5
[row, col] = size(z2);
if (row == 1)
    a2 = softmax(z2);
    label = find(a2 == max(a2)); %1x5
else
    for i=1:length(z2)
        out = softmax(z2(i,:));
        label(i) = find(out == max(out));
    end
    label = label'; %examples x 5
end

%{
%% Classify using KNN
parameters = {train_data,train_label,k};
output = zeros(length(input(:,1),1);
for j=1:length(test_data)
    output(j) = ClassifyX_KNN(test_data(j,:),parameters);
end
correct = length(find(output == test_label));
test_accuracy(knn) = 100 * (correct/length(test_data))
%}

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

    function class = ClassifyX_KNN(input,parameters)
    k=parameters{3};
    data = parameters{1};
    output = parameters{2};
    [n,dims] = size(data);
    for z=1:n
        sumdist = 0;
        for j=1:dims
            sumdist = sumdist + ( (input(j) - data(z,j))^2);
        end
        dist(z) = sumdist^0.5;
    end
    ascending_dist = sort(dist);

    [Bo,I] = mink(dist,k);
    num_outputs = output(I);

    class=0;
    maxnum=0;
    for iter=1:5
        num = length(find(num_outputs == iter));
        if num > maxnum
            maxnum = num;
            class = iter;
        end
    end
end

end
