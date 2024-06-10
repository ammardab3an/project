[alphabets, result] = prprob;
% Create the backpropagation network
net = newff(minmax(alphabets), [10 26], {'logsig', 'logsig'}, 'trainlm');
net.trainParam.epochs = 800;
net.trainParam.goal = 0.001;
net.trainparam.lr =0.3;
% Training the neural network
[net,info, tearn] = train(net, alphabets, result);
% test to M
M = alphabets(:,13);
 figure, plotchar(M);
output = sim(net,M);
output = compet(output)
answer = find(compet(output) == 1); % find the index (out of 26) of network output
%figure, plotchar(alphabets(:, answer));
noisM = alphabets(:,13)+randn(35,1)*0.2;
%figure; plotchar(noisM);
output2 = sim(net, noisM);
answer2 = find(compet(output2) == 1);
%figure; plotchar(alphabets(:,answer2));