% This script uses the method in Question-1 to predict the user preference
% based on the t-threshold partition. Record the precision and recall of 10
% test folds and calculate the mean value as for different t. Finally,
% there should be 3 ROC curves each corresponding to one k value.

%% Initialize threshold array and two arrays to record the precision and recall.
threshold = linspace(1.0, 5.0, 20);
precision = zeros(size(threshold, 2), 3);
recall = zeros(size(threshold, 2), 3);
f1score = zeros(size(threshold, 2), 3);

%% Traverse the predicted_R matrix and and original R matrix to calculate the precision and recall.
for t = 1:size(threshold, 2)
    for k = 1:3
        precision(t, k) = (length(find((predicted_R(:,:,k) > threshold(1,t)) & R_matrix > 3)))/(length(find(predicted_R(:,:,k) > threshold(1,t))));
        recall(t, k) = (length(find((predicted_R(:,:,k) > threshold(1,t)) & R_matrix > 3)))/(length(find(R_matrix > 3)));
		f1score(t, k) = 2 * (precision(t,k) * recall(t,k))/(precision(t,k) + recall(t,k));
				
    end
end

%% Plot the results.
figure;
subplot(4,1,1)
plot(threshold(:), precision(:,1), 'r', threshold(:), precision(:,2), 'b',threshold(:), precision(:,3), 'g')
title('Precision vs threshold')
xlabel('Threshold')
ylabel('Precision')
legend('k = 10', 'k = 50', 'k = 100')

subplot(4,1,2)
plot(threshold(:), recall(:,1), 'r', threshold(:), recall(:,2), 'b',threshold(:), recall(:,3), 'g')
title('Recall vs threshold')
xlabel('Threshold')
ylabel('Recall')
legend('k = 10', 'k = 50', 'k = 100')

subplot(4,1,3)
plot(threshold(:), f1score(:,1), 'r', threshold(:), f1score(:,2), 'b',threshold(:), f1score(:,3), 'g')
title('F-1 score vs threshold')
xlabel('Threshold')
ylabel('F-1 score')
legend('k = 10', 'k = 50', 'k = 100')

subplot(4,1,4)
plot(recall(:,1), precision(:,1), 'r', recall(:,2), precision(:,2), 'b',recall(:,3), precision(:,3), 'g')
title('Recall vs precision(ROC)')
axis([0,1,0.55,1])
xlabel('Recall')
ylabel('Precision')
legend('k = 10', 'k = 50', 'k = 100')

