%% Initialize result matrices
index_permutation = randperm(length(ratings));
average_error_for_every_entry = zeros(3, 3, 10);
predicted_R = zeros(max(user_ids), num_movies, 3, 3);
lambda_list = [0.01, 0.1, 1];

%% 10-fold cross validation
for fold = 1 : 10
    fprintf('Calculating the %dth fold\n', fold);
    
    weight_matrix_train = zeros(max(user_ids), num_movies);
    R_matrix_train = zeros(max(user_ids), num_movies);
    
    
    % Fill in all data
    for i = 1 : length(ratings)
        index = index_permutation(i);
        R_matrix_train(user_ids(index), movie_index(movie_ids(index))) = ratings(index);
        weight_matrix_train(user_ids(index), movie_index(movie_ids(index))) = 1;
    end
    
    % Disable the data used for testing
    for i = (10000*fold-9999) : (10000*fold)
        index = index_permutation(i);
        R_matrix_train(user_ids(index), movie_index(movie_ids(index))) = 0;
        weight_matrix_train(user_ids(index), movie_index(movie_ids(index))) = 0;
    end
    
    % For different k values, calculate average error
    option_struct = struct('iter', 10, 'dis', false);
    for i = 1 : 3
        for lambda_index = 1 : 3
            lambda = lambda_list(lambda_index);
            fprintf('   --->Calculating k=%d & lambda=%f\n', k_list(i), lambda);

            [U,V] = factorize_regularized(R_matrix_train, k_list(i), weight_matrix_train, lambda,option_struct);
            UV_matrix = U * V;
            error_sum = 0;
            for j = (10000*fold-9999) : (10000*fold)
                index = index_permutation(j);
                actual_value = R_matrix(user_ids(index), movie_index(movie_ids(index)));
                predicted_value = UV_matrix(user_ids(index), movie_index(movie_ids(index)));
                error_sum = error_sum + abs(actual_value - predicted_value);
                predicted_R(user_ids(index), movie_index(movie_ids(index)), i, lambda_index) = predicted_value;
            end
            average_error_for_every_entry(i, lambda_index, fold) = error_sum/10000;
        end
    end
end

%% Print results
for i = 1 : 3
    for lambda_index = 1 : 3
        lambda = lambda_list(lambda_index);
        fprintf('\n\nError information (k = %d, lambda = %.2f):\n', k_list(i), lambda);
        fprintf('    Average error for each fold:\n');
        for j = 1 : 10
            fprintf('        Error for fold-%d : %f\n', j, average_error_for_every_entry(i, lambda_index, j));
        end
        fprintf('\n    Average value for average absolute error : %f\n', mean(average_error_for_every_entry(i, lambda_index, :)));
        fprintf('    Highest value for average absolute error : %f\n', max(average_error_for_every_entry(i, lambda_index, :)));
        fprintf('    Lowest value for average absolute error : %f\n', min(average_error_for_every_entry(i, lambda_index, :)));
    end
end






%% Initialize threshold array and two arrays to record the precision and recall.
threshold = linspace(1.0, 5.0, 20);
precision = zeros(size(threshold, 2), 3, 3);
recall = zeros(size(threshold, 2), 3, 3);
f1score = zeros(size(threshold, 2), 3, 3);

for lambda_index = 1 : 3
    %% Traverse the predicted_R matrix and and original R matrix to calculate the precision and recall.
    for t = 1:size(threshold, 2)
        for k = 1:3
            precision(t, k, lambda_index) = (length(find((predicted_R(:,:,k, lambda_index) > threshold(1,t)) & R_matrix > 3)))/(length(find(predicted_R(:,:,k,lambda_index) > threshold(1,t))));
            recall(t, k, lambda_index) = (length(find((predicted_R(:,:,k,lambda_index) > threshold(1,t)) & R_matrix > 3)))/(length(find(R_matrix > 3)));
			f1score(t, k, lambda_index) = 2 * (precision(t,k,lambda_index) * recall(t,k,lambda_index))/(precision(t,k,lambda_index) + recall(t,k,lambda_index));
					
        end
    end

    %% Plot the results.
    figure;
    subplot(4,1,1)
    plot(threshold(:), precision(:,1,lambda_index), 'r', threshold(:), precision(:,2,lambda_index), 'b',threshold(:), precision(:,3,lambda_index), 'g')
    title('Precision vs threshold')
    xlabel('Threshold')
    ylabel('Precision')
    legend('k = 10', 'k = 50', 'k = 100')

    subplot(4,1,2)
    plot(threshold(:), recall(:,1,lambda_index), 'r', threshold(:), recall(:,2,lambda_index), 'b',threshold(:), recall(:,3,lambda_index), 'g')
    title('Recall vs threshold')
    xlabel('Threshold')
    ylabel('Recall')
    legend('k = 10', 'k = 50', 'k = 100')
	
	subplot(4,1,3)
	plot(threshold(:), f1score(:,1,lambda_index), 'r', threshold(:), f1score(:,2,lambda_index), 'b',threshold(:), f1score(:,3,lambda_index), 'g')
	title('F-1 score vs threshold')
	xlabel('Threshold')
	ylabel('F-1 score')
	legend('k = 10', 'k = 50', 'k = 100')

    subplot(4,1,4)
    plot(recall(:,1,lambda_index), precision(:,1,lambda_index), 'r', recall(:,2,lambda_index), precision(:,2,lambda_index), 'b',recall(:,3,lambda_index), precision(:,3,lambda_index), 'g')
    title('Recall vs precision(ROC)')
    axis([0,1,0.55,1])
    xlabel('Recall')
    ylabel('Precision')
    legend('k = 10', 'k = 50', 'k = 100')

end





