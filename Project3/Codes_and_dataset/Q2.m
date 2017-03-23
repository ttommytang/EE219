index_permutation = randperm(length(ratings));
average_error_for_every_entry = zeros(3, 10);
predicted_R = zeros(max(user_ids), num_movies, 3);

%% 10-fold cross validation
for fold = 1 : 10
    fprintf('Calculating the ');
    fprintf(num2str(fold));
    fprintf('th fold\n')
    
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
    option_struct = struct('iter', 30, 'dis', false);
    for i = 1 : 3
        fprintf('   --->Calculating k=%d\n', k_list(i));
        
        [U,V] = factorize(R_matrix_train, k_list(i), weight_matrix_train, option_struct);
        UV_matrix = U * V;
        error_sum = 0;
        for j = (10000*fold-9999) : (10000*fold)
            index = index_permutation(j);
            actual_value = R_matrix(user_ids(index), movie_index(movie_ids(index)));
            predicted_value = UV_matrix(user_ids(index), movie_index(movie_ids(index)));
            error_sum = error_sum + abs(actual_value - predicted_value);
            predicted_R(user_ids(index), movie_index(movie_ids(index)), i) = predicted_value;
        end
        average_error_for_every_entry(i, fold) = error_sum/10000;
    end
end

%% Print results
for i = 1 : 3
    fprintf('\n\nError information (k = %d):\n', k_list(i));
    fprintf('    Average error for each fold:\n');
    for j = 1 : 10
        fprintf('        Error for fold-%d : %d\n', j, average_error_for_every_entry(i, j));
    end
    fprintf('\n    Average value for average absolute error : %f\n', mean(average_error_for_every_entry(i, :)));
    fprintf('    Highest value for average absolute error : %f\n', max(average_error_for_every_entry(i, :)));
    fprintf('    Lowest value for average absolute error : %f\n', min(average_error_for_every_entry(i, :)));
end









