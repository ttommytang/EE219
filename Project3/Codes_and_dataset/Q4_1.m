%% Swap the R_matrix and weight_matrix
R_matrix_0_1 = weight_matrix;
weight_matrix_rating = R_matrix;


%% Calculate total least squared error for different cases
k_list = [10, 50, 100];
total_squared_error_list = [0, 0, 0];

option_struct = struct('iter', 200, 'dis', true);
for i = 1 : 3
    fprintf('    ---> Calculating k=%d...\n', k_list(i));
    
    [U,V] = factorize(R_matrix_0_1,k_list(i),weight_matrix_rating,option_struct);
    UV_matrix = weight_matrix_rating .* (U * V);
    error_matrix = R_matrix - UV_matrix;
    total_squared_error_list(i) = sum(sum(error_matrix.^2));
end

%% Print results
for i = 1 : 3
    fprintf('The total least squared error for k=');
    fprintf(num2str(k_list(i)));
    fprintf(' is: ');
    fprintf(num2str(total_squared_error_list(i)));
    fprintf('\n');
end



