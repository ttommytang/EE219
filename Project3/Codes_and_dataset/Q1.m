%% Clear existing data and variables
clear;
clc;

%% Load data from files
user_ratings = importdata('ratings.csv');
user_ids = user_ratings.data(:,1);
movie_ids = user_ratings.data(:,2);
ratings = user_ratings.data(:,3);
%{
%% Load data from files
user_ratings = importdata('u.data');
user_ids = user_ratings(:,1);
movie_ids = user_ratings(:,2);
ratings = user_ratings(:,3);
%}
%% Build movie_ID and movie_index correspondence
movie_index = 1 : max(movie_ids);
for i = 1 : max(movie_ids)
    movie_index(i) = -1;
end

%% Count the number of movies and set values of movie_index
num_movies = 0;
for i = 1 : length(movie_ids)
    if(movie_index(movie_ids(i)) == -1)
        num_movies = num_movies + 1;
        movie_index(movie_ids(i)) = num_movies;
    end
end

%% Create R matrix and weight matrix
fprintf('Start creating matrix R and weight matrix...\n\n');
weight_matrix = zeros(max(user_ids), num_movies);
R_matrix = zeros(max(user_ids), num_movies);
for i = 1 : length(ratings)
    R_matrix(user_ids(i), movie_index(movie_ids(i))) = ratings(i);
    weight_matrix(user_ids(i), movie_index(movie_ids(i))) = 1;
end
fprintf('Matrix R and weight matrix created successfully\n\n');




%% Calculate total least squared error for different cases
k_list = [10, 50, 100];
total_squared_error_list = [0, 0, 0];

option_struct = struct('iter', 500, 'dis', true);
for i = 1 : 3
    fprintf('    ---> Calculating k=%d...\n', k_list(i));
    
    [U,V] = factorize(R_matrix,k_list(i),weight_matrix,option_struct);
    UV_matrix = weight_matrix .* (U * V);
    error_matrix = R_matrix - UV_matrix;
    total_squared_error_list(i) = sum(sum(error_matrix.^2));
end

%% Print results
for i = 1 : 3
    fprintf('The total squared error for k=');
    fprintf(num2str(k_list(i)));
    fprintf(' is: ');
    fprintf(num2str(total_squared_error_list(i)));
    fprintf('\n');
end



