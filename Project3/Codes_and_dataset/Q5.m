%% Swap the R_matrix and weight_matrix
R_matrix_0_1 = weight_matrix;
weight_matrix_rating = R_matrix;


%%
max_L = 1288;
hit_rate_list = zeros(max_L,1,3);
false_alarm_rate_list = zeros(max_L,1,3);


for k_index = 1 : 3
%% Perform 10-fold cross validation
    index_permutation = randperm(length(ratings));
    predicted_R = zeros(max(user_ids), num_movies);
    for fold = 1 : 10
        R_matrix_0_1_train = R_matrix_0_1;
        weight_matrix_rating_train = weight_matrix_rating;

        for i = (10000*fold-9999) : (10000*fold)
            index = index_permutation(i);
            R_matrix_0_1_train(user_ids(index), movie_index(movie_ids(index))) = 0;
            weight_matrix_rating_train(user_ids(index), movie_index(movie_ids(index))) = 0;
        end

        option_struct = struct('iter', 50, 'dis', true);
        [U,V] = factorize(R_matrix_0_1_train, 100, weight_matrix_rating_train, option_struct);
        UV_matrix = weight_matrix_rating.*(U * V);

        for i = (10000*fold-9999) : (10000*fold)
            index = index_permutation(i);
            predicted_R(user_ids(index), movie_index(movie_ids(index))) = UV_matrix(user_ids(index), movie_index(movie_ids(index)));
        end

    end



    %% Calculate precision of the algorithm
    a = predicted_R;
    b = R_matrix;

    recommendation = zeros(max(user_ids), num_movies);

    % Find the top 5 recommended movie for each user
    for x = 1 : max(user_ids)
        movie_ratings = a(x,:);
        for y = 1 : num_movies
            if(movie_ratings(y) <= 3)
                movie_ratings(y) = 0;
            end
        end
        [max5, max5index] = sort(movie_ratings);
        max5index= fliplr(max5index);
        max5= fliplr(max5);
        for i = 1 : num_movies
            if(max5(i) <= 3)
                break;
            end
            recommendation(x, i) = max5index(i);
        end
    end


    % Calculate total recommendation precision for every user
    precision_list = zeros(max(user_ids), 1);
    for x = 1 : max(user_ids)
        hit = 0;
        count = 0;
        for y = 1 : 5
            if(recommendation(x,y) > 0)
                count=count+1;
                if(b(x,recommendation(x,y))>3)
                    hit=hit+1;
                end
            end
        end
        precision_list(x) = hit/count;
    end

    disp('Total precision is:')
    disp(mean(precision_list));

    % Count the actual number of movies liked or disliked by all users
    actual_like = 0;
    actual_dislike = 0;
    for x = 1 : max(user_ids)
        for y = 1 : num_movies
            if(b(x,y) > 0)
                if(b(x,y) > 3)
                    actual_like = actual_like+1;
                else
                    actual_dislike = actual_dislike +1;
                end
            end
        end

    end         

    % Calculate the hit rate and false-alarm rate
    for L = 1 : max_L
        true_positive_num = 0;
        false_positive_num = 0;
        for x = 1 : max(user_ids)
            for y = 1 : L
                if(recommendation(x,y) > 0)
                    if(b(x,recommendation(x,y))>3)
                        true_positive_num = true_positive_num +1;
                    else
                        false_positive_num = false_positive_num +1;
                    end
                end
            end

        end
        hit_rate_list(L,1,k_index) = true_positive_num/actual_like;
        false_alarm_rate_list(L,1,k_index) = false_positive_num/actual_dislike;
    end
end


%% Plot figure
figure;

plot(false_alarm_rate_list(:,1,1), hit_rate_list(:,1,1), 'r',false_alarm_rate_list(:,1,2), hit_rate_list(:,1,2), 'g',false_alarm_rate_list(:,1,3), hit_rate_list(:,1,3), 'b');
title('Hit Rate vs False Alarm Rate');
xlabel('False Alarm Rate');
ylabel('Hit Rate');
legend('k = 10', 'k = 50', 'k = 100')

figure;
plot(1:max_L, hit_rate_list(:,1,1), 'r',1:max_L, hit_rate_list(:,1,2), 'g',1:max_L, hit_rate_list(:,1,3), 'b');
title('Hit Rate vs Number of recommended movies');
xlabel('Number of recommended movies');
ylabel('Hit Rate');
legend('k = 10', 'k = 50', 'k = 100')



figure;
plot(1:max_L, false_alarm_rate_list(:,1,1), 'r',1:max_L, false_alarm_rate_list(:,1,2), 'g',1:max_L, false_alarm_rate_list(:,1,3), 'b');
title('False Alarm Rate vs Number of recommended movies');
xlabel('Number of recommended movies');
ylabel('False Alarm Rate');
legend('k = 10', 'k = 50', 'k = 100')




