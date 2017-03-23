

############################## EE219 PROJECT 3 ##############################
904590047 Guanchu Ling
404592020 Yingchao Tang
304681050 Yang Guo

########################## COLLABORATIVE FILTERING ##########################

The following README contains the requirements and steps that are required to
execute codes in Project 3.

The structure of the folders have also been explained.


######################## Implementation Dependencies ########################

The codes in this project makes use of "wnmfrule" function in Matrix
Factorization Toolbox in MatLab.

This project is coded under environment MatLab R2014b.


################################ Instruction ################################

1. Put all codes and the dataset "ratings.csv" in the same folder.

2. Unzip the file to local directory, put the dataset "ratings.csv" and all
   MatLab codes in the same folder.

3. The code files are listed in detail below. Each code solves one question
   and depends on previous results.

   Please DO NOT close or re-launch MatLab before finishing running all the
   codes.

   Please DO NOT clear variables in "Workspace" after each code is finished.
  
   PLEASE RUN THE CODES IN THE REQUIRED ORDER BELOW:
    

     (1) Q1.m solves QUESTION(1). It calculates the total least square error
              for different k values.

     (2) Q2.m solves QUESTION(2). It performs 10-fold cross validation on the
              dataset and calculates the average absolute error.

     (3) Q3.m solves QUESTION(3). It plots the ROC curves as required.

     (4) Q4_1.m solves the first part of QUESTION(4). It swaps the weight
              matrix and the R matrix, then calculate the total squared error
              with different parameters.

     (5) Q4_2.m solves the second part of QUESTION(4). It added a
              regularization term to the cost function, then calculate the
              error and plot the ROC curve.

     (6) Q5.m solves QUESTION(5). It builds the recommendation system and
              tests it with the dataset. Precision, hit rate and false alarm
              rate are calculated.

          
- Graphs will be displayed automatically.
- Output of each question is displayed in the Command Window.


############################# Folder Structure ##############################

- Codes_and_dataset
     - Q1.m
     - Q2.m
     - Q3.m
     - Q4_1.m
     - Q4_2.m
     - Q5.m
     - factorize.m
     - factorize_regularized.m
     - mergeOption.m
     - matrixNorm.m
     - ratings.csv

- Readme.txt

- Report.pdf
