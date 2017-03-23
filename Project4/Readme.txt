

############################## EE219 PROJECT 2 ##############################
904590047 Guanchu Ling
404592020 Yingchao Tang
304681050 Yang Guo

########################## CLASSIFICATION ANALYSIS ##########################

The following README contains the requirements and steps that are required to
execute codes in Project 4.

The structure of the folders have also be explained.


######################## Implementation Dependencies ########################

Please make sure these packages are correctly pre-installed before running
the codes
   a. nltk v3.2.1
   b. numpy v1.11.1
   c. matplotlib v1.5.3
   d. sklearn v0.18.1

This project is coded using Python 2.7.12
under environment Anaconda 4.2.0 (x86_64) / Spyder 3.0.0


################################ Instruction ################################

1. Download and install all required packages from their official websites.

2. Unzip the file to local directory, make sure your computer is connected
   to internet for the purpose of data downloading.

3. The code files are listed in detail below. Each file solves one question
   and can be run seperately.

   - Q1.py solves QUESTION(1). It turns the documents into numerical feature
   vectors and creates the TFxIDF representation.
   
   - Q2.py solves QUESTION(2). It applied K-means cluster algorithm(2-cluster) 
   on the raw TFxIDF matrix and print out the socres with confusion matrix.
   
   - Q3.py solves QUESTION(3). It reduced the dimension of TFxIDF matrix using
   three different methods to different optimized number of dimensions and present
   the evaluation results.
   
   - Q4.py solves QUESTION(4). It visualized the clustering results using 2-D
   plots of the data points with different color and dot-shape.
   
   - Q5-plot.py and Q5-visual.py solves QUESTION(6). It clustered the whole 
   dataset from 20 topics into 20 different clusters. Explored the optimized 
   number of dimension in Q5-plot.py then visualized the clustering results 
   in Q5-visual.py.
   
   - Q6.py solves QUESTION(6). It clustered the same dataset as Q5 but into
   6 clusteres as the 6 topic-class goes, and explored the optimized dimension
   reduction based on the plot of scores.

 

- Graphs will be displayed in console automatically.
- Output of each question is displayed in the console output.


############################# Folder Structure ##############################

- Codes
     - Q1.py
     - Q2.py
     - Q3.py
     - Q4.py
     - Q5-plot.py
     - Q5-visual.py
     - Q6.py

- Readme.txt

- Report.pdf
