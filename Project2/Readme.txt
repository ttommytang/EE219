

############################## EE219 PROJECT 2 ##############################
904590047 Guanchu Ling
404592020 Yingchao Tang
304681050 Yang Guo

########################## CLASSIFICATION ANALYSIS ##########################

The following README contains the requirements and steps that are required to
execute codes in Project 2.

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

     - a.py solves QUESTION(a). It plots the histograms of the documents per
            topic and counts the number of documents in the two groups.

     - b.py solves QUESTION(b). It turns the documents into numerical feature
            vectors and creates the TFxIDF representation.

     - c.py solves QUESTION(c). It finds the 10 most significant terms in the
            four required classes.

     - d.py solves QUESTION(d). It applies LSI technique and maps the
            documents to a 50-dimensional vector.

     - e.py solves QUESTION(e). It uses linear SVM to classify the dataset.

     - f.py solves QUESTION(f). It uses soft margin SVM with different gamma
            value to classify the dataset and finds the optimal gamma.

     - g.py solves QUESTION(g). It uses multinomial Naive Bayes Classifier
            to classify the dataset.

     - h.py solves QUESTION(h). It uses Logistic Regression Classifier to
            classify the dataset.

     - i.py solves QUESTION(i). It adds a regularization term to the Logistic
            Regression Classifier and uses l1 and l2 norm regularization
            respectively to do the same task as in QUESTION(h).

     - multiclass.py solves MULTICLASS CLASSIFICATION. It performs Naive
            Bayes classification and multicalss SVM classification with both
            OneVsOne and OneVsRest strategy.


- Graphs will be displayed in console automatically.
- Output of each question is displayed in the console output.


############################# Folder Structure ##############################

- Codes
     - a.py
     - b.py
     - c.py
     - d.py
     - e.py
     - f.py
     - g.py
     - h.py
     - i.py
     - multiclass.py

- Readme.txt

- Report.pdf
