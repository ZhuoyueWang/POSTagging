# POSTAgging
##  LING 406 MP3
### Zhuoyue Wang, zhuoyue2

Files:

Zhuoyue_Wang_MP3_Report.pdf: The file which contains the performance difference table for both machine learning models and my answers to question 1-4.
tagging.py: the program to run both machine learning models in NLTK.


How to run the program:

Be sure in the working directory which contains the file "pos-eng-5000.data.csv" and type "python3 tagging.py" in the terminal. The terminal will print out the accuracy for both models.


Comments:

It is noticeable that the accuracy made by both models in my program is pretty close (both are around 60%), which is quite different from what I got from WEKA. I am not sure whether the difference is due to different evaluation methods on the model. In this program, I simply use train-test (80% vs. 20%) split to divide the data set instead of the cross-validation. It might create an inaccurate result.
