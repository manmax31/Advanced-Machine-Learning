READ ME

Project Contains 3 folders: 1) CRF 2) SVM 3) results
1) CRF contains *.py files for Q1a and Q2a and a folder TrainCRF which contains all files *.py needed for Q2
2) SVM contains *.py files for Q3 and Q4
3) results contains the desired results.

All files are independent of each other and therefore can be run as "python filename.py".
Before you can run the scripts, some below mentioned folders needs to exist beforehand.
Please change the variable "path" or directory location in the main functions of all scripts according to where your test.txt, train.txt etc. files are located.


Folder CRF contains:
1) Decoder.py : This script implements Max-Sum and Brute Force for Q 1c and outputs a file "results/decode_output.txt"
2) Gardient.py: This script calculates the log probability and its gradient and output a file "results/gradient.txt"
3) Folder TrainCRF contains scripts for Q 2:
   a) lib.py      : This script contains functions used by other scripts.
   b) learn.py    : This script learns w matrix and T matrix from the training data, and writes them to learnt_w.txt, learnt_t.txt, and solution.txt.
   c) test.py     : This script uses learnt w and T matrix to predict the letters on the test data, and writes the result to prediction.txt.
   d) gradient.py : This script calculates the log probability and its gradient and output a file "results/gradient.txt"
   e) inference.py: This script implements Max-Sum and Brute Force to decode and outputs a file "results/decode_output.txt"



Folder SVM contains:
1) Run svm-hmm.py
This script runs the binary executables for SVM-STRUCT i.e. svm_hmm_learn and svm_hmm_classify.
All the binary executables must be inside SVM/svm_hmm/. Make sure you create a folder "svm_hmm" and place the executables there.
It outputs a model file that gets saved in SVM/model/. Make sure the folder "model" exists before.
It also outputs the predicted labels. These get saved inside SVM/outtags/. Make sure the folder "outtags" exists before.

2) SVM-Struct_prediction.py
This script prints out the letter wise and word wise accuracy based on SVM-Struct. Value of C can be changed in the main() function.

3) SVM-MC_prediction.py
This script implements multiclass classification based on liblinearutil.py and outputs the letter and word wise accuracy.
Make sure liblinearutil.py is inside the folder SVM/  . Value of C can be changed in the main() function.

4) Tampering.py
This script implements Q4 i.e. rotates or translates the features and gives us the letter and word wise accuracy based on SVM-MC.
Value of C and no of lines to be tampered can be changed in the main() function.



Folder results contains:
1) decode_output.txt
This file contains the decoded output of Q1c.

2) gradient.txt
This file contains the gradient of the the likelihood function of Q 2a.

3) solution.txt
This function contains the learnt W and T matrix.

4) prediction.txt
This file contains the predicted letters.