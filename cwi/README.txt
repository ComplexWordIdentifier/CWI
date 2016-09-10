1) cwi_training.txt

The cwi_training.txt file is the original dataset for the SemEval 2016 Complex Word Identification task.
The test set will be distributed in the same format.
Each instance in the training dataset is distributed in the following format:

<sentence> <word> <index> <label>

All components are separated by a tabulation marker.
The <sentence> component is a sentence extracted from a certain source.
The <word> component is a word in <sentence>.
The <index> component is the position of <word> in the tokens of <sentence>.
The <label> component is a label that receives value 1 if the word has been judged complex by at least one annotator, and value 0 otherwise.

---------------------------------------------------------------------------------------------------------

2) cwi_training_allannotations.txt

The cwi_training_allannotations.txt file contains all annotations made over the dataset.
This file can be used as a complementary source of information
Each instance in the training dataset is distributed in the following format:

<sentence> <word> <index> <label_1> <label_2> ... <label_19> <label_20>

All components are separated by a tabulation marker.
The <sentence> component is a sentence extracted from a certain source.
The <word> component is a word in <sentence>.
The <index> component is the position of <word> in the tokens of <sentence>.
The <label_i> components are labels that receive value 1 if the ith annotator has judged the word to be complex, and value 0 otherwise.