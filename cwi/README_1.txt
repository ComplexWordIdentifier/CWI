1) cwi_testing_annotated.txt

The cwi_testing_annotated.txt file is the annotated version of the original testset for the SemEval 2016 Complex Word Identification task.
Each instance in the annotated testset is distributed in the following format:

<sentence> <word> <index> <label>

All components are separated by a tabulation marker.
The <sentence> component is a sentence extracted from a certain source.
The <word> component is a word in <sentence>.
The <index> component is the position of <word> in the tokens of <sentence>.
The <label> component is a label that receives value 1 if the word has been judged complex by the annotator, and value 0 otherwise.