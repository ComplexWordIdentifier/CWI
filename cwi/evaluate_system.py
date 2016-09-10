from __future__ import division, print_function
import sys
import argparse

"""
	1) Introduction:
	This package contains the evaluation script for Task 11: Complex Word Identification of SemEval 2016. 2) Content:
	- README.txt: This file.
	- evaluate_system.py: System evaluation script in Python. 3) Running:
	The command line that runs the evaluation script is:
	
		python evaluate_system.py [-h] --gold GOLD --pred PRED
		
	If you use the "-h" option, you will get detailed instructions on the parameters required.
	The "--gold" parameter must be a dataset with gold-standard labels in the format provided by the task's organizers.
	The "--pred" parameter must be the file containing the predicted labels.
"""

def evaluateIdentifier(gold, pred):
	"""
	Performs an intrinsic evaluation of a Complex Word Identification approach.
	@param gold: A vector containing gold-standard labels.
	@param pred: A vector containing predicted labels.
	@return: Accuracy, Recall and F-1.
	"""
	
	#Initialize variables:
	accuracyc = 0
	accuracyt = 0
	recallc = 0
	recallt = 0
	
	#Calculate measures:
	for gold_label, predicted_label in zip(gold, pred):
		if gold_label==predicted_label:
			accuracyc += 1
			if gold_label==1:
				recallc += 1
		if gold_label==1:
			recallt += 1
		accuracyt += 1
	
	accuracy = accuracyc / accuracyt
	recall = recallc / recallt
	fmean = 0
	
	try:
		fmean = 2 * (accuracy * recall) / (accuracy + recall)
	except ZeroDivisionError:
		fmean = 0
	
	#Return measures:
	return accuracy, recall, fmean


if __name__=='__main__':
	#Parse arguments:
	description = 'Evaluation script for Task 11: Complex Word Identification.'
	description += ' The gold-standard file is a dataset with labels in the format provided by the task organizers.'
	description += ' The predicted labels file must contain one label 0 or 1 per line, and must have the same number of lines as the gold-standard.'
	epilog = 'Returns: Accuracy, Recall and F1.'
	parser=argparse.ArgumentParser(description=description, epilog=epilog)
	parser.add_argument('--gold', required=True, help='File containing dataset with gold-standard labels.')
	parser.add_argument('--pred', required=True, help='File containing predicted labels.')
	args = vars(parser.parse_args())
	#Retrieve labels:
	gold = [int(line.strip().split('\t')[3]) for line in open(args['gold'])]
	pred = [int(line.strip()) for line in open(args['pred'])]
	#Calculate scores:
	p, r, f = evaluateIdentifier(gold, pred)
	#Present scores:
	print('Accuracy: ' + str(p))
	print('Recall: ' + str(r))
	print('F1: ' + str(f))
