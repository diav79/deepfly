import numpy as np
import sys
import os
import re
# import getBoundingBoxes
import random


# Seed the random number generator (for repeatability)
np.random.seed(1234)

if len(sys.argv) != 3:
	trainFileName = 'train_list.txt'
	outFileName = 'trainingData.txt'
else:
	trainFileName = sys.argv[1]
	outFileName = sys.argv[2]

# Open the file whose lines are to be shuffled
trainFile = open(trainFileName, 'r')
# Read lines from the file
lines = trainFile.readlines()
# Create a random permutation of the lines
newLines = np.random.permutation(lines)
# Open the output file to write the permuted lines
outFile = open(outFileName, 'w')
# Write the permuted lines to the output file
for line in newLines:
	outFile.write(line)
outFile.close()