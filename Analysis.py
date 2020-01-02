"""
	Author: Michael Berk
	Date: Spring 2020
	Description:
		- file where analyis is conducted on RC data by calling functions from other files

"""

# Load and clean data: python3 -c 'import LoadAndClean as lc; lc.runData()'

##################################
##################################
# References
##################################
##################################

# my files
from Subsets import SUBSETS
import AnalysisHelpers 

# create local example of class 
h = AnalysisHelpers.helpers()

##################################
##################################
# RUN
##################################
##################################
df = h.readData()
h.getSubset(df, col='Ocean')

