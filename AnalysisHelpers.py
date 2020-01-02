"""
	Author: Michael Berk
	Date: Spring 2020
	Description:
		- customized visualization functions 
		- data subsetting functions
		- statistical analysis functions
"""

##################################################
##################################################
# Imports and Globals
##################################################
##################################################
# my files
from LoadAndClean import DATA

# general
import pandas as pd
import numpy as np

# data table visualization 
import webbrowser
from pivottablejs import pivot_ui #create js file for visualization
from tabulate import tabulate

# plotting
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# stats
from scipy.stats.mstats import normaltest
from scipy import stats
from itertools import combinations # tuple permutations
import statsmodels.stats.multitest as multi

##################################################
##################################################
# Helpers Class 
##################################################
##################################################
class helpers():
	def __init__(self):
		# change print options
		pd.set_option('display.max_columns', 200)
		pd.set_option('display.max_roows', 500)
	
	########################################
	# Visuzalizations 
	########################################

	def showDF(self,df,how='print'):
		"""Visualize pandas based on given how parameter."""

		if how == 'print':
			#pretty print string
			print(df.to_string())

		elif how == 'pivot':
			#render pivot table in chrome
			pivot_ui(df,outfile_path='x.html')
			webbrowser.open(SPATH+'x.html',new=2)

		elif how == 'tabulate':
			#print string with borders
			print(tabulate(df, headers='keys', tablefmt='psql'))

		else:
			print('invalid "how" in showDF()')

		print(df.shape)
	

	########################################
	# Data Manipulation 
	########################################

	def readData(self, fileType='pickle'):
		"""Read and return data according to fileType"""

		if fileType == 'pickle':
			return pd.read_pickle(DATA.DPATH+DATA.FILENAME)

		elif fileType == 'csv':
			return pd.read_csv(DATA.DPATH+DATA.FILENAME+'.csv')

		else:
			print('fileType in readData() type incorrect')
			return

	def getSubset(self, df, col, minSampleSize=0):
		"""Get subsets for all unqiue values in column based on col parameter.

		Param
			df: data frame (pd.DataFrame)
			col: column name to subset on (str)
			minSampleSize: minimum number of instances to be included as a subset

		Return
			unique column value and subsetted DF. Note that all vals is included as first value (dict)
		"""

		# print update
		print("Creating subsets based on: "+col)

		# prepare index for return 
		returnDict = {'ALL':df}

		# handle column specific subsets
		counts = df[col].value_counts()
		vals = counts[counts > minSampleSize].index.unique()
		
		# itereate and create boolean list 
		for v in vals:
			returnDict[v] = df[df[col] == v]

		# remove NA value
		del returnDict['NA']

		return returnDict

	########################################
	# Stats 
	########################################

	def oneVsAllVar(self, df, colsToInclude):
		"""Perform comparisons for variance between subsets and population.

		Param
			df: data frame to analyze (pd.DataFrame)
			subsetType: subset type keyword to be bassed to getSubset() (str)
		
		Return
			None	
		"""
		#TODO: NOT TESTED
		#create empty df
		d = {}
		for c in colsToInclude:
			d[c] = np.array([])

		#iterate through subsets
		for s in self.h.getSubset(subset):
			temp = self.df[self.df[subset] == s]

			#iterate through all columns
			for c in colsToInclude:
				#calculate general vals
				v = np.var(temp[c])
				corrSubset = np.corrcoef(temp[c])

				if analysisType == "one/all var":
					#get other variances and return
					totalVar = np.var(self.df[c])
					relativeVar = v/totalVar

					d[c] = np.append(d[c], relativeVar)

		return d


	def oneVsAllCorr(self, df, subsetType):
		"""Perform comparisons for correlations between subsets and population.

		Param
			df: data frame to analyze (pd.DataFrame)
			subsetType: subset type keyword to be bassed to getSubset() (str)
		
		Return
			None	
		"""

		# setup return dict
		returnDict = {}

		# calcualte correlation matrix for all variables
		temp = df[COMMON_COLUMNS]
		self.correlationMatrix(temp, 'No Subset')

		# get data subset
		for s in self.h.getSubset(subsetType):
			# get subset
			subsetDF = df[df[subsetType] == s][COMMON_COLUMNS]

			# get correlation matrix
			self.correlationMatrix(subsetDF, s+' Subset')

