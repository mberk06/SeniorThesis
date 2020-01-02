"""
	Author: Michael Berk
	Date: Spring 2020
	Description:
		- read in data from csv file
		- reshape data and merge different data files
		- clean columns (remove whitespace, make capital, standardize factors)
		- test for data quality
"""

################################################################
################################################################
# Imports and Globals
################################################################
################################################################
import pandas as pd
import numpy as np

import sys
from Subsets import SUBSETS

# create globals module
DATA = sys.modules[__name__]

DATA.DPATH = '../Data/' # data path 
DATA.FILENAME = 'df1.0' # save file name

class data():
	def __init__(self):
		print("'data' class created")
		pass

	def readData(self, csv): 
		"""Read in files from csv or excel path and return all files in list."""

		# read file from path base don file type
		if csv:
			a = pd.read_csv(DATA.DPATH+'Belt.csv')
			b = pd.read_csv(DATA.DPATH+'Static Descriptors.csv')
			c = pd.read_csv(DATA.DPATH+'Non-Static Descriptors.csv')
			d = pd.read_csv(DATA.DPATH+'Substrate.csv')

		else:
			a = pd.read_excel(DATA.DPATH+'Belt.xlsx')
			b = pd.read_excel(DATA.DPATH+'Static Descriptors.xlsx')
			c = pd.read_excel(DATA.DPATH+'Non-Static Descriptors.xlsx')
			d = pd.read_excel(DATA.DPATH+'Substrate.xlsx')

		return [a,b,c,d]		

	def mergeData(self, data):
		"""Merge 4 data tables so each row represents a single dive (location+time are unique)

		Param
			data: belt DF, statitic desc DF, nonstatic desc DF, substrate DF (list)

		Return
			merged df (pd.DataFrame)
		"""

		# index tables 
		belt = data[0]
		static = data[1]
		nonstatic = data[2].drop_duplicates(['Reef ID','Date','Depth']) # drop duplicates
		substrate = data[3].rename(columns={'DATE':'Date','depth':'Depth'}) # recode col names

		#standardize organism codes
		belt['Organism Code'] = [o.strip().upper() for o in belt['Organism Code']]

		#sum all values for transects
		belt['Count'] = belt['S1']+belt['S2']+belt['S3']+belt['S4'] 
		#nonUnique = belt[belt.duplicated(['Reef ID','Date','Depth','Organism Code'])] #check if there are duplciates before pivot

		#pivot data
		pivotedBelt = belt.pivot_table(index=['Reef ID','Date','Depth'], columns='Organism Code', values='Count')

		#merge data
		df = static.merge(nonstatic, on='Reef ID', how='inner')	
		df = df.merge(pivotedBelt, on=['Reef ID','Date','Depth'], how='inner')

		return df
	
	def cleanData(self, df):
		"""Perform a variety of cleaning tasks and return clean df. Tasks involve reshaping data, recoding columns, and changing data types."""

		# remove Time of day work began, Time of day work ended (all the same value)
		df = df.drop(['Time of day work began', 'Time of day work ended'], axis=1)

		# trunc columns with more than 100 chars
		#df = df.applymap(lambda c: c[:100] if isinstance(c, str) else c)

		# make uppercase
		df = df.apply(lambda x: x.astype(str).str.upper())

		# strip whitespace if string
		df = df.applymap(lambda c: c.strip() if isinstance(c, str) else c)
	
		# remove errors
		df = df[df['Errors?'] != 'VERDADERO']
		
		# recode to NA
		df = df.replace([np.nan, 'nan', 'NAN', float('nan')], 'NA', regex=True)

		# variables to recode
		factorCols = ['Dynamite Fishing?','Sewage pollution']
		factorVals = ['NA','YES','NO','HIGH','MEDIUM','LOW','NONE','MODERATE','PRIOR',
					  'FALSO','VERDADERO','TRUE','FALSE']

		# iterate through vars
		for c in factorCols:
			df[c] = df[c].replace([x for x in df[c] if x not in factorVals], 'NA')

		# recode specific factors
		df = df.replace('MED','MEDIUM')
		
		# recode all blachings to percentages
		for c in ['Percent colonies bleached','Percent Bleaching','Percent of each colony']:
			df[c] = df[c].apply(lambda x: str(x).replace('<',''))
			df[c] = df[c].apply(lambda x: str(x).replace('>',''))
			df[c] = df[c].apply(lambda x: str(x).replace('%',''))

		# columns to convert to numeric
		colsToFloat = ['TRASH GENERAL','TRASH FISH NETS','CORAL DAMAGE OTHER','CORAL DAMAGE DYNAMITE',
					   'CORAL DAMAGE ANCHOR','Percent colonies bleached','Percent Bleaching',
					   'Percent of each colony','Latitude Seconds','Latitude Minutes','Latitude Degrees',
					   'Longitude Seconds','Longitude Minutes','Longitude Degrees']+SUBSETS.ALL_ORGANISMS
		for c in colsToFloat:
			df[c] = pd.to_numeric(df[c], errors='coerce') # convert NA to NaN

			# not for lat/lon
			if 'Latitude' not in c and 'Longitude' not in c:
				df[c] = df[c].apply(lambda x : x*100 if x < 1 and x != 0 else x) #change decimals to percentages
				df[c] = df[c].apply(lambda x : x/10 if x > 100 else x) #change decimals to percentages

		# recode lat, long
		df['Lat'] = df['Latitude Seconds'].truediv(3600) + df['Latitude Minutes'].truediv(60) + df['Latitude Degrees'] # combine cols
		df['Lon'] = df['Longitude Seconds'].truediv(3600) + df['Longitude Minutes'].truediv(60) + df['Longitude Degrees']

		df.loc[df['Latitude Cardinal Direction'] == 'S', 'Lat'] *= -1 # convert to negative according to cardinal direction
		df.loc[df['Longitude Cardinal Direction'] == 'W', 'Lon'] *= -1

		return df

	def save(self, df):
		"""Save df as csv and pickle file."""

		# save as csv
		f = open(DATA.DPATH+DATA.FILENAME+'.csv','w')
		f.write(df.to_csv(index=True))
		f.close()

		# save as pickle
		df.to_pickle(DATA.DPATH+DATA.FILENAME)

############################# testing ###########################
class test():
	def __init__(self):
		pass

	#Purpose: test data merge
	#Params: pandas data frame
	#Return: dictionary of test results
	def testDataMerge(self, unmerged, merged):
		#setup testing vars
		testResults = {}
		nonUniqueIDs = [str(merged['Reef ID'][i]) + str(merged['Date'][i]) + str(merged['Depth'][i]) for i in range(len(merged['Date']))]
		uniqueIDs = set(nonUniqueIDs)
		
		#Reef ID + TS is unique
		testResults['ReefID + Date + Depth is unique'] = 1 if len(uniqueIDs) == len(nonUniqueIDs) else 0

		#num rows == num unique ids
		testResults['num rows == num unqiue ids'] = 1 if len(uniqueIDs) == len(merged) else 0

		#columns with unique names
		testResults['columns have unique names'] = 1 if len(set(list(merged.columns.values))) == len(list(merged.columns.values)) else 0

		#check for NA/nan
		testResults['NAN/empty recoded to NA'] = 0 if 'nan' in merged.values or '' in merged.values else 1 

		#check for whitespace
		testResults['no whitespace'] = 1 if True not in merged.applymap(lambda c: c[0] == ' ' or c[-1] == ' ' if isinstance(c, str) else False).any().values else 0
		
		#print values
		for k,v in testResults.items():
			print(str(v) + ": " + k)

		return testResults


######################################################
######################################################
# Run
######################################################
######################################################
class runData():
	def __init__(self):
		self.runIt()

	def runIt(self):
		"""Create classes and call data merge/clean functions then run tests."""

		# create classes
		d = data()
		t = test()

		# read in data
		df = d.readData(csv=True)

		# merge data and run test
		df = d.mergeData(df)
		#t.testDataMerge(data,df)

		# clean data
		df = d.cleanData(df) 

		# save as csv and pickle
		d.save(df)

		# return df
		return df

