#This file will be used to read in and merge data files from Reef Check
#The desired output will have each row representing a single dive (group by dive site and TS)
#Date: 9/2019
#Author: Michael Berk

##################### imports and global vars ###################
import pandas as pd
import numpy as np
from statistics import mean

import webbrowser
from pivottablejs import pivot_ui #create js file for visualization
from tabulate import tabulate

PATH = '/Users/michaelberk/documents/Penn 2019-2020/Senior Thesis/Data/'
SPATH = 'file:///Users/michaelberk/documents/Penn 2019-2020/Senior Thesis/Scripts/'
FILENAME = 'uniqueReefs'

######################### helpers ###########################
class helpers():
	def __init__(self):
		print("Class 'helpers' created")

	#Purpose: visualize pandas DF
	#Params:df, how the data should be visualized
	#Return: NA
	def showDF(self,df,how='print'): 
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

	#Purpose: read files from xlsx path
	#Params: specify if read from csv or excel
	#Return: list of [belt, static descriptors, nonstatic descriptors]  
	def readData(self, csv): 
		print("Reading in data") 
		#read file from path
		if csv:
			a = pd.read_csv(PATH+'Belt.csv')
			b = pd.read_csv(PATH+'Static Descriptors.csv')
			c = pd.read_csv(PATH+'Non-Static Descriptors.csv')
		else:
			a = pd.read_excel(PATH+'Belt.xlsx')
			b = pd.read_excel(PATH+'Static Descriptors.xlsx')
			c = pd.read_excel(PATH+'Non-Static Descriptors.xlsx')

		return [a,b,c]		

	#Purpose: merge 3 data tables so each row represents a single dive at a single time
	#Params: belt DT, static descriptors DT, nonstatic descriptors DT
	#Return: merged DB
	def mergeData(self, data):
		print("Merging belt, static, and nonstatic DFs")
		#index tables and drop duplicate dives and reefs
		belt = data[0]
		static = data[1]
		nonstatic = data[2].drop_duplicates(['Reef ID','Date','Depth'])

		#standardize organism codes
		belt['Organism Code'] = [o.strip().upper() for o in belt['Organism Code']]
		belt['Organism Code']['HAEMULIDAE'] = 'GRUNTS'

		#sum all values for transects
		belt['Count'] = belt['S1']+belt['S2']+belt['S3']+belt['S4'] 
		#nonUnique = belt[belt.duplicated(['Reef ID','Date','Depth','Organism Code'])] #check if there are duplciates before pivot

		#pivot data
		pivotedBelt = belt.pivot_table(index=['Reef ID','Date','Depth'], columns='Organism Code', values='Count')

		#merge data
		df = static.merge(nonstatic, on='Reef ID', how='inner')	
		df = df.merge(pivotedBelt, on=['Reef ID','Date','Depth'], how='inner')

		#clean data
		df = self.cleanData(df)
	
		return df
	
	def cleanData(self, df):
		"""Perform a variety of cleaning tasks and return clean df. Tasks involve reshaping data, recoding columns, and changing data types."""

		#remove Time of day work began, Time of day work ended (all the same value)
		df = df.drop(['Time of day work began', 'Time of day work ended'], axis=1)

		#tunc columns with more than 100 chars
		#df = df.applymap(lambda c: c[:100] if isinstance(c, str) else c)

		#make uppercase
		df = df.apply(lambda x: x.astype(str).str.upper())

		#strip whitespace if string
		df = df.applymap(lambda c: c.strip() if isinstance(c, str) else c)
	
		#remove errors
		df = df[df['Errors?'] != 'VERDADERO']
		
		#recode to NA
		df = df.replace([np.nan, 'nan', 'NAN', float('nan')], 'NA', regex=True)

		#variables to recode
		factorCols = ['Dynamite Fishing?','Sewage pollution']
		factorVals = ['NA','YES','NO','HIGH','MEDIUM','LOW','NONE','MODERATE','PRIOR',
					  'FALSO','VERDADERO','TRUE','FALSE']

		#iterate through vars
		for c in factorCols:
			df[c] = df[c].replace([x for x in df[c] if x not in factorVals], 'NA')

		#recode specific factors
		df = df.replace('MED','MEDIUM')
		
		#recode all blachings to percentages
		for c in ['Percent colonies bleached','Percent Bleaching','Percent of each colony']:
			df[c] = df[c].apply(lambda x: str(x).replace('<',''))
			df[c] = df[c].apply(lambda x: str(x).replace('>',''))
			df[c] = df[c].apply(lambda x: str(x).replace('%',''))

		#columns to convert to numeric
		colsToFloat = ['TRASH GENERAL','TRASH FISH NETS','CORAL DAMAGE OTHER','CORAL DAMAGE DYNAMITE',
					   'CORAL DAMAGE ANCHOR','Percent colonies bleached','Percent Bleaching',
					   'Percent of each colony','Latitude Seconds','Latitude Minutes','Latitude Degrees',
					   'Longitude Seconds','Longitude Minutes','Longitude Degrees']+ALL_ORGANISMS
		for c in colsToFloat:
			df[c] = pd.to_numeric(df[c], errors='coerce') #convert NA to NaN

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

	def saveAsCSV(self, df):
		"""Save df as csv."""

		f = open(FILENAME+'.csv','w')
		f.write(df.to_csv(index=True))
		f.close()
	
	def saveAsPickle(self, df):
		"""Save df as pickle."""

		df.to_pickle(FILENAME)

############################# testing ###########################
class testing():
	def __init__(self):
			print("Class 'testing' created")

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


############################# run ###########################
class run():

	#Purpose: run file (to be called from TrendAnalysis.py
	#Params: NA 
	#Return: NA
	def runIt(self, howShowData=None):
		#create class instances
		h = helpers()
		t = testing()

		#read in data
		data = h.readData(csv=True)

		#merge data
		df = h.mergeData(data)

		#save as csv and pickle
		h.saveAsCSV(df)
		h.saveAsPickle(df)

		#visualize
		if howShowData is not None:
			h.showDF(df.iloc[1:1000,], how=howShowData)

		#run tests
		t.testDataMerge(data,df)

		#return df
		return df








