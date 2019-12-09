#This file will be used to develop conclusions about the trends of bioinidcators species on coral reefs over time
#The desired output will be visualizations and charts that provide insight into these trends 
#Date: 11/2019
#Author: Michael Berk

##################### imports and global vars ###################
import DataMerge as dm 
import pandas as pd
import numpy as np

import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

from scipy.stats.mstats import normaltest 
from scipy import stats
from itertools import combinations # tuple permutations
import statsmodels.stats.multitest as multi

######################### variables #######################
#types of columns
DISEASES = ['ASPERGILLOSIS','BLACK BAND','WHITE BAND','WHITE PLAGUE']
PRODUCERS = ['GORGONIAN','SEA FAN']
CORAL_COVER = ['BLEACHING (% OF COLONY)','BLEACHING (% OF POPULATION)']

MISC = ['LOBSTER','MANTA','EDIBLE SEA CUCUMBER']
PRIMARY_CONSUMERS = ['ARABIAN BUTTERFLYFISH','BANDED CORAL SHRIMP','BLACK URCHIN','BROOMTAIL WRASSE',
					 'BUMPHEAD PARROT','BUTTERFLYFISH','COTS','COWRIES','DARK BUTTERFLYFISH','DIADEMA',
					 'FLAMINGO TONGUE','GIANT CLAM 10-20 CM',
                     'GIANT CLAM 20-30 CM','GIANT CLAM 30-40 CM','GIANT CLAM 40-50 CM',
                     'GIANT CLAM <10 CM','GIANT CLAM >50 CM','GIANT CLAM TOTAL','ORANGE SPINE UNICORNFISH',
					 'PARROTFISH','PENCIL URCHIN','SHORT SPINE URCHIN','SPIDER CRAB','TRIPNEUSTES',
					 'TURTLES','YELLOW TANG','YELLOWTAIL TANG']
SECONDARY_CONSUMERS = ['ARABIAN BUTTERFLYFISH','BANDED CORAL SHRIMP','BLACK SPOTTED GRUNT','BLUELINE SNAPPER',
					   'BROOMTAIL WRASSE','BUTTERFLYFISH','DARK BUTTERFLYFISH','EDIBLE SEA CUCUMBER','GOATFISH',
					   'GREY GRUNT','GRUNTS','HELMET CONCH','HUMPHEAD WRASSE','KING ANGELFISH',
					   'LONGFIN BANNERFISH','KING ANGELFISH','MEXICAN HOGFISH','QUEEN CONCH','SPIDER CRAB',
					   'SPOTTED GRUNT','TRITON','TURTLES','YELLOW GOATFISH','YELLOWBAR ANGELFISH','BLUE SEA STAR']
TERCIARY_CONSUMERS = ['BARRAMUNDI COD','GIANT HAWKFISH','GROUPER 30-40 CM','JACKS','LIONFISH',
					  'NASSAU GROUPER 30-40 CM','ORANGE SPOTTED GROUPER 30-40 CM','PEACOCK GROUPER 30-40 CM']
QUARTERNARY_CONSUMERS = ['BARRACUDA','GROUPER 40-50 CM','GROUPER 50-60 CM','GROUPER >60 CM','GROUPER TOTAL',
						 'MORAY EEL','NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
						 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','NASSAU GROUPER 30-40 CM',
						 'NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
						 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','ORANGE SPOTTED GROUPER 40-50 CM',
					     'ORANGE SPOTTED GROUPER 50-60 CM','ORANGE SPOTTED GROUPER >60 CM',
					     'ORANGE SPOTTED GROUPER TOTAL','PEACOCK GROUPER 40-50 CM',
					     'PEACOCK GROUPER 50-60 CM','PEACOCK GROUPER >60 CM','PEACOCK GROUPER TOTAL',
						 'SHARKS']

ALL_ORGANISMS = ['ARABIAN BUTTERFLYFISH','ASPERGILLOSIS','BANDED CORAL SHRIMP','BARRACUDA',
				 'BARRAMUNDI COD','BLACK BAND','BLACK BAND','BLACK SPOTTED GRUNT',
				 'BLACK URCHIN','BLEACHING (% OF COLONY)','BLEACHING (% OF POPULATION)',
				 'BLUE SEA STAR','BLUELINE SNAPPER','BROOMTAIL WRASSE',
				 'BUMPHEAD PARROT','BUTTERFLYFISH','COTS',
				 'COWRIES','DARK BUTTERFLYFISH','DARK BUTTERFLYFISH','DIADEMA',
				 'EDIBLE SEA CUCUMBER','FLAMINGO TONGUE','GIANT CLAM 10-20 CM',
				 'GIANT CLAM 20-30 CM','GIANT CLAM 30-40 CM','GIANT CLAM 40-50 CM',
				 'GIANT CLAM <10 CM','GIANT CLAM >50 CM','GIANT CLAM TOTAL',
				 'GIANT HAWKFISH','GOATFISH','GORGONIAN','GREY GRUNT','GROUPER 30-40 CM',
				 'GROUPER 40-50 CM','GROUPER 50-60 CM','GROUPER >60 CM','GROUPER TOTAL',
				 'GRUNTS','HAEMULIDAE','HELMET CONCH','HUMPHEAD WRASSE',
				 'JACKS','KING ANGELFISH','LIONFISH','LOBSTER','LONGFIN BANNERFISH',
				 'MANTAS','MEXICAN HOGFISH','MORAY EEL','NASSAU GROUPER 30-40 CM',
				 'NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
				 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','NASSAU GROUPER 30-40 CM',
				 'NASSAU GROUPER 40-50 CM','NASSAU GROUPER 50-60 CM',
				 'NASSAU GROUPER >60 CM','NASSAU GROUPER TOTAL','ORANGE SPINE UNICORNFISH',
				 'ORANGE SPOTTED GROUPER 30-40 CM','ORANGE SPOTTED GROUPER 40-50 CM',
				 'ORANGE SPOTTED GROUPER 50-60 CM','ORANGE SPOTTED GROUPER >60 CM',
				 'ORANGE SPOTTED GROUPER TOTAL','PARROTFISH',
				 'PEACOCK GROUPER 30-40 CM','PEACOCK GROUPER 40-50 CM',
				 'PEACOCK GROUPER 50-60 CM','PEACOCK GROUPER >60 CM',
				 'PEACOCK GROUPER TOTAL','PENCIL URCHIN','QUEEN CONCH','SEA FAN','SHARKS',
				 'SHORT SPINE URCHIN','SLATE PENCIL URCHIN','SNAPPER',
				 'SPIDER CRAB','SPOTTED GRUNT','TRIPNEUSTES','TRITON','TROCHUS','TURTLES',
				 'WHITE BAND','WHITE PLAGUE','WHITE BAND','YELLOW GOATFISH','YELLOW TANG',
				 'YELLOWBAR ANGELFISH','YELLOWTAIL TANG']

NONSTATIC_DESCRIPTORS = ['Siltation','Dynamite Fishing?','Poison Fishing?',
						 'Aquarium fish collection','Harvest of inverts for food',
						 'Harvest of inverts for curio','Tourist diving/snorkeling',
						 'Sewage pollution','Industrial pollution','Commercial fishing',
						 'Live food fishing','Artisinal/recreational','Other forms of fishing?',
						 'Other Fishing','Yachts','Level of other impacts?','Other impacts?',
						 'Is site protected?','Is protection enforced?','Level of poaching?',
						 'Spearfishing?','Commercial fishing?','Recreational fishing?',
						 'Invertebrate/shell collection?','Anchoring?','Diving?','Other (specify)',
						 'Nature of protection?','Other site description comments?',
						 'Comments from organism sheet','Grouper Size','Percent Bleaching',
						 'Percent colonies bleached','Percent of each colony','Suspected Disease?',
						 'CORAL DAMAGE ANCHOR','CORAL DAMAGE DYNAMITE','CORAL DAMAGE OTHER',
						 'TRASH FISH NETS','TRASH GENERAL','Rare Animals?']
DIVE_DESCRIPTORS = ['Depth','Time of day work began','Time of day work ended','Weather'
 					'Air Temp','Water temp at surface','Water temp at 5m','Water temp at 10m',
					'Approx popn size (x1000)','Horizontal Visibility in water',
					'Best Reef Area?','Why was this site selected?','Sheltered or exposed?',
					'Any major storms in last years?','When storms?','Overall anthro impact?',
					'What kind of impacts?']

COMMON_COLUMNS = ['TRASH GENERAL','GROUPER TOTAL','SNAPPER','PENCIL URCHIN','PARROTFISH',
				  'MORAY EEL','LOBSTER','CORAL DAMAGE OTHER','BUTTERFLYFISH']

######################### cleaning class ###########################
class cleaning():
	def __init__(self):
		print("Class 'cleaning' created")

		#setup for printing
		pd.set_option('display.max_rows', 500)
		pd.set_option('display.max_columns', 500)

    #Purpose: read in data from dm
    #Params: NA 
    #Return: NA 
	def readData(self):
		df = dm.run().runIt(howShowData=None)
		return self.clean(df)
	
    #Purpose: do extra cleaning (recode factors, remove outliers, etc.)
    #Params: df 
    #Return: clean df 
	def clean(self, df):
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
					   'Percent of each colony']+ALL_ORGANISMS
		for c in colsToFloat:
			df[c] = pd.to_numeric(df[c], errors='coerce') #convert NA to NaN
			df[c] = df[c].apply(lambda x : x*100 if x < 1 and x != 0 else x) #change decimals to percentages
			df[c] = df[c].apply(lambda x : x/10 if x > 100 else x) #change decimals to percentages

		return df

######################### helpers class ###########################
#notes: df is global and will not be changed
class helpers():
	def __init__(self, df):
		print("Class 'helpers' created")
		self.df = df

    #Purpose: get subsets for columns
    #Params: column name 
    #Return: unique values to subset df
	def getSubset(self, col):
		if col == 'Latitude':
			print(self.df[col]) # not done 
		elif col == "Ocean":
			vals = self.df[col].unique()
			return vals[vals != 'NA']
		elif col == 'Country':
			#return all countries with more than 5 dives
			counts = self.df[col].value_counts()
			return counts[counts > 5].index.unique()

    #Purpose: subset and aggregate data 
    #Params: subset by region
    #Return: self.df that is aggregated on day and given params 
	def aggDF(self, region=None):
		#gropu by value
		temp = self.df.groupby(['Date']).mean()
		print(temp)
		return temp

    #Purpose: print unique values in column 
    #Params: column(s) 
    #Return: NA 
	def printUniqueCol(self, c):
		#check if iterable
		if type(c) == list:
			if type(c[0]) == str:
				for i in c:
					print(str(i)+": "+str(self.df[i].unique().tolist()))
			elif type(c[0]) == int:
				for i in c:
					print(str(i)+": "+str(self.df.iloc[:,i].unique().tolist()))
			else:
				print("Incorrect type 'printUniqueCol'")
		#if not iterable
		else:
			if type(c) == str:
				print(str(c)+": "+str(self.df[c].unique().tolist()))
			elif type(c) == int:
				print(str(c)+": "+str(self.df.iloc[:,c].unique().tolist()))
			else:
				print("Incorrect type 'printUniqueCol'")

######################### analysis class ###########################
#notes: df is global and will not be changed
class analysis():
	def __init__(self, df):
		print("Class 'analysis' created")
		self.df = df
		self.h = helpers(self.df)

	# Purpose: test if orgnaisms are bivariate normal
	# Params: 
	# Return: 
	############### INCOMPLETE ####################
	def bivariateNormalTest(self, df):
		# get the data
		n = len(COMMON_COLUMNS)


		# get columns
		a = np.array(df[COMMON_COLUMNS[2]])
		b = np.array(df[COMMON_COLUMNS[3]])
		print(a)
		print(b)
		temp = np.append(a,b)
		print(temp.shape)
		print(normaltest(temp))
		
		# set up return matrix
		mat = np.empty([n,n])

		# iterate through matrix
		for i in range(n):
			for j in range(n):
				ci = df[COMMON_COLUMNS[i]]
				cj = df[COMMON_COLUMNS[j]]
				temp = pd.DataFrame([ci,cj])

    #Purpose: analyze one vs. all correlation 
    #Params: data frame, subset type keyword
    #Return: pandas df with variance
	def oneVsAllCorr(self, df, subsetType):
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

		# divide each by correlation matrix for all variables

    #Purpose: conduct analysis based on df subsets
    #Params: type of analysis, subsets, cols to include
    #Return: pandas df with variance
	def subsetAnalysis(self, analysisType, colsToInclude, subset):
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

				if analysisType == "one/all corr":
					#get correlation
					cors = self.df[colsToInclude].corr()
					d[c] = cors.iloc[1:,:]

		return pd.DataFrame(d, index=self.h.getSubset(subset))

    #Purpose: develop correlational matrix
    #Params: data frame for correlation matrix, graph title 
    #Return: NA 
	def correlationMatrix(self, df, title):
		#setup 
		temp = df
		labs = temp.columns.values
		mat = temp.corr().values

		#add annotations
		annotations = go.Annotations()
		for i,r in enumerate(mat):
			for ii,rr in enumerate(r):
				annotations.append(go.Annotation(
					text=str('%.2f'%(rr)), x=labs[i], y=labs[ii],
					 xref='x1', yref='y1', showarrow=False,
					 font=dict(size=8, color='black')))
		
		#develop layout
		layout = go.Layout( 
			title={"text": title},
			xaxis={"tickfont": {"size": 8}, "showgrid":False},
			yaxis={"tickfont": {"size": 8}, "showgrid":False},
			plot_bgcolor='grey',
			autosize= False,
			annotations=annotations
		)

		#add data from temp
		data = go.Heatmap(
			type='heatmap',
			x=temp.columns.values,
			y=temp.columns.values,
			z=mat,
			colorscale='blues'
		)

		#create and show figure
		fig = go.Figure(data=data, layout=layout)
		fig.show()

    #Purpose: develop bivariate plot matrix
    #Params: NA 
    #Return: NA 
	def scatterMatrix(self, which):
		#graph all combinations in groups of 5
		#if which == 'all':
			#iterate through all combinations by 5
			#print(self.df.columns.values)
			#for i in range(self.df.shape[1]):


		fig = px.scatter_matrix(self.df.iloc[:,90:93])
		fig.show()

    # Purpose: correlation hypothesis test
    # Params: df 
    # Return: NA
	def correlationHypothesisTest(self, df):
		# get all unqiue pairs and setup vars
		uniquePairs = list(combinations(COMMON_COLUMNS, 2))
		corrs = np.array([])
		pVals = np.array([])
		names = np.array([]) 

		# get variable vectors to test
		for a,b in uniquePairs:
			temp = df[[a,b]].dropna()
			c1, c2 = temp.iloc[:,0], temp.iloc[:,1]

			# get hypothesis test for correlation	
			r, pVal = stats.pearsonr(c1, c2)

			# add to array
			corrs = np.append(corrs, r)
			pVals = np.append(pVals, pVal)
			names = np.append(names, a+' --- '+b)

		# perform benjamini hochberg correction
		bools, adjustedVals, aS, aB = multi.multipletests(pVals, method='fdr_bh')

		# print values for input into word
		print('B-H Correction Sig.,Pearson R, Organism Pair')
		for i in range(len(pVals)):
			print(str(bools[i])+', '+str(round(corrs[i], 4))+', '+names[i])

		# print(str(r)+', '+str(pVal)+': '+a+' | '+b)

    # Purpose: perform a two sample indepent t test on correlation coefficient
    # Params:  a, b, alpha, null hypothesis, alternative hypothesis 
    # Return: (z-score, p-value) 
	def tTest(self, a, b, alpha, hNull, hAlt):
		# remove nas
		temp = pd.DataFrame({'a':a,'b':b}).dropna()
		a, b = temp['a'], temp['b']

		# get stdev
		stdev = np.stdev([a,b])
		

		# get stdev
		stdev = np.sqrt((varA+varB)/2)

		# get test stat
		tStat = (np.mean(a)-np.mean(b))/(stdev/np.sqrt(2/n))

		# calcalte degrees of freedom and get p-value
		dof = 2*n-2
		pVal = 1-stats.t.cdf(tStat,df=dof)

		t2, p2 = stats.ttest_ind(a,b)
		if pVal == p2:
			return (tStat, pVal)
		else:
			print("incorrect p value")
			return (t2, p2)

######################### run ###########################
#read in data
df = cleaning().readData()
#print(df.describe())
#print(set(df['Errors?']))
a = analysis(df)

#analyze
# ONE VS ALL VARIANCE
#var = a.subsetAnalysis(analysisType='one/all var', colsToInclude=COMMON_COLUMNS, subset="Ocean")
#print(var.apply(lambda x: np.mean(x), axis=0))
#print(var.apply(lambda x: np.mean(x), axis=1))

# BIVARIATE NORMAL TEST
#a.bivariateNormalTest(df)

# ONE VS ALL CORRELATION 
#a.oneVsAllCorr(df, 'Ocean')

# HYPOTHESIS TESSTING
a.correlationHypothesisTest(df)

#print(corrs) #NOT WORKING
#print(h.aggDF(df))
#h.printUniqueCol(df, NONSTATIC_DESCRIPTORS[31:len(NONSTATIC_DESCRIPTORS)])

#correlation matrix
#h.correlationMatrix(df)
#h.scatterMatrix(df, which='all')

#dm.helpers().showDF(df, how='pivot')
