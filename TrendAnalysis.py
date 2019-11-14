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


#TODO break down into categorical and numeric variables

######################### helpers ###########################
class helpers():
	def __init__(self):
		print("Class 'helpers' created")

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
		colsToFloat = ['TRASH GENERAL','TRASH FISH NETS','CORAL DAMAGE OTHER','CORAL DAMAGE DYNAMITE','CORAL DAMAGE ANCHOR','Percent colonies bleached','Percent Bleaching','Percent of each colony']
		for c in colsToFloat:
			df[c] = pd.to_numeric(df[c], errors='coerce') #convert NA to NaN
			df[c] = df[c].apply(lambda x : x*100 if x < 1 and x != 0 else x) #change decimals to percentages

		#remove errors
		return df

    #Purpose: print unique values in column 
    #Params: df, column(s) 
    #Return: NA 
	def printUniqueCol(self, df, c):
		#check if iterable
		if type(c) == list:
			if type(c[0]) == str:
				for i in c:
					print(str(i)+": "+str(df[i].unique().tolist()))
			elif type(c[0]) == int:
				for i in c:
					print(str(i)+": "+str(df.iloc[:,i].unique().tolist()))
			else:
				print("Incorrect type 'printUniqueCol'")
		#if not iterable
		else:
			if type(c) == str:
				print(str(c)+": "+str(df[c].unique().tolist()))
			elif type(c) == int:
				print(str(c)+": "+str(df.iloc[:,c].unique().tolist()))
			else:
				print("Incorrect type 'printUniqueCol'")

    #Purpose: subset and aggregate data 
    #Params: data, subset by region
    #Return: df that is aggregated on day and given params 
	def aggDF(self, df, region=None):
		#gropu by value
		temp = df.groupby('Date')
		print(temp[0])
		return temp

    #Purpose: develop correlational matrix
    #Params: data 
    #Return: NA 
	def correlationMatrix(self, df):
		#setup
		df = df.iloc[:,160:]
		labs = df.columns.values
		mat = df.corr().values

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
			title={"text": "Correlation Matrix"},
			xaxis={"tickfont": {"size": 8}, "showgrid":False},
			yaxis={"tickfont": {"size": 8}, "showgrid":False},
			plot_bgcolor='grey',
			autosize= False,
			annotations=annotations
		)

		#add data from df
		data = go.Heatmap(
			type='heatmap',
			x=df.columns.values,
			y=df.columns.values,
			z=mat,
			colorscale='blues'
		)

		#create and show figure
		fig = go.Figure(data=data, layout=layout)
		fig.show()

    #Purpose: develop bivariate plot matrix
    #Params: data 
    #Return: NA 
	def scatterMatrix(self, df, which):
		#graph all combinations in groups of 5
		#if which == 'all':
			#iterate through all combinations by 5
			#print(df.columns.values)
			#for i in range(df.shape[1]):




		fig = px.scatter_matrix(df.iloc[:,90:93])
		fig.show()

######################### run ###########################
#create class instnaces
h = helpers()

#read in data
df = h.readData()
df = h.clean(df)
#print(h.aggDF(df))
#h.printUniqueCol(df, NONSTATIC_DESCRIPTORS[31:len(NONSTATIC_DESCRIPTORS)])

#correlation matrix
#h.correlationMatrix(df)
#h.scatterMatrix(df, which='all')

