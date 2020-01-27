"""
	Author: Michael Berk 
	Date: Spring 2020
	Description:
		- assess historical trends between variables
			- linear + correlation, smoother
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

# other imports
import pandas as pd
import numpy as np

# create local example of class 
v = AnalysisHelpers.visualizations()
d = AnalysisHelpers.data()
s = AnalysisHelpers.s() # my stats class

##################################
##################################
# Vars 
##################################
##################################
df = d.readData()

##################################
##################################
# Non-linear Relationship 
##################################
##################################
def twoVarScatter(c1Name, c2Name):
	# create local copy of df 
	ldf = df[[c1Name, c2Name]]

	# rename for convinence
	ldf = ldf.rename(columns={c1Name:'x', c2Name:'y'})

	# remove outliers
	#ldf.x = ldf.x[ldf.x.between(ldf.x.quantile(.0), ldf.x.quantile(.95))] # without outliers
	#ldf.y = ldf.y[ldf.y.between(ldf.y.quantile(.0), ldf.y.quantile(.95))] # without outliers

	# clean 
	ldf = ldf.replace(0, np.nan) # remove 0s
	ldf = ldf.dropna()

	# sort ldf
	ldf = ldf.sort_values(by=['x'])

	# get average at each count
	x, y = np.array([]), np.array([])
	
	for i in ldf.x.unique():
		# add x subset
		x = np.append(x, i)

		# add y average
		sub = ldf[ldf.x == i][ldf.y.name]
		y = np.append(y, np.mean(sub))
	
	# graph
	v.scatterPlot(ldf.x, ldf.y, None, title=c1Name+'_'+c2Name)
	v.scatterPlot(x, y, None, title=c1Name+'_'+c2Name)

# 2 Var Scatter Analsysis
#c1Name = 'TRASH GENERAL'
#c2Name = 'GIANT CLAM TOTAL'
#c1Name = 'MORAY EEL'
#c2Name = 'BUTTERFLYFISH'
#c1Name = 'CORAL DAMAGE OTHER'
#c2Name = 'GORGONIAN'
#c1Name = 'CORAL DAMAGE OTHER'
#c2Name = 'BUTTERFLYFISH'
c1Name = 'CORAL DAMAGE OTHER'
c2Name = 'SNAPPER'
twoVarScatter(c1Name, c2Name)

def timeSeries(yName):
    # get local df and sort
    ldf = df[['Date',yName]]
    ldf = ldf.dropna()

    # remove outliers
    if REMOVE_OUTLIERS:
        ldf[yName] = ldf[yName][ldf[yName].between(ldf[yName].quantile(.0), ldf[yName].quantile(.95))] 

    if REMOVE_ZEROS:
        ldf = ldf[ldf[yName] != 0]

    # create average
    if AVG:
        # get unique dates
        ldf = ldf.groupby(['Date']).mean() 
    ldf['Date'] = ldf.index
    ldf.index.name = None

    # convert date and sort
    ldf.Date = [pd.to_datetime(d, format='%m/%d/%y') for d in ldf.Date]
    ldf = ldf.sort_values(by='Date')
   
    # display graph
    v.scatterPlot(ldf.Date,ldf[yName],None,title=yName+' over Time')

        

###############################33
REMOVE_OUTLIERS = True
REMOVE_ZEROS = True
AVG = True
# Time series
y = 'BLEACHING (% OF COLONY)'
#timeSeries(y)

#############################
# INTER TROPHIC RELATIONSHIP
def interTrophic():
    """Get sums for organisms for each trophic level the run correlation matrix"""
    
    # create copy and fill na
    ldf = df.copy()
    
    # create correlation mat
    #v.correlationMatrix(ldf[['1_C','2_C','3_C','4_C']], title='Trophic Level Scatter Matrix')

    # show plot
    ldf.sort_values(by=['1_C','2_C'])
    v.scatterPlot(ldf['1_C'],ldf['2_C'], None, 'Primary vs. Secondary Consumers')

   
#interTrophic()
