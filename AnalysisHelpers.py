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
from LoadAndClean import SUBSETS 

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
from scipy import signal

from itertools import combinations # tuple permutations
import statsmodels.stats.multitest as multi
import statsmodels.api as sm
    
########################################
# Visuzalizations 
########################################
class visualizations():
    def showDF(self,df,how='print'):
        """Visualize pandas based on given how parameter."""

        if how == 'print':
            #pretty print string
            print(df.to_string())

        elif how == 'pivot':
            #render pivot table in chrome
            pivot_ui(df,outfile_path='x.html')
            webbrowser.open('x.html',new=2)

        elif how == 'tabulate':
            #print string with borders
            print(tabulate(df, headers='keys', tablefmt='psql'))

        else:
            print('invalid "how" in showDF()')

        print(df.shape)

    def correlationMatrix(self, df, title):
        """Create and display a correlation matrix for every column in df."""

        #setup
        ldf = df.copy()
        ldf.replace(0, np.nan, inplace=True)
        labs = ldf.columns.values
        mat = ldf.corr(min_periods=10).values # get correlation mat with minimum samples for calculation

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

        #add data from ldf
        data = go.Heatmap(
            type='heatmap',
            x=ldf.columns.values,
            y=ldf.columns.values,
            z=mat,
            colorscale='blues'
        )

        #create and show figure
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def scatterMatrix(self, df):
        """Create and show a scatter matrix with the given columns."""

        # remove 0s
        ldf = df[columns].dropna()

        # create scatter matrix and show
        fig = px.scatter_matrix(ldf)
        fig.show()

    def scatterPlot(self, x, y, df2=None, title=None):
        """Create and show a scatter plot between two variables."""

        # setup names
        """tempDict = {x:np.array([]),
                 y:np.array([])}

        for i in ldf[x].unique():
            sub = ldf[ldf[x] == i][y]
            tempDict[x] = np.append(tempDict[x], i)
            tempDict[y] = np.append(tempDict[y], np.mean(sub))

        tempDF = pd.DataFrame(tempDict)
        """

        # create  figure with title
        fig = go.Figure()
        fig.update_layout(title=title)
        #fig.update_xaxes(title_text=)
        #fig.update_yaxes(title_text='Value A')

        # add x,y vals
        fig.add_trace(go.Scatter(x=x, y=y,
                    mode='markers',
                    name='V1'))

        # add lowess
        lowess = sm.nonparametric.lowess
        lvals = [l[1] for l in lowess(y, x, frac= .05)]
        fig.add_trace(go.Scatter(
            x=x,
            y=lvals,
            mode='lines',
            name='Lowess'
        ))

        # create line of best fit
        m, yint, r_value, p_value, std_err = stats.linregress(x,y)
        line = m*x+yint

        print('slope: ' + str(m))
        print('yint: ' + str(yint))

        # add line of best fit
        fig.add_trace(go.Scatter(
            x=x, 
            y=line,
            mode='lines',
            name='V2')
        )

        """    if df2 is not None:
            fig.add_trace(go.Scatter(x=ldf2.iloc[:,0], y=ldf2.iloc[:,0],
                        mode='markers',
                        name='V2'))
        """
        fig.show()

        # get trend line information
        #results = px.get_trendline_results(fig)
        #summary = results.px_fit_results.iloc[0].summary()
        #print(summary)

        return fig

    def map(self):
        """Map points according to lat/lon"""
        # create figure
        fig = px.scatter_mapbox(self.df, lat="Lat", lon="Lon", hover_name="Reef Name",
                        color_discrete_sequence=["red"], zoom=3)

        # update background
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
              ])
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()

    def violinPlot(self, df, yName, title=None):
        """Create and show a violin plot"""

        # create violin plot
        fig = px.violin(df, y=yName)
        fig.show()

    def histogram(self, df, xName, title=None):
        """Create and show historgram."""

        # call histogram
        fig = px.histogram(df, x=xName, title=title, nbins=30)
        fig.show()

########################################
# Data Manipulation 
########################################
class data():
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
        
        # create ldf copy
        ldf = df.copy()

        # print update
        print(col)
        print("Creating subsets based on: "+col)

        # prepare index for return 
        returnDict = {'ALL':ldf}

        # handle column specific subsets
        counts = ldf[col].value_counts()
        vals = counts[counts > minSampleSize].index.unique()
        
        # itereate and create boolean list 
        for v in vals:
            returnDict[v] = ldf[ldf[col] == v]

        # remove NA value
        del returnDict['NA']

        return returnDict

########################################
# Stats 
########################################
class s(): # stats
    def correlationHypothesisTest(self, df):
        """Perform correlation hypotehsis test with FDR"""

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
    def oneVsAllVar(self, df, subsetType):
        """Perform comparisons for variance between subsets and population.

        Param
            df: data frame to analyze (pd.DataFrame)
            subsetType: subset type keyword to be bassed to getSubset() (str)
        
        Return
            None    
        """
        # create ldf copy of df
        ldf = df.copy()

        #TODO: NOT TESTED
        #create empty ldf
        d = {}
        for c in ldf.columns.values:
            d[c] = np.array([])

        #iterate through subsets
        for s in self.getSubset(ldf, subsetType):
            temp = self.ldf[self.ldf[subsetType] == s]

            #iterate through all columns
            for c in colsToInclude:
                #calculate general vals
                v = np.var(temp[c])
                corrSubset = np.corrcoef(temp[c])

                if analysisType == "one/all var":
                    #get other variances and return
                    totalVar = np.var(self.ldf[c])
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

        # create local df
        ldf = df.copy()

        # setup return dict
        returnDict = {}

        # calcualte correlation/matrix for all variables
        self.correlationMatrix(ldf, 'No Subset')

        # get data subset
        for s in self.getSubset(ldf, subsetType):
            # get subset
            subsetDF = ldf[ldf[subsetType] == s][SUBSETS.COMMON_COLUMNS]

            # get correlation matrix
            self.correlationMatrix(subsetDF, s+' Subset')

        return returnDict

    def getMovingAverageValues(self, x, y, nX=1):
        """Calculate moving average values from data frame values.

        Param
            x: values to act as x-axis corresponding to nX (pd.Series)
            y: values to act as y-axis corresponding to mean vals (pd.Series)
            nX: x range for moving average (int)

        Return
            DF with mean(x range) as x values and moving average  as y values (pd.DataFrame)
        """    

        # calculate moving average
        xx = np.array([])
        yy = np.array([])
        #TODO

        for i in x:
            temp = y[x.values == i]
            avg = np.nanmean(temp.iloc[:,0])
            print(np.nanmean(temp.iloc[:,0]))
            yy = np.append(yy, avg)
            xx = np.appned(xx, i)
        
        yVals = y.rolling(window=nX).mean()    
        xVals = x.rolling(window=nX).mean()
        print(xVals)
        print(yVals)

        # return dataframe
        rdf = pd.DataFrame({x.name:xx, y.name:yy})

        return rdf
