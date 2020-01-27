"""
    Author: Michael Berk
    Date: Spring 2020
    Description: perform visualizations that summarize the data. None of these will be findings per se, but will inform the direction of future analysis.
"""

################################################################
################################################################
# Imports and Globals
################################################################
################################################################
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

df = d.readData()

# Load and clean data: python3 -c 'import LoadAndClean as lc; lc.runData()'

################################################################
################################################################
# Summary Class
################################################################
################################################################
class summary():
    def __init__(self):
        print("Summary class created")

    def violinPlotAnalysis(self, cName):
        # call violin plot
        v.violinPlot(df, cName, title=cName)

    def histogramAnalysis(self, cName):
        # create local copy of df
        ldf = df[[cName]]

        # rename for convinence
        ldf = ldf.rename(columns={cName:'x'})

        # remove outliers
        if max(ldf.x) > 100: # make sure not a percentage or low count
            ldf.x = ldf.x[ldf.x.between(ldf.x.quantile(.0), ldf.x.quantile(.95))] # without outliers

        # clean
        ldf = ldf.replace(0, np.nan) # remove 0s
        ldf = ldf.dropna()

        # call histogram plot
        v.histogram(ldf,'x',title=cName+'Zeros removed')

# create class
ss = summary()

# test merge 
v.showDF(df, how='pivot')

# VIOLIN PLOTS
#for x in SUBSETS.COMMON_COLUMNS:
#    ss.violinPlotAnalysis(x)

# HISTOGRAMS
#for x in SUBSETS.COMMON_COLUMNS:
#    ss.histogramAnalysis(x)

# SUMMER vs. WINTER temperatures
