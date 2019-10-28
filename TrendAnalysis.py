#This file will be used to develop conclusions about the trends of bioinidcators species on coral reefs over time
#The desired output will be visualizations and charts that provide insight into these trends 
#Date: 11/2019
#Author: Michael Berk

##################### imports and global vars ###################
import DataMerge as dm 
import pandas as pd

from plotly.graph_objs import *
import plotly.express as px
import matplotlib.pyplot as plt

######################### helpers ###########################
class helpers():
	def __init__(self):
		print("Class 'helpers' created")

    #Purpose: read in data from dm
    #Params: NA 
    #Return: NA 
	def readData(self):
		return dm.run().runIt()

    #Purpose: develop correlational matrix
    #Params: data 
    #Return: NA 
	def correlationMatrix(self, df):
		#calculate correlation values
		df = df.iloc[:,:]
		f = plt.figure(figsize=(19, 15))
		plt.matshow(df.corr(), fignum=f.number)
		plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
		plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
		cb = plt.colorbar()
		cb.ax.tick_params(labelsize=14)
		plt.title('Correlation Matrix', fontsize=16);
		plt.show()

    #Purpose: develop bivariate plot matrix
    #Params: data 
    #Return: NA 
	def scatterMatrix(self, df):
		fig = px.scatter_matrix(df.iloc[:,55:60])
		fig.show()

######################### run ###########################
#create class instnaces
h = helpers()

#read in data
df = h.readData()

#correlation matrix
h.correlationMatrix(df)

