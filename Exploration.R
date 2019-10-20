#Explore the data from reef check

#############3 Read in Files #############3
library("readxl")
path = '~/Documents/Penn 2019-2020/Senior Thesis/Data/'

belt <- read_excel(paste(path, 'Belt.xlsx', sep = ''))
descriptors_non <- read_excel(paste(path, 'Non-Static Descriptors.xlsx', sep = ''))
descriptors_static <- read_excel(paste(path, 'Static Descriptors.xlsx', sep = ''))
substrate <- read_excel(paste(path, 'Substrate.xlsx', sep = ''))

