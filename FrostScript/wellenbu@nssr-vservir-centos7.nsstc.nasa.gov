# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 13:02:58 2016

@author: Lee Ellenburg
walter.l.ellenburg@nasa.gov

This script has adapted a previous frost prediction algorithm
created by Cerese Albers. The algorithm was deveolped for a sensor
network, the original methodology can be found in the document entiled:

FROST FORECASTING USING THE SERVIR WIRELESS SENSOR NETWORK
FOR HUNTSVILLE ALABAMA

Here the methodology was adapted in to python to injest WRF 48 hour
forcast model runs.

"""


import numpy as np
import pandas as pd
import os
from datetime import datetime
from osgeo import gdal, osr
from gdalconst import *
import urllib2
from mpl_toolkits.basemap import Basemap

#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt

############################################
#Definitions
############################################

#The GDAL conversions only work for Mercator projections
#If run for other areas in different projections
#The script need to be altered!
#Contact script author id needed

#Converts GRIB files to GeoTiff
def gribToTiff(gribFile, inputGribBand,	geoTiff_data):

    command = gdalTranslateExe+" -of GTiff -b "+str(inputGribBand)+" "+gribFile+" "+geoTiff_data
    os.system(command)


# Function to read tiff projection:
def GetGeoInfo(FileName):
    SourceDS = gdal.Open(FileName, GA_ReadOnly)
    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    DataType = SourceDS.GetRasterBand(1).DataType
    DataType = gdal.GetDataTypeName(DataType)
    return NDV, xsize, ysize, GeoT, Projection, DataType

# Function to write a new file.
def CreateGeoTiff(Name, Array, driver,
                  xsize, ysize, GeoT, Projection, DataType):
    if DataType == 'Float64':
        DataType = gdal.GDT_Float64
    NewFileName = Name+'.tif'
    # Set up the dataset
    DataSet = driver.Create( NewFileName, xsize, ysize, 1, DataType )
            # the '1' is for band 1.
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection( Projection.ExportToWkt() )
    # Write the array
    DataSet.GetRasterBand(1).WriteArray( Array )
    return NewFileName


# Function to write a new file.

def urlGrib(url, filename, date):
    print 'Downloading file: ' + gribName
    response = urllib2.urlopen(url) 
    with open(filename,'w') as f: f.write(response.read())
    return filename
        

#####################################################################
#####################################################################

###########################################
#Set Working directories
##########################################

tempdir = "/nas/rhome/ellenbw/Frost/Temp/"
datadir = "/nas/rhome/ellenbw/Frost/MetData/"
frostdir = "/nas/rhome/ellenbw/Frost/Frost/"

ftp = 'ftp://geo.msfc.nasa.gov/SPoRT/modeling/wrf/eafsport/grib2/'

#############################################
#Set global variables
############################################

#Set GDAL Executablew loations. In general: /usr/local/bin/

gdalTranslateExe = "/nas/rhome/ellenbw/anaconda2/bin/gdal_translate"

#Set date variable: 

date = pd.to_datetime(datetime.now())

###############################################################
###############################################################


#First download and save a grib file (any of the forcast runs), convert it to a projected
#tiff then capture the geograohical information for later.

#This captures the current days 0 hour forecast

gribName = 'EAF-SPORT_'+str(date.year)[-2:]+str(date.month).zfill(2)+str(date.day).zfill(2)+\
           '0000_wrfout_arw_d02.grb2f000000'

gribFile = urlGrib(ftp+gribName, tempdir+'tempGrib.grb2', date)

gribToTiff(gribFile, 1, tempdir+'0.tif')#, tempdir+'1.tif')

#NDV, xsize, ysize, GeoT, Projection, DataType = GetGeoInfo(tempdir+'1.tif')
NDV, xsize, ysize, GeoT, Projection, DataType = GetGeoInfo(tempdir+'0.tif')

#Define output arrays : This will hold all the frost flags/indices for each forecast hour for each pixel

#Flag array, grouped into 5 flags, 
hrarray = np.zeros((48,ysize,xsize))

#Index array, idenifies the most extreme frost condition met (of the original 15)
condList= []

#Loop through each forcast period (hr)
for fcst in range(1,49):

    #Download grib file : this will continue to overwrite the previous temp grib file created
   
    gribName = 'EAF-SPORT_'+str(date.year)[-2:]+str(date.month).zfill(2)+str(date.day).zfill(2)+\
               '0000_wrfout_arw_d02.grb2f'+str(fcst).zfill(2)+'0000'

    gribFile = urlGrib(ftp+gribName, tempdir+'tempGrib.grb2', date)
    

    #Save GeoTiffs for each variable: 2m Temp, 2m RH, and 10m WS (avgerage values)
    #Grib variable locations are 326, 332, and 337 respectively

    T_tiff = datadir + 'Temp2m_'+ str(date.month).zfill(2)+str(date.day).zfill(2)+'_'+str(fcst).zfill(2)+'.tif'
    RH_tiff = datadir + 'RH2m_'+ str(date.month).zfill(2)+str(date.day).zfill(2)+'_'+str(fcst).zfill(2)+'.tif'
    WS_tiff = datadir + 'WS2m_'+ str(date.month).zfill(2)+str(date.day).zfill(2)+'_'+str(fcst).zfill(2)+'.tif'
    
    gribToTiff(gribFile, 326, T_tiff)
    gribToTiff(gribFile, 332, RH_tiff)
    gribToTiff(gribFile, 337, WS_tiff)


    #Open each tiff, grab 1st band and set to array
    #Assign no data with numpy nan
    
    t2mbar = gdal.Open(T_tiff, GA_ReadOnly)
    t2mbar = t2mbar.GetRasterBand(1).ReadAsArray()
    t2mbar[t2mbar == NDV] = np.nan

    rh2mbar = gdal.Open(RH_tiff, GA_ReadOnly)
    rh2mbar = rh2mbar.GetRasterBand(1).ReadAsArray()
    rh2mbar[rh2mbar == NDV] = np.nan

    ws10mbar = gdal.Open(WS_tiff, GA_ReadOnly)
    ws10mbar = ws10mbar.GetRasterBand(1).ReadAsArray()
    ws10mbar[ws10mbar == NDV] = np.nan

    ########################################################

    #Next loop through each value of the arrays:
    
    #If one of the intial checks retuns 0 (no frost)
    #the routine for that value is stopped and a flag of 0 is reported
    
    #If the initial check is anyting > 0 the routine moves to more thorough
    #frost checks
	
    hrList = []
    for i in range(0,ysize-1):
        jlist = []
        for j in range(0,xsize-1):

            ############################################################################
            #Variable Assignments
            
            T = t2mbar[i,j]
            RH = rh2mbar[i,j] 
            WS = ws10mbar[i,j]

            #Dew Point is only reported instantaneously, and here we are using the hourly
            #avegerages, so DP is calculated from temperature and relative humidity
            #using the August-Roche-Magnus approximation. 
            DP = 243.04*(np.log(RH/100.0)+((17.625*T)/(243.04+T)))/(17.625-np.log(RH/100.0)-\
                                                                    ((17.625*T)/(243.04+T)))
            DEP = T-DP
            
            #################################################################################
            
            #Intial Check
            # Checks to see if value is nan
            if np.isnan(T) or  np.isnan(RH) or  np.isnan(WS):
                gflag = np.nan
                
            # If relative Humidity is less than 50%: Radiative frost not likely'
            if RH < 50:
                gflag = 0
                
            #Specify if winds are sufficient for risk of frost formation: i.e. < 1 m/s
                
            if WS < 3:
                gflag = 1
                
            elif WS > 3:
                gflag = 0

            if gflag < 1:
                flag = gflag
                index = np.nan
                
            else:
                
                flag = 0
                index = np.nan
                    
                #Condition 1: Alpha, Beta, Gamma:
                if 1 <= T < 2 and 1.5 < DEP <= 2:
                    flag =1
                    index = 'alpha'
                    #'Alpha: Frost conditions may be likely soon'
                if 1 <= T < 2 and 0.75 < DEP <= 1.5:
                    flag = 1 
                    index = 'beta'
                    #'Beta: Frost conditions should appear soon')
                if 1 <= T < 2 and 0 < DEP <= 0.75:
                    flag = 1  
                    index = 'gamma'
                    #Gamma: Pockets of frost likely'

                #Condition 2: Delta, Epsilon, Zeta, Eta, Theta:
                if 0 <= T < 1 and 1.5 < DEP <= 2:
                    flag = 2  
                    index = 'delta'
                    #'Delta: Light frost likely'
                if 0 <= T < 1 and 0.75 < DEP <= 1.5:
                    flag = 2  
                    index = 'epsilon'
                    #'Epsilon: Light to moderate frost likely '
                if 0 <= T < 1 and 0 < DEP <= 0.75:
                    flag = 2
                    index = 'zeta'
                    #'Zeta: Moderate frost likley'
                if -2 <= T < 0 and 1.5 < DEP <= 2:
                    flag = 2  
                    index = 'eta'
                    #'Eta: Light frost likely'
                if -2 <= T < 0 and 0.75 < DEP <= 1.5:
                    flag = 2 
                    index = 'theta'
                    #'Theta: Light to moderate frost likely'

                #Condition 3: Iota:
                if -2 <= T < 0 and 0 < DEP <= 0.75:
                    flag = 3
                    index = 'iota'
                    #'Iota: Monderate to heavy frost likely'
                    
                #Condition 4: Kappa, Lambda, Mu:
                if -5 <= T < -2 and 1.5 < DEP <= 2:
                    flag = 4
                    index = 'kappa'
                    #'Kappa: prolonged light frost likely'
                if -5 <= T < -2 and 0.75 < DEP <= 1.5:
                    flag = 4  
                    index = 'lambda'
                    #'Lambda: Prolonged light to moderate frost likely'
                if -5 <= T < -2 and 0 < DEP <= 0.75:
                    flag = 4
                    index = 'mu'
                    #'Mu: Prolonged moderate frost likely'
                    
                #Condition 5: Nu, Xi, Omicron:
                if T < -5 and 1.5 < DEP <= 2:
                    flag = 5 
                    index = 'nu'
                    #'Nu: Prolonged moderate to heavy frost likely'
                if T < -5 and 0.75 < DEP <= 1.5:
                    flag = 4 
                    index = 'xi'
                    #'Xi: Prolonged heavy frost likely '
                if T < -5 and 0 < DEP <= 0.75:
                    flag = 4
                    index = 'omicron'
                    #'Severly damaging frost likely'

            #For each pixel in j row
            jlist.append(index)
            
            hrarray[fcst-1,i,j] = flag
            
        #At the end of each row            
        hrList.append(jlist)
    #At the end of each hr
    condList.append(hrList)

            
    print '\n Forecast hour: '+str(fcst)+ ' is complete\n'

#Separate into two days	
d1array = hrarray[slice(0,24),:,:]
d2array = hrarray[slice(24,49),:,:]

#Find the max index that occurs
d1max = np.max(d1array,0)
d2max = np.max(d2array,0)

#Write to geoTiffs
driver = gdal.GetDriverByName('GTiff')

FrostD1 = CreateGeoTiff(frostdir+'FrostDay1_'+str(date.day).zfill(2)+'-'+str(date.month).zfill(2)+'-'+str(date.year)[2:], d1max, driver,
                        xsize, ysize, GeoT, Projection, DataType)

FrostD2 = CreateGeoTiff(frostdir+'FrostDay2_'+str(date.day).zfill(2)+'-'+str(date.month).zfill(2)+'-'+str(date.year)[2:], d2max, driver,
                        xsize, ysize, GeoT, Projection, DataType)


##################################################################
#End of Frost Script
#################################################################
####################################################################

################################################################
#Python plotting
################################################################


#This can be cut and pasted in ipython with the %paste command
#so that you can plot without having to rerun the script each time


'''
#Begin plotting code

def convertXY(xy_source, inproj, outproj):
    # function to convert coordinates

    shape = xy_source[0,:,:].shape
    size = xy_source[0,:,:].size

    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(inproj, outproj)
    xy_target = np.array(ct.TransformPoints(xy_source.reshape(2, size).T))

    xx = xy_target[:,0].reshape(shape)
    yy = xy_target[:,1].reshape(shape)

    return xx, yy

for day in [FrostD1, FrostD2]:


    # Read the data and metadata
    ds = gdal.Open(day)

    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    xres = gt[1]
    yres = gt[5]

    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5


    ds = None

    # create a grid of xy coordinates in the original projection
    xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]

    fig = plt.figure(figsize=(12, 6))
    m = Basemap(projection='robin', lon_0=0, resolution='c')

    # Create the projection objects for the convertion
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)

    # Get the target projection from the basemap object
    outproj = osr.SpatialReference()
    outproj.ImportFromProj4(m.proj4string)

    # Convert from source projection to basemap projection
    xx, yy = convertXY(xy_source, inproj, outproj)

    # plot the data (first layer)
    im1 = m.pcolormesh(xx, yy, data[:,:].T, cmap=plt.cm.jet)
    #im1 = m.pcolormesh(xy_source[0,:,:], xy_source[1,:,:], data[:,:].T, cmap=plt.cm.jet)

    # annotate
    m.drawcountries()
    m.drawcoastlines(linewidth=.5)

    plt.savefig('world.png',dpi=75)


 '''       

        




 



