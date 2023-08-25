"""A collection of functions for the analysis of astronomical data cubes and images.

                NICCI
Naive functIons Collection for Cubes and Images
"""
#############################################################################################
#
# FILENAME: main
# VERSION: 0.14
# DATE: 25/08/2023
# CHANGELOG:
#- chanmap
#   - fixed channel range to account for python 0-counting
#   - fixed error message if mask and data cube have different sizes
#   - fixed error message if some cube properties must be retrieved by cubestat
#   - improved figure size
#   - added option to choose the colormap
#   - added option to choose the contours colormap
#   - removed the option to save the plot
#   - collapsed the output arguments into a single argument
#   - minor code rearranging to improve code visualization
#- cubedo
#   - fixed spectral axis calculation if the spectral resolution is negative
#   - fixed integer conversion not being performed if write_fits=False for operation 'toint'
#   - fixed channel range calculation for operation 'blank','cut','mom0'
#   - fixed mask and cube shape comparison for operation 'mom0'
#   - improved some argument naming
#   - improved argument handling for operation 'mirror'
#   - improved excecution time of operation 'shuffle'
#   - removed data type storing
#   - collapsed the output arguments into a single argument
#   - collapsed the rotation center arguments for 'mirror' into a single argument
#   - minor code rearranging to improve code visualization
#- cubestat
#   - improved functionality. Now can take a dictionary (like the configuration file) as input and updates it with the missing information
#   - added fluxrange option for rms calculation
#   - removed rms and sensitivity error computation
#   - rewritten the outputs
#   - minor code rearranging to improve code visualization
#- fitsarith
#   - improved data import
#   - rewritten the outputs
#   - collapsed the output arguments into a single argument
#   - minor code rearranging to improve code visualization
#- fixmask
#   - collapsed the output arguments into a single argument
#   - minor code rearranging to improve code visualization
#- gaussfit
#   - fixed mask import
#   - fixed spectral axis calculation if the spectral resolution is negative
#   - improved mandatory inputs retrival and check
#   - improved fitting routine
#   - added option to fit a double component
#   - collapsed the output arguments into a single argument
#   - minor code rearranging to improve code visualization
#- getpv
#   - improved mandatory input retrival and check
#   - improved figure size
#   - added fluxrange option for rms calculation
#   - added option to choose the colormap
#   - added option to choose the contours colormap
#   - removed the option to save the plot
#   - removed the suptitle option
#   - removed objname from the kwargs
#   - collapsed the output arguments into a single argument
#   - minor code rearranging to improve code visualization
#- lines finder
#   - fixed SNR calculation
#   - fixed error when providing list of spectra but a single rms value
#   - improved the setup for the reliability calculation
#   - improved the processing of the reliability outcomes
#   - improved the linker
#   - improved the reliability calculation
#   - added the mean and the median of the spectrum to the source catalogue
#   - collapsed the output arguments into a single argument
#- noise_variations
#   - fixed error when a glitchy pixel is in the cube
#   - improved plots
#   - added fluxrange option for rms calculation
#   - added noise cube as fits and spectral variations as csv to the outputs
#   - added arguments to import_parameters
#   - added arguments to create_config
#- plotmom
#   - fixed error on beamarea convertion to arcsec if no beam is given
#   - improved visualization if a single map is plotted
#   - improved figure size
#   - improved subplots positioning
#   - improved automatic countours calculation for moment 0 map
#   - improved moment 0 colorbar tick labels
#   - improved default colormaps (moment 1 and moment 2)
#   - improved conotur colors and visualization (moment 1 and moment 2)
#   - added countour levels to the colobars
#   - removed the option to save the plot
#   - removed the option to use the data cube, as this will be done automatically if necessary
#   - collapsed the output arguments into a single argument
#   - minor code rearranging to improve code visualization
#- removemod
#   - collapsed the output arguments into a single argument
#   - minor code rearranging to improve code visualization
#- rotcurve
#   - collapsed the output arguments into a single argument
#- stacking
#   - fixed median periodicity plot
#   - improved y-axis label of stacked spectra
#   - collapsed the output arguments into a single argument
#- velfi
#   - collapsed the output arguments into a single argument
#- import_parameters
#   - changed imports according to changes in the above functions
#- create_config
#   - fixed typos
#   - improved and fixed descriptions
#   - changed according to changes in the above functions
#- flux
#   - fixed error due to wrong variable naming
#   - fixed box loading when input data is a string
#- getHImass
#   - fixed box loading when input data is a string
#   - minor code rearranging to improve code visualization
#- __plot_stack_spectrum
#   - improved y-axis label of stacked spectra
#- __source_linker
#   - (!!! SE USI QUELLO VECCHIO!!!) improved performance
#   - (!!! SE USI QUELLO NUOVO!!!) completely rewritten. Now acts as the one in SoFiA2
#- __reliability (RENAMED)
#   - completely rewritten. Now acts as the one in SoFiA2
#   - added the Skellam diagnostic plot
#-  __rms (RENAMED)
#   - this function now handles the fluxrange. All the other functions, which were using the fluxrange argument, have been changed accordingly
#- (NEW) __covariance_to_error_ellipse
#- (REMOVED) __get_sources_type
#- improved memory management of large (>1 Gb) fits file by reverting the memory mapping of astropy to True and brute force delete the mapped memory when the fits is closed
#- created custom colormaps for the moment 1, moment 2 and moment 2 contours
#- the margins for plotting the ancillary information are now controlled through the configuration file in all the plotting functions
#- changed strings in f-strings
#- minor fixes
#- improved documentation
#
#   TO DO:  - sistema i docstrings delle private functions
#           - aggiungere il logger (con logger.)
#
#############################################################################################

from astropy import units as u
from astropy.coordinates import ICRS
from astropy.coordinates import SkyCoord
import astropy.convolution as conv
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter as fit
from astropy.visualization import hist
from pvextractor import Path
from pvextractor import PathFromCenter
from pvextractor import extract_pv_slice
from scipy.special import erf
from scipy.stats import chi2 as statchi2
from scipy.stats import anderson
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm
import configparser
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as cl
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox
import numpy as np
import os
import pandas as pd
import subprocess
import warnings

def _full_warning(message,category,filename,lineno,file=None,line=None):
    return 'WARNING: %s:%s: %s: %s\n' %(filename,lineno,category.__name__,message)
def _simple_warning(message,category,filename,lineno,file=None,line=None):
    return 'WARNING: %s: %s\n' %(category.__name__,message)
    
warnings.formatwarning=_simple_warning

plt.rc('font',size=14) #set the default fontsize for the text in the atlas
plt.rcParams['mathtext.default']='regular'
plt.rcParams['xtick.direction']='in'
plt.rcParams['xtick.major.size']=10
plt.rcParams['xtick.major.width']=1
plt.rcParams['xtick.minor.size']=5
plt.rcParams['xtick.minor.width']=1
plt.rcParams['ytick.direction']='in'
plt.rcParams['ytick.major.size']=10
plt.rcParams['ytick.major.width']=1
plt.rcParams['ytick.minor.size']=5
plt.rcParams['ytick.minor.width']=1
plt.rcParams['axes.formatter.use_mathtext']=True
#plt.rcParams['axes.formatter.limits']=(-2,2) #use scientific notation if log10 of the axis range is smaller than the first or larger than the second

fits.Conf.use_memmap=True #disable the memomory map when open a fits

__all__=['import_parameters','chanmap','cubedo','cubestat','fitsarith','fixmask','gaussfit','getpv','lines_finder','noise_variations','radial_profile','plotmom','removemod','rotcurve','stacking','velfi','converttoHI','cosmo','getHImass','flux','create_config'] #list of public function to be imported when running 'from nicci import *'

## --- CUSTOM COLORMAPS --- ##
#change to plt.colormaps.register when updating matplotlib

if 'mom1' not in plt.colormaps(): #register the colormap for the moment 1 maps if not existing
    plt.register_cmap(cmap=cl.LinearSegmentedColormap.from_list('mom1',[(95/255,235/255,255/255),(0/255,0/255,255/255),(0/255,0/255,125/255),(255/255,255/255,255/255),(125/255,0/255,0/255),(255/255,0/255,0/255),(255/255,95/255,135/255)]))
    
if 'mom1_alt' not in plt.colormaps(): #register the colormap for the moment 1 maps if not existing
    plt.register_cmap(cmap=cl.LinearSegmentedColormap.from_list('mom1_alt',[(95/255,235/255,255/255),(0/255,0/255,255/255),(0/255,0/255,125/255),(0/255,255/255,0/255),(125/255,0/255,0/255),(255/255,0/255,0/255),(255/255,95/255,135/255)]))
    
if 'mom2' not in plt.colormaps(): #register the colormap for the moment 2 maps if not existing
    plt.register_cmap(cmap=cl.LinearSegmentedColormap.from_list('mom2',[(1,1,1),(0/255,255/255,255/255,0.3),(255/255,255/255,0/255),(255/255,127/255,0/255),(200/255,0/255,0/255)])) 

if 'mom2_ctr' not in plt.colormaps(): #register the colormap for the moment 2 contours if not existing
    plt.register_cmap(cmap=cl.LinearSegmentedColormap.from_list('mom2_ctr',[(0/255,155/255,255/255,0.75),(150/255,0/255,0/255)]))

#############################################################################################
def import_parameters(parameter_file=''):
    """Read an INI-structured text file and store each row in a dictionary entry that can be passed to other functions. This is the main function used to control the arguments of other functions using a parameter file.
    
    Args:
        parameter_file (str): Name or path+name of the text file structured as an INI configuration file (i.e., a list of rows with various [section] headers) containing the parameter values. Parameters not in the file will be initialized to default values.
    
    Returns:
        dict: Python dictionary with all the parameters needed to run the functions.
        
    Raises:
        None
    """
    config=configparser.ConfigParser(allow_no_value=True,inline_comment_prefixes=('#',';')) #initialize the import tool
    config.read(parameter_file) #read the parameters file
    parameters={} #initialize the parameters dictionary
    
    #GENERAL section
    parameters['GENERAL']='----------------- GENERAL -----------------'
    parameters['verbose']=config.getboolean('GENERAL','verbose',fallback=False) #print the outputs option. If it is not given, set it to False
    
    #INPUT section
    parameters['INPUT']='----------------- INPUT -----------------'
    parameters['path']=config.get('INPUT','path',fallback=os.getcwd()+'/') #path to the working directory (generally where the data are stored). If it is not given, set it to the currect directory
 
    #COSMOLOGY section (default values: https://ui.adsabs.harvard.edu/abs/2014ApJ...794..135B/abstract)
    parameters['COSMOLOGY']='----------------- COSMOLOGY -----------------'
    parameters['H0']=config.getfloat('COSMOLOGY','H0',fallback=69.6) #Hubble parameter. If it is not given, set it to 69.6
    parameters['Omega_matter']=config.getfloat('COSMOLOGY','Omegam',fallback=0.286) #Omega matter. If it is not given, set it to 0.286
    parameters['Omega_vacuum']=config.getfloat('COSMOLOGY','Omegav',fallback=0.714) #Omega vacuum. If it is not given, set it to 0.714
    
    #GALAXY section
    parameters['GALAXY']='----------------- GALAXY -----------------'
    parameters['objname']=config.get('GALAXY','objname',fallback=None) #name of the object. If it is not given, set it to None
    parameters['redshift']=config.getfloat('GALAXY','redshift',fallback=None) #redshift of the object. If it is not given, set it to None
    parameters['distance']=config.getfloat('GALAXY','distance',fallback=None) #distance of the object in Mpc. If it is not given, set it to None
    parameters['asectokpc']=config.getfloat('GALAXY','asectokpc',fallback=None) #arcsec to kpc conversion. If it is not given, set it to None
    parameters['vsys']=config.getfloat('GALAXY','vsys',fallback=None) #systemic velocity of the object in km/s. If it is not given, set it to None
    parameters['pa']=config.getfloat('GALAXY','pa',fallback=None) #position angle of the object in deg. If it is not given, set it to None
    parameters['inc']=config.getfloat('GALAXY','inc',fallback=None) #inclination angle of the object in deg. If it is not given, set it to None
    
    if parameters['distance'] is None or parameters['asectokpc'] is None: #if no distance is or angular scale is given
        if parameters['redshift'] is not None:  #if the redshift is given
            distance=cosmo(parameters['redshift'],parameters['H0'],parameters['Omega_matter'],parameters['Omega_vacuum'],verbose=parameters['verbose']) #calculate the cosmology
            if parameters['distance'] is None: #if the distance is not given
                parameters['distance']=distance['distance [Mpc]'] #get it from the cosmology
            if parameters['asectokpc'] is None: #if the angular scale is not given
                parameters['asectokpc']=distance['angular scale'] #get it from the cosmology
        else:
            warnings.warn('No redshift is provided. Cannot compute the cosmology.')
    
    #CUBEPAR section
    parameters['CUBEPAR']='----------------- CUBEPAR -----------------'
    parameters['pixunits']=config.get('CUBEPAR','pixunits',fallback=None) #spatial units. If it is not given, set it to None
    parameters['specunits']=config.get('CUBEPAR','specunits',fallback=None) #spectral units. If it is not given, set it to None
    parameters['fluxunits']=config.get('CUBEPAR','fluxunits',fallback=None) #spatial units. If it is not given, set it to None
    parameters['spectralres']=config.getfloat('CUBEPAR','spectralres',fallback=None) #spectral resolution of km/s. If it is not given, set it to None
    parameters['pixelres']=config.getfloat('CUBEPAR','pixelres',fallback=None) #pixel resolution in arcsec. If it is not given, set it to None
    parameters['rms']=config.getfloat('CUBEPAR','rms',fallback=None) #root-mean-square value in Jy/beam. If it is not given, set it to None
    
    #BEAM section
    parameters['BEAM']='----------------- BEAM -----------------'
    parameters['bmaj']=config.getfloat('BEAM','bmaj',fallback=None) #beam major axis in arcsec. If it is not given, set it to None
    parameters['bmin']=config.getfloat('BEAM','bmin',fallback=None) #beam minor axis in arcsec. If it is not given, set it to None
    parameters['bpa']=config.getfloat('BEAM','bpa',fallback=None) #beam position angle in arcsec. If it is not given, set it to None

    #CORRECTION section
    parameters['CORRECTION']='----------------- CORRECTION -----------------'
    parameters['pbcorr']=config.getboolean('CORRECTION','pbcorr',fallback=False) #primary beam correction option. If it is not given, set it to False

    #FITS section
    parameters['FITS']='----------------- FITS -----------------'
    parameters['datacube']=config.get('FITS','datacube',fallback=None) #name of the fits file of the data cube including .fits. If it is not given, set it to None
    parameters['beamcube']=config.get('FITS','beamcube',fallback=None) #name of the fits file of the beam cube including .fits. If it is not given, set it to None
    parameters['maskcube']=config.get('FITS','maskcube',fallback=None) #name of the fits file of the mask cube including .fits. If it is not given, set it to None
    parameters['mask2d']=config.get('FITS','mask2d',fallback=None) #name of the fits file of the 2D mask including .fits. If it is not given, set it to None
    parameters['channelmap']=config.get('FITS','channelmap',fallback=None) #name of the fits file of the channel map including .fits. If it is not given, set it to None
    parameters['modelcube']=config.get('FITS','modelcube',fallback=None) #name of the fits file of the model cube including .fits. If it is not given, set it to None
    parameters['mom0map']=config.get('FITS','mom0map',fallback=None) #name of the fits file of the moment 0 map including .fits. If it is not given, set it to None
    parameters['mom1map']=config.get('FITS','mom1map',fallback=None) #name of the fits file of the moment 1 map including .fits. If it is not given, set it to None
    parameters['mom2map']=config.get('FITS','mom2map',fallback=None) #name of the fits file of the moment 2 map including .fits. If it is not given, set it to None
    parameters['vfield']=config.get('FITS','vfield',fallback=None) #name of the fits file of the velocity field including .fits. If it is not given, set it to None
    #CHANMAP
    parameters['CHANMAP']='----------------- CHANMAP -----------------'
    parameters['from_chan']=config.getint('CHANMAP','chanmin',fallback=1) #starting channel to plot in the channel map. If it is not given, set it to 0
    parameters['to_chan']=config.getint('CHANMAP','chanmax',fallback=None) #ending channel to plot in the channel map. If it is not given, set it to None
    parameters['chansep']=config.getint('CHANMAP','chansep',fallback=1) #channel separation to plot in the channel map (chamin,chanmin+chansep,chanmin+2*chansep,...,chanmax). If it is not given, set it to 1
    parameters['chanbox']=config.get('CHANMAP','box').split(',') if config.has_option('CHANMAP','box') else None #comma-separated pixel edges of the box to extract the channel mapS in the format [xmin,xmax,ymin,ymax]. If it is not given, set it to None
    if parameters['chanbox'] != [''] and parameters['chanbox'] is not None: #if the box is given
        parameters['chanbox']=[int(i) for i in parameters['chanbox']] #convert string to float
    parameters['chansig']=config.getfloat('CHANMAP','nsigma',fallback=3) #rms threshold to plot the contours (lowest contours will be nsigma*rms). If it is not given, set it to 3
    parameters['chanmask']=config.getboolean('CHANMAP','usemask',fallback=False) #use a mask in the channel map [True,False]. If it is not given, set it to False
    parameters['chanmap_out']=config.get('CHANMAP','output',fallback='') #output directory/name to save the plot. If it is not given, set it to None
     
    #CUBEDO section
    parameters['CUBEDO']='----------------- CUBEDO -----------------'
    parameters['cubedo']=config.get('CUBEDO','datacube',fallback=None) #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['operation']=config.get('CUBEDO','operation',fallback=None) #operation to perform on the cube [blank,clip,crop,cut,extend,mirror,mom0,shuffle,toint]. If it is not given, set it to None
    parameters['chanmin']=config.getint('CUBEDO','chanmin',fallback=1) #first channel for the operations 'blank,cut,mom0'. If it is not given, set it to 0
    parameters['chanmax']=config.getint('CUBEDO','chanmax',fallback=None) #last channel for the operations 'blank,cut,mom0'. If it is not given, set it to None
    parameters['inbox']=config.get('CUBEDO','box').split(',') if config.has_option('CUBEDO','box') else None #comma-separated pixel edges of the box to extract for operation 'cut' in the format [xmin,xmax,ymin,ymax]. If it is not given, set it to None
    if parameters['inbox'] != [''] and parameters['inbox'] is not None: #if the box is given
        parameters['inbox']=[int(i) for i in parameters['inbox']] #convert string to float
    parameters['addchan']=config.getint('CUBEDO','addchan',fallback=None) #number of channels to add in operation 'extend'. Negative values add lower channels, positive add higher channels. If it is not given, set it to None
    parameters['value']=config.get('CUBEDO','value',fallback='blank') #value to assign to blank pixel in operation 'blank' (blank is np.nan). If it is not given, set it to 'blank'
    if parameters['value'] != 'blank': #if the value is not blank
        parameters['value']=float(parameters['value']) #convert it to float
    parameters['with_mask']=config.getboolean('CUBEDO','usemask',fallback=False) #use a 2D mask in the operation 'clip' [True,False]. If it is not given, set it to False
    parameters['cubedo_mask']=config.get('CUBEDO','mask',fallback=None) #name of the fits file of the 2D mask including .fits. If empty, is the same of [FITS] mask2d. If it is not given, set it to None
    parameters['cliplevel']=config.getfloat('CUBEDO','cliplevel',fallback=0.5) #clip threshold as % of the peak (0.5 is 50%) for operation 'clip'. If it is not given, set it to 0.5
    parameters['rot_center']=config.getfloat('CUBEDO','center',fallback=None) #the rotation center as [x0,y0,z0] around which the cube is mirrored when operation is 'mirror'. If it is not given, set it to None
    parameters['cubedo_out']=config.get('CUBEDO','output',fallback='') #output directory/name to save the new cube. If it is not given, set it to None
        
    #CUBESTAT section 
    parameters['CUBESTAT']='----------------- CUBESTAT -----------------'
    parameters['nsigma']=config.getfloat('CUBESTAT','nsigma',fallback=3) #rms threshold in terms of nsigma*rms for detection limit. If it is not given, set it to 3
    parameters['cs_statistics']=config.get('CUBESTAT','statistics',fallback='mad') #statistic to be used in the noise measurement. If it is not given, set it to mad
    parameters['cs_fluxrange']=config.get('CUBESTAT','fluxrange',fallback='negative') #flux range to be used in the noise measurement. If it is not given, set it to negative
    
    #FITSARITH section
    parameters['FITSARITH']='----------------- FITSARITH -----------------'
    parameters['fits1']=config.get('FITSARITH','fits1',fallback=None) #name of reference fits file including .fits. If it is not given, set it to None
    parameters['fits2']=config.get('FITSARITH','fits2',fallback=None) #name of second fits file including .fits. If it is not given, set it to None
    parameters['do']=config.get('FITSARITH','operation',fallback=None) #operation to do between the two fits [sum,sub,mul,div]. If it is not given, set it to None
    parameters['fits_out']=config.get('FITSARITH','output',fallback='') #output directory/name to save the fits file. If it is not given, set it to None

    #FIXMASK section
    parameters['FIXMASK']='----------------- FIXMASK -----------------'
    parameters['refcube']=config.get('FIXMASK','datacube',fallback=None) #name of the fits file of the reference data cube including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['masktofix']=config.get('FIXMASK','maskcube',fallback=None) #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube. If it is not given, set it to None
    parameters['fix_out']=config.get('FIXMASK','output',fallback='') #output directory/name to save the fits file. If it is not given, set it to Nonee
    
    #GAUSSFIT section
    parameters['GAUSSFIT']='----------------- GAUSSFIT -----------------'
    parameters['cubetofit']=config.get('GAUSSFIT','datacube',fallback=None) #name of the fits file of the data cube to fit including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['gaussmask']=config.get('GAUSSFIT','mask',fallback=None) #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube. The fit will be done inside the mask. If it is not given, set it to None
    parameters['components']=config.getfloat('GAUSSFIT','components',fallback=1) #number of components (1 or 2) to fit. If it is not given, set it to 1
    parameters['linefwhm']=config.getfloat('GAUSSFIT','linefwhm',fallback=15) #first guess on the fwhm of the line profile in km/s. If it is not given, set it to 15
    parameters['amp_thresh']=config.getfloat('GAUSSFIT','amp_thresh',fallback=0) #amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum. If it is not given, set it to 0
    parameters['p_reject']=config.getfloat('GAUSSFIT','p_reject',fallback=1) #p-value threshold for fit rejection. If a best-fit as p>p_reject, it will be rejected . If it is not given, set it to 1
    parameters['clipping']=config.getboolean('GAUSSFIT','clipping',fallback=False) #clip the spectrum to a % of the profile peak [True,False]. If it is not given, set it to False
    parameters['clip_threshold']=config.getfloat('GAUSSFIT','threshold',fallback=0.5) #clip threshold as % of the peak (0.5 is 50%) if clipping is True. If it is not given, set it to 0.5
    parameters['errors']=config.getboolean('GAUSSFIT','errors',fallback=False) #compute the errors on the best-fit [True,False]. If it is not given, set it to False
    parameters['write_field']=config.getboolean('GAUSSFIT','write_field',fallback=False) #compute the best-fit velocity field [True,False]. If it is not given, set it to False
    parameters['gauss_out']=config.get('GAUSSFIT','output',fallback='') #output directory/name to save the fits file. If it is not given, set it to Nonee

    #GETPV section
    parameters['GETPV']='----------------- GETPV -----------------'
    parameters['pvcube']=config.get('GETPV','datacube',fallback=None) #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['pvwidth']=config.getfloat('GETPV','width',fallback=None) #width of the slice in arcsec. If not given, will be the beam size. If it is not given, set it to None
    parameters['pvpoints']=config.get('GETPV','points').split(',') if config.has_option('GETPV','points') else None #RA-DEC comma-separated coordinate of the slice points in decimal degree. If two are given ([x,y]), it is assumed they are the center of the slice. Else, they need to be the starting and ending coordinates ([xmin,xmax,ymin,ymax]). If it is not given, set it to None
    if parameters['pvpoints'] != [''] and parameters['pvpoints'] is not None: #if the pv points are given
        parameters['pvpoints']=[float(i) for i in parameters['pvpoints']] #convert string to float
    parameters['pvangle']=config.getfloat('GETPV','angle',fallback=None) #position angle of the slice in degree. If not given, will be the object position angle. If it is not given, set it to None
    parameters['pvchmin']=config.getint('GETPV','chanmin',fallback=1) #starting channel of the slice. If it is not given, set it to 0
    parameters['pvchmax']=config.getint('GETPV','chanmax',fallback=None)#ending channel of the slice. If it is not given, set it to None
    parameters['pv_statistics']=config.get('GETPV','statistics',fallback='mad') #statistic to be used in the noise measurement. If it is not given, set it to mad
    parameters['pv_fluxrange']=config.get('GETPV','fluxrange',fallback='negative') #flux range to be used in the noise measurement. If it is not given, set it to negative
    parameters['pvsig']=config.getfloat('GETPV','nsigma',fallback=3) #rms threshold to plot the contours (lowest contours will be nsigma*rms). If it is not given, set it to 3
    parameters['pv_out']=config.get('GETPV','output',fallback='') #output directory/name to save the fits file. If it is not given, set it to None
          
    #LINESFINDER section
    parameters['LINESFINDER']='----------------- LINESFINDER -----------------'     
    parameters['smooth_kernel']=config.get('LINESFINDER','smooth_ker').split(',') if config.has_option('LINESFINDER','smooth_ker') else None #kernel size (or comma-separated kernel sizes) in odd number of channels for spectral smoothing prior the source finding. Set to None or 1 to disable. If it is not given, set it to None
    if parameters['smooth_kernel'] is not None:
        parameters['smooth_kernel']=[int(i) for i in parameters['smooth_kernel']] #convert string to int
        if len(parameters['smooth_kernel'])==1: #if only 1 kernel is provided
            parameters['smooth_kernel']=parameters['smooth_kernel'][0] #convert to single float
    parameters['sc_threshold']=config.getfloat('LINESFINDER','threshold',fallback=3) #number of rms to reject flux values in the source finder. If it is not given, set it to None
    parameters['sc_replace']=config.getfloat('LINESFINDER','replace',fallback=2) #before smoothing the spectrum during n source finder iteration, every flux value that was already detected in a previous iteration will be replaced by this value multiplied by the original noise level. If it is not given, set it to 2
    parameters['sc_fluxrange']=config.get('LINESFINDER','fluxrange',fallback='negative') #flux range to be used in the noise measurement of the source finder. If it is not given, set it to negative
    parameters['sc_statistics']=config.get('LINESFINDER','statistics',fallback='mad') #statistic to be used in the noise measurement process of the source finder. If it is not given, set it to mad
    parameters['link_kernel']=config.getint('LINESFINDER','link_ker',fallback=3) #minimum odd number of channels covered by a spectral line. If it is not given, set it to 3
    parameters['min_size']=config.getint('LINESFINDER','min_size',fallback=3) #minimum number of channels a source can cover. If not given, set it to 3
    parameters['rel_threshold']=config.getfloat('LINESFINDER','rel_thresh',fallback=0) #minimum value (from 0 to 1) of the reliability to consider a source reliable. Set to 0 to disable the reliability calculation. If it is not given, set it to 0
    parameters['rel_kernel']=config.getfloat('LINESFINDER','rel_kernel',fallback=0.4) #scaling factor for the size of the Gaussian kernel used when estimating the density of positive and negative detections in the reliability parameter space.. If it is not given, set it to 0.4
    parameters['rel_snrmin']=config.getfloat('LINESFINDER','rel_snrmin',fallback=3) #minimum SNR of a detected line to be reliable. If it is not given, set it to 3
    parameters['keep_negative']=config.getboolean('LINESFINDER','negative',fallback=True) #don't discard detections with negative flux at the end of the process. If it is not given, set it to True    
    parameters['gaussianity']=config.getboolean('LINESFINDER','gauss_tests',fallback=False) #if set to True, two statistical tests (Anderson-Darling and Kanekar) will be performed to quantify the gaussianity of the noise in the input spectra. If not given, set it to False
    parameters['p_value']=config.getfloat('LINESFINDER','p-value',fallback=0.05) #p-value threshold of the Anderson-Darling test to reject non-Gaussian spectra (a spectrum has Gaussian noise if p>p_value). If not given, set it to 0.05  
    parameters['kanekar_threshold']=config.get('LINESFINDER','kanekar').split(',') if config.has_option('LINESFINDER','kanekar') else [0.8,1.16] #an array like of two float elements representing the minimum and maximum value of the Kanekar test to reject non-Gaussian spectra (a spectrum has Gaussian noise if kanekar_min<test_value<kanekar_max). If not given, set it to None
    if len(parameters['kanekar_threshold']) == 2: #if two elements are given
        parameters['kanekar_threshold']=[float(i) for i in parameters['kanekar_threshold']] #convert string to float
    else:
        parameters['kanekar_threshold']=[0.8,1.16] #set to the default values
    parameters['write_catalogue']=config.getboolean('LINESFINDER','catalogue',fallback=False) #option to write the source catalague as csv file. If it is not given, set it to False  
    parameters['finder_out']=config.get('LINESFINDER','output',fallback='') #output directory/name to save the new cube. If it is not given, set it to None

    #NOISE VARIATIONS section
    parameters['NOISE VARIATIONS']='----------------- NOISE VARIATIONS -----------------'
    parameters['nv_statistics']=config.get('NOISE VARIATIONS','statistics',fallback='mad') #statistic to be used in the noise measurement. If it is not given, set it to mad
    parameters['nv_fluxrange']=config.get('NOISE VARIATIONS','fluxrange',fallback='negative') #flux range to be used in the noise measurement. If it is not given, set it to negative
    parameters['noise_out']=config.get('NOISE VARIATIONS','output',fallback='') #output directory/name to save the new cube. If it is not given, set it to None
    
    #PLOTMOM section
    parameters['PLOTMOM']='----------------- PLOTMOM -----------------'
    parameters['plotmom_out']=config.get('PLOTMOM','output',fallback='') #output directory/name to save the plot. If it is not given, set it to None
            
    #REMOVEMOD section
    parameters['REMOVEMOD']='----------------- REMOVEMOD -----------------'
    parameters['method']=config.get('REMOVEMOD','method',fallback='subtraction') #method to remove the model [all,blanking,b+s,negblank,subtraction]. If it is not given, set it to 'subtraction'
    parameters['blankthreshold']=config.getfloat('REMOVEMOD','threshold',fallback=0) #flux threshold for the 'all,blanking,b+s' methods in units of cube flux. If it is not given, set it to 0
    parameters['removemod_out']=config.get('REMOVEMOD','output',fallback='') #output directory/name to save the fits file. If it is not given, set it to None

    #ROTCURVE section
    parameters['ROTCURVE']='----------------- ROTCURVE -----------------'
    parameters['rotcenter']=config.get('ROTCURVE','center').split(',') if config.has_option('ROTCURVE','center') else None #x-y comma-separated coordinates of the rotational center in pixel. If it is not given, set it to None
    if parameters['rotcenter'] is not None:
        parameters['rotcenter']=[float(i) for i in parameters['rotcenter']] #convert string to float
    parameters['save_csv']=config.getboolean('ROTCURVE','save_csv',fallback=False) #store the output in a csv file [True,False]. If it is not given, set it to False
    parameters['rotcurve_out']=config.get('ROTCURVE','output',fallback='') #output directory/name to save the plot. If it is not given, set it to None

    #STACKING section
    parameters['STACKING']='----------------- STACKING -----------------'
    parameters['stackcenter']=config.get('STACKING','center').split(',') if config.has_option('STACKING','center') else None #x-y comma-separated coordinates of the galactic center in pixel. If it is not given, set it to None
    if parameters['stackcenter'] is not None:
        parameters['stackcenter']=[float(i) for i in parameters['stackcenter']] #convert string to float
    parameters['nregions']=config.getint('STACKING','nregions',fallback=None) #number of regions from which the spectra are extracted and stacked. If it is not given, set it to None
    parameters['shape']=config.get('STACKING','shape',fallback='cones') #shape of regions to stack between 'cells', 'cones' and 'concentric'. If it is not given, set it to 'cones'
    parameters['from_to']=config.get('STACKING','radii').split(',') if config.has_option('STACKING','radii') else None #comma-separated min and max radius from the galactic center in pixunits from which the spectra are extracted and stacked. If it is not given, set it to None'
    if parameters['from_to'] is not None:
        parameters['from_to']=[float(i) for i in parameters['from_to']] #convert string to float
    parameters['between_angles']=config.get('STACKING','between').split(',') if config.has_option('STACKING','between') else None #comma-separated min and max angle from which the stacking regions are defined. Set to None to disable. If it is not given, set it to None'
    if parameters['between_angles'] is not None:
        parameters['between_angles']=[float(i) for i in parameters['between_angles']] #convert string to float
    parameters['weighting']=config.get('STACKING','weighting',fallback=None) #type of weight to apply during the stacking between 'None' and 'rms'. If it is not given, set it to None
    parameters['stack_fluxrange']=config.get('STACKING','fluxrange',fallback='negative') #flux range to be used in the noise measurement. If it is not given, set it to negative
    parameters['stack_statistics']=config.get('STACKING','statistics',fallback='mad') #statistic to be used in the noise measurement. If it is not given, set it to mad
    parameters['stack_out']=config.get('STACKING','output',fallback='') #output directory/name to save the new cube. If it is not given, set it to None

    #VELFI section
    parameters['VELFI']='----------------- VELFI -----------------'
    parameters['radii']=config.get('VELFI','radii').split(',') if config.has_option('VELFI','radii') else None #radii in pixunits at which the rotation velocity is measured. If it is not given, set it to None
    if parameters['radii'] is not None:
        parameters['radii']=[float(i) for i in parameters['radii']] #convert string to float
    parameters['vrot']=config.get('VELFI','vrot').split(',') if config.has_option('VELFI','vrot') else None #rotation velocities (subtracted from the systemic velocities) in specunits for the radii. If it is not given, set it to None
    if parameters['vrot'] is not None:
        parameters['vrot']=[float(i) for i in parameters['vrot']] #convert string to float
    parameters['vrad']=config.get('VELFI','vrad').split(',') if config.has_option('VELFI','vrad') else None #expansion velocity for the radii. If it is not given, set it to None
    if parameters['vrad'] is not None:
        parameters['vrad']=[float(i) for i in parameters['vrad']] #convert string to float
    parameters['vcenter']=config.get('VELFI','center').split(',') if config.has_option('VELFI','center') else None #x-y comma-separated coordinates of the rotational center in pixel. If it is not given, set it to None
    if parameters['vcenter'] is not None:
        parameters['vcenter']=[float(i) for i in parameters['vcenter']] #convert string to float
    parameters['extend_only']=config.getboolean('VELFI','extend_only',fallback=False) #extend a given velocity field (True) or write a new one from scratch (False). If it is not given, set it to False
    parameters['correct']=config.getboolean('VELFI','correct',fallback=False) #correct the input rotation velocities for the inclination angles (True) or not (False). If it is not given, set it to False
    parameters['velfi_out']=config.get('VELFI','output',fallback='') #output directory/name to save the new cube. If it is not given, set it to None
    
    #PLOTSTYLE section
    parameters['PLOTSTYLE']='----------------- PLOTSTYLE -----------------'
    parameters['ctr_width']=config.getfloat('PLOTSTYLE','ctr_width',fallback=2) #width of the contours levels. If it is not given, set it to 2
    parameters['left_space']=config.getfloat('PLOTSTYLE','left_space',fallback=0.05) #width of the left margin in axis units (from 0 to 1). If it is not given, set it to 0.05
    parameters['upper_space']=config.getfloat('PLOTSTYLE','upper_space',fallback=0.95) #width of the upper margin in axis units (from 0 to 1). If it is not given, set it to 0.95
    parameters['lower_space']=config.getfloat('PLOTSTYLE','lower_space',fallback=0.05) #width of the lower margin in axis units (from 0 to 1). If it is not given, set it to 0.05
    parameters['plot_format']=config.getfloat('PLOTSTYLE','format',fallback='pdf') #file format for the plots. If it is not given, set it to pdf
            
    return parameters

#############################################################################################
def chanmap(datacube='',from_chan=None,to_chan=None,chansep=None,chanmask=False,chanmap_out='',**kwargs):
    """Plot the channel maps of a data cube in a defined channel range every given number of channels.

    Args:
        datacube (str/ndarray): Name or path+name of the FITS data cube.
        from_chan (int): First channel to be plotted in the channel map.
        to_chan (int): Last channel to be plotted in the channel map.
        chansep (int): Channel separation in the channel map. The channels plotted are from from_chan to to_chan, with a separation of chansep.
        chanmask (bool): Option to use a detection mask (True) or not (False) to highlight the 'real' emission in the channel map. If True, a 3D mask must be provided using the 'maskcube' kwarg (see kwargs arguments below).
        chanmap_out (str): The output folder/file name.

    Kwargs:
        path (str): Path to the data cube if the datacube is a name and not a path+name.
        maskcube (str/ndarray): Name or path+name of the FITS 3D mask cube. Used if chanmask=True.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        pixelres (float): Cube spatial resolution in pixunits (Default: None).
        spectralres (float): Cube spectral resolution in specunits (Default: None).
        bmaj (float): Beam major axis in arcsec in pixunits (Default: None).
        bmin (float): Beam minor axis in arcsec in pixunits (Default: None).
        bpa (float): Beam position angle in degrees (Default: None).
        rms (float): RMS of the data cube as a float (Default: None).
        chanbox (array-like): Spatial box as [xmin, xmax, ymin, ymax] (Default: None).
        vsys (float): Object systemic velocity in m/s (Default: 0).
        asectokpc (float): Arcsec to kpc conversion to plot the spatial scale (Default: None).
        objname (str): Name of the object (Default: '').
        contours (array-like): Contour levels in units of rms. These values will replace the default levels (Default: None)
        chansig (float): Lowest contour level in terms of chansig * rms (Default: 3).
        ctr_width (float): Line width of the contours (Default: 2).
        cmap (str/ndarray): Name of the colormap to be used. Accepted values are those of matplotlib.colors.Colormap. (Default: 'Greys')
        ctrmap (str/ndarray): Name of the colormap to be used for the contour levels. Accepted values are those of matplotlib.colors.Colormap. (Default: 'hot')
        left_space (float): Width of the left margin in axis units (from 0 to 1) (Default: 0.025).
        upper_space (float): Width of the upper margin in axis units (from 0 to 1) (Default: 0.025).
        lower_space (float): Width of the lower margin in axis units (from 0 to 1) (Default: 0.975).
        plot_format (str): File format of the plots (Default: pdf).
        verbose (bool): Option to print messages and plot to terminal if True (Default: False).

    Returns:
        None

    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no path is provided.
        ValueError: If no mask cube is provided when chanmask is True.
        ValueError: If wrong spatial units are provided.
        ValueError: If wrong spectral units are provided.
        ValueError: If no spectral information is available through the input arguments or the cube header.
        ValueError: If mask cube and data cube dimensions do not match.
        ValueError: If size of spatial box is not 4.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf') #store the plot output format
    datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    if chanmask: #if a mask cube is used
        maskcube=kwargs.get('maskcube',None)
        if maskcube is None: #if no masckcube is provided
            chanmask=False #set to False the switch to use the mask
            if verbose:
                warnings.warn('You chose to use a mask but no mask cube is provided. No mask will be load.')
        else:
            maskcube=__read_string(maskcube,'maskcube',**kwargs)  #store the path to the data cube
    output=chanmap_out+plot_format #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    #CHECK THE KWARGS#
    pixunits=kwargs.get('pixunits',None) #store the spatial units
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    specunits=kwargs.get('specunits',None) #store the spectral units
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')   
    pixelres=kwargs.get('pixelres',None) #store the spatial resolution
    spectralres=kwargs.get('spectralres',None) #store the spectral resolution
    bmaj=kwargs.get('bmaj',None) #store the beam major axis
    bmin=kwargs.get('bmin',None) #store the beam minor axis
    bpa=kwargs.get('bpa',None) #store the beam position angle
    rms=kwargs.get('rms',None) #store the rms
    chanbox=kwargs.get('chanbox',None) #store the spatial box
    vsys=kwargs.get('vsys',None) #store the systemic velocity
    if vsys is None: #if not given
        vsys=0 #set it to 0
        warnings.warn('No systemic velocity is given: set it to 0 m/s!')
    cs_statistics=kwargs.get('cs_statistics','mad') #store the statistics to be use for rms calculation
    asectokpc=kwargs.get('asectokpc',None) #store the arcseconds to kpc conversion
    objname=kwargs.get('objname','') #store the object name
    ctr=kwargs.get('contours',None) #store the contour levels
    sigma=kwargs.get('chansig',3) #store the lowest contour in terms of rms*sigma
    ctr_width=kwargs.get('ctr_width',2) #store the contour width
    cmap=kwargs.get('cmap','Greys') #store the colormap
    ctrmap=kwargs.get('ctrmap','hot') #store the contours colormap
    left_space=kwargs.get('left_space',0.025) #store the left margin for annotations position
    lower_space=kwargs.get('lower_space',0.025) #store the bottom margin for annotations position
    upper_space=kwargs.get('upper_space',0.975) #store the top margin for annotations position

    #---------------   START THE FUNCTION   ---------------#    
    #OPEN THE DATACUBE#
    data,header=__load(datacube) #open the data cube
        
    #CHECK FOR THE RELEVANT INFORMATION#
    #------------   CUBE PROPERTIES    ------------#
    prop=[pixunits,specunits,pixelres,spectralres,bmaj,bmin,bpa,rms] #list of cube properties values
    prop_name=['pixunits','specunits','pixelres','spectralres','bmaj','bmin','bpa','rms'] #list of cube properties names
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if one or more cube parameters are not given
        if verbose:
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            warnings.warn(f'I am missing some information: {not_found}. Running cubestat to retrieve them!')
        stats=cubestat(datacube,params_dict=dict(zip(prop_name,prop)),cs_statistics=cs_statistics,verbose=False) #calculate the statistics of the cube
        if pixunits is None: #if the spatial units are not given
            pixunits=stats['pixunits'] #take the value from the cubestat results
        if specunits is None: #if spectral units are not given
            specunits=stats['specunits'] #take the value from the cubestat results
        if pixelres is None: #if pixel resolution is not given
            pixelres=stats['pixelres'] #take the value from the cubestat results
        if spectralres is None: #if spectral resolution is not given
            spectralres=stats['spectralres'] #take the value from the cubestat results
        if bmaj is None: #if bmaj is not given
            bmaj=stats['bmaj'] #take the value from the cubestat results
        if bmin is None: #if bmin is not given
            bmin=stats['bmin'] #take the value from the cubestat results
        if bpa is None: #if bpa is not given
            bpa=stats['bpa'] #take the value from the cubestat results
        if rms is None: #if rms is not given
            rms=stats['rms'] #take the value from the cubestat results
    prop=[pixunits,specunits,pixelres,spectralres,bmaj,bmin,bpa,rms] #update the list of cube properties values
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if still one or more cube parameters are not given
        if verbose:
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            warnings.warn(f'I am still missing some information: {not_found}. I cannot display the information on the plot!')   
    #CONVERT THE VALUES IN STANDARD UNITS#
    if pixunits == 'deg': #if the spatial units are deg
        bmaj*=3600 #convert the beam major axis in arcsec
        bmin*=3600 #convert the beam minor axis in arcsec
        pixelres*=3600 #convert the pixel size in arcsec^2
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        bmaj*=60 #convert the beam major axis in arcsec
        bmin*=60 #convert the beam minor axis in arcsec
        pixelres*=60 #convert the pixel size in arcsec^2
    beamarea=1.13*(bmin*bmaj) #calculate the beam area
    if pixelres<0: #if the spatial resolution is negative
        pixelres=-pixelres #convert it to positive
        
    #-----------   STARTING VELOCITY    -----------#
    if 'CRVAL3' in header and 'CRPIX3' in header: #if the header has the starting spectral value
        v0=header['CRVAL3']+(header['CRPIX3']-1)*np.abs(spectralres) #store the starting spectral value
    else:
        raise ValueError('ERROR: no starting channel spectral value found: aborting!')
    if specunits=='m/s': #if the spectral units are m/s
        spectralres/=1000 #convert the spectral resolution in km/s
        v0/=1000 #convert starting spectral value to km/s
        vsys/=1000 #covert the systemic velocity in km/s
    wcs=WCS(header).dropaxis(2) #store the WCS info and drop the spectral axis

    #CHECK IF A MASK IS USED AND OPEN IT#
    if chanmask: #if a mask must be used
        mask,_=__load(maskcube) #open the mask cube
        if mask.shape[0] != data.shape[0] or mask.shape[1] != data.shape[1] or mask.shape[2] != data.shape[2]: #if the mask cube has different size than the data cube
            raise ValueError(f'ERROR: mask cube and data cube has different shapes:\n'
                                '{mask.shape} the mask cube and {data.shape} the data cube. Aborting!')
        mask[mask>1]=1 #convert the mask into a 0/1 array
             
    #CHECK THE CHANNELS#
    if from_chan is None or from_chan < 1: #if the lower channel is not given or less than 1
        warnings.warn(f'Starting channel wrongly set. You give {from_chan} but should be at least 1. Set to 1')
        from_chan=1 #set it to 1
    if to_chan is None or to_chan > data.shape[0]: #if the upper channel is not set or larger than the data size
        warnings.warn(f'Last channel wrongly set. You give {to_chan} but should be at maximum {data.shape[0]}. Set to {data.shape[0]}.')
        to_chan=data.shape[0]+1 #select until the last channel
    if to_chan < from_chan: #if the higher channel is less than the lower
        warnings.warn(f'Last channel ({to_chan}) lower than starting channel ({from_chan}). Last channel set to {data.shape[0]}.')
        to_chan=data.shape[0]+1 #select until the last channel
    from_chan-=1 #remove 1 from the first channel to account for python 0-counting. We do not have to do this for the to_chan as the indexing is exclusive, meaning that if to_chan=10, the operation will be done until pythonic channel 9, which is channel 10 when 1-counting
    if chansep is None: #if the channel separation is not given
        warnings.warn(f'Channel separation not set. Set to 1')
        chansep=1 #set it to 1
    chans=np.arange(from_chan,to_chan,int(chansep)) #channels to be used in channel map
    
    #CHECK THE SPATIAL BOX#
    xmin,xmax,ymin,ymax=__load_box(data,chanbox) #load the spatial box

    if verbose:
        print('The channel map will be plotted with the following parameters:\n'
        f'Spectral resolution: {spectralres:.1f} km/s\n'
        f'Spatial resolution: {pixelres:.1f} arcsec\n'
        f'Starting velocity: {v0-vsys:.1f} km/s\n'
        f'Systemic velocity: {vsys:.1f} km/s\n'
        f'Beam: {bmaj:.1f} x {bmin:.1f} arcsec\n'
        f'From pixel ({xmin},{ymin}) to pixel ({xmax},{ymax})\n'
        f'From channel {from_chan} to channel {to_chan} every {chansep} channels\n'
        '-------------------------------------------------')
        
    #SETUP THE FIGURE#    
    nrows=int(np.floor(np.sqrt(len(chans)))) #number of rows in the channel map
    ncols=int(np.floor(np.sqrt(len(chans)))) #number of columns in the channel map
    fig=plt.figure(figsize=(5*ncols,5*nrows)) #create the figure
    k=0 #index to run over the channels
    
    #SETUP THE SUBPLOT PROPERTIES#    
    #-------------   LIMITS  -------------#
    if chanbox is None: #if no box is applied
        xlim=[np.min(np.where(~np.isnan(data))[2])-5,np.max(np.where(~np.isnan(data))[2])+5] #set the xlim to the first and last not-nan values
        ylim=[np.min(np.where(~np.isnan(data))[1])-30,np.max(np.where(~np.isnan(data))[1])+5] #set the ylim to the first and last not-nan values
    else:
        xlim=[0,xmax-xmin] #set the xlim to the x-size of the box
        ylim=[-(ymax-ymin)*(bottommargin+0.1),(ymax-ymin)*(2.05-topmargin)] #extend the ylim to place the ancillary information. 0.1 and 2.05 are for the text size
    #it is highly raccomended to have an equal aspect ratio for the maps. So we have to extend the shorter axis to match the larger
    if ylim[1]-ylim[0] > xlim[1]-xlim[0]: #if the y-axis is bigger than the x-axis
        extend=(ylim[1]-ylim[0]-xlim[1]+xlim[0])/2 #calculate how much to extend
        xlim[1]+=extend
        xlim[0]-=extend
    else: #if the x-axis is larger
        extend=(xlim[1]-xlim[0]-ylim[1]+ylim[0])/2 #calculate how much to extend
        ylim[1]+=extend
        ylim[0]-=extend
    #----------   NORMALIZATION  ---------#
    vmin=0.05*np.nanmin(data) #set the lower limit for the normalization
    vmax=0.15*np.nanmax(data) #set the upper limit for the normalization
    norm=cl.PowerNorm(gamma=0.3,vmin=vmin,vmax=vmax) #define the data normalization
    #------------   CONTOURS  ------------#
    if ctr is None: #if contours levels are not given:
        maxsig=round(np.log(np.sqrt(vmax)/rms)/np.log(sigma)) #calculate the max sigma in the data
        ctr=np.power(sigma,np.arange(1,maxsig,2)) #contours level in units of nsigma^i (i from 1 to max sigma in step of 2)
        if len(ctr) < 5: #if not enough contours are produced
            ctr=np.power(sigma,np.arange(1,maxsig,1)) #contours level in units of nsigma^i (i from 1 to max sigma in step of 1)
        if len(ctr) < 5: #if still not enough contours are produced
            ctr=np.power(sigma,np.arange(1,maxsig,0.5)) #contours level in units of nsigma^i (i from 1 to max sigma in step of 0.5)
    else:
        ctr=np.array(ctr) #convert the list of contours into an arr
    if verbose:
        ctrtoHI=converttoHI(ctr*rms,beamarea=beamarea,pixunits='arcsec',spectralres=spectralres,specunits='km/s')
        print(f"Contours level: {['%.1e' % elem for elem in ctrtoHI]} cm⁻²")
        
    #DO THE PLOT#
    for k in tqdm(range((nrows*ncols)),desc='Channels plotted'):
        ax=fig.add_subplot(nrows,ncols,k+1,projection=wcs) #create the subplot
        chanmap=data[chans[k],ymin:ymax,xmin:xmax] #select the channel
        im=ax.imshow(chanmap,cmap=cmap,norm=norm,aspect='equal') #plot the channel map in units of detection limit
        if chanmask: #if a mask must be used
            ax.contour(chanmap/rms,levels=ctr,cmap=ctrmap,linewidths=ctr_width/2,linestyles='solid') #add the contours
            ax.contour(chanmap/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width/2,linestyles='dashed') #add the negative contours
            ax.contour(chanmap*mask[chans[k],ymin:ymax,xmin:xmax]/rms,levels=ctr,cmap='hot',linewidths=ctr_width,linestyles='solid') #add the contours within the mask
        else:
            ax.contour(chanmap/rms,levels=ctr,cmap=ctrmap,linewidths=ctr_width,linestyles='solid') #add the contours
            ax.contour(chanmap/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width,linestyles='dashed') #add the negative contours
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if k not in np.arange(0,nrows*ncols,ncols): #if not plotting the first column
            ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
        if k not in np.arange((nrows-1)*ncols,nrows*ncols): #if not plotting the last row
            ax.coords[0].set_ticklabel_visible(False) #hide the x-axis ticklabels and labels
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.text(left_space,lower_space,f'V$_{{rad}}$: {chans[k]*spectralres+v0-vsys:.1f} km/s',transform=ax.transAxes) #add the information of the channel velocity
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            if k in np.arange(0,nrows*ncols,nrows): #in the first column
                ax=__plot_kpcline(pixelres,asectokpc,xlim,left_space,0.925) #draw the 10-kpc line
            
    fig.subplots_adjust(wspace=0.015,hspace=0.015) #fix the position of the subplots in the figure
    fig.savefig(output+plot_format,dpi=300,bbox_inches='tight') #save the figure

    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    plt.close(fig)
    
#############################################################################################
def cubedo(cubedo='',operation=None,chanmin=None,chanmax=None,inbox=None,addchan=None,value='blank',with_mask=False,cubedo_mask=None,cliplevel=0.5,rot_center=None,write_fits=False,cubedo_out='',**kwargs):
    """Perform various operations on a FITS data cube, such as blanking, clipping, cropping, cutting, extending, mirroring, calculating moment 0 map, shuffling, and converting to an integer cube.

    Args:
        cubedo (str): The name or path+name of the FITS data cube.
        operation (str): The operation to perform. Accepted values are:
            - 'blank': Blanks all the data in the given [chanmin,chanmax,xmin,xmax,ymin,ymax] range.
            - 'clip': Blanks all the data below the cliplevel value. For each pixel, the values in its spectrum below the threshold times the spectrum peak will be set to 0. 
            - 'crop': Spatially crops the data cube to remove the blanked edges.
            - 'cut': Extracts the a cubelet in the given [chanmin,chanmax,xmin,xmax,ymin,ymax] range.
            - 'extend': Adds channels to the spectral axis.
            - 'mirror': Mirrors the data around a rotation point (x0,y0,z0).
            - 'mom0': Computes the moment 0 map from chanmin to chanmax.
            - 'shuffle': Aligns the spectral profiles with the given velocity field.
            - 'toint': Converts the data cube into an integer cube.
        chanmin (int): The first channel for operations blank, cut, and mom0.
        chanmax (int): The last channel for operations blank, cut, and mom0.
        inbox (array-like): The spatial cut box as [xmin,xmax,ymin,ymax].
        addchan (int): The number of channels to add in the operation extend. If < 0, the channels are added at the beginning of the spectral axis; otherwise, they are added at the end.
        value (float/string): The value to give to the blanked pixel in the operation blank. If the string 'blank' is used, it will be np.nan.
        with_mask (bool): Specifies whether to use a detection mask (True) or not (False) for clip and mom0. If True, a 2D mask for clip or a 3D mask for mom0 must be provided using the cubedo_mask argument or the 'mask2d' and 'maskcube' kwargs (see kwargs arguments below).
        cubedo_mask (str): The name or path+name of the FITS 2D or 3D mask. Used if with_mask=True.
        cliplevel (float): The clip threshold as a percentage of the peak (0.5 is 50%) for clip.
        rot_center (array-like): The rotation center as [x0,y0,z0] around which the cube is mirrored.
        write_fits (bool): Specifies whether to store the output in a FITS file (True) or return a variable (False).
        cubedo_out (str): The output folder/file name.
        
    Kwargs:
        datacube (str/ndarray): The name or path+name of the FITS data cube if cubedo is not given.
        path (str): The path to the data cube if the datacube is a name and not a path+name.
        vfield (str): The name or path+name of the FITS velocity field to be used for the operation shuffle.
        specunits (str): The string with the spectral units for the operation mom0 and shuffle (Default: m/s). Accepted values are:
            - 'km/s'
            - 'm/s'
            - 'Hz'
        spectralres (float): The cube spectral resolution in specunits for the operation mom0 and shuffle (Default: None).
        v0 (float): The starting velocity to create the spectral axis for the operation mom0 and shuffle (Default: None).
        mask2d (str): The name or path+name of the FITS 2D mask for the operation clip. Used if with_mask=True.
        maskcube (str/ndarray): The name or path+name of the FITS 3D mask cube for the operation mom0. Used if with_mask=True.
        verbose (bool): Specifies whether to print messages to the terminal if True (Default: False).
    
    Returns:
        Resulting cube or moment map as a FITS file.
        
    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no output name is provided.
        ValueError: If no operation is set.
        ValueError: If the operation set does not match the accepted values.
        ValueError: If no velocity field is given when 'shuffle' is set.
        ValueError: If no spectral resolution is provided when 'mom0' or 'shuffle' is set.
        ValueError: If no mask is provided when usemask is True.
        ValueError: If no spectral information is available through the input arguments or the cube header.
        ValueError: If the mask cube and data cube dimensions do not match.
        ValueError: If the size of the spatial box is not 4.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    datacube=__read_string(cubedo,'datacube',**kwargs) if type(cubedo) in [str,type(None)] else cubedo #store the path to the data cube if the datacube is a string
    if operation is None: #if no operation is set
        raise ValueError('ERROR: no operation set: aborting')
    if operation not in ['blank','clip','crop','cut','extend','mirror','mom0','shuffle','toint']: #if wrong operation is given
        raise ValueError("ERROR: wrong operation. Accepted values: ['blank','clip','crop','cut','extend','mirror','mom0','shuffle','toint']. Aborting")
    if operation in ['shuffle']: #if the datacube must be shuffled
        vfield=kwargs.get('vfield',None)
        if vfield is None: #if no vfield is provided
            raise ValueError("Selected operation is 'shuffle', but no velocity field is provided. Aborting.")
        if type(vfield)==str: #if the velocity field is a string
            vfield=__read_string(vfield,'vfield',**kwargs)
    if operation in ['clip']:  #if the datacube must be clipped
        usemask=with_mask #store the use mask switch from the input parameters
        if usemask: #if a 2D mask is used
            mask2d=__read_string(cubedo_mask,'mask2d',**kwargs) if type(cubedo_mask) in [str,type(None)] else cubedo_mask #if the 2D mask is a string store the 2D mask from the input parameters
        threshold=cliplevel #store the clipping threshold from the input parameters
    if operation in ['mirror']: #if the cube must be mirrored
        if rot_center is not None and len(rot_center) != 3: #if the rotation center has the wrong size
            raise ValueError('ERROR: Please provide the rotation center in the format [x0,y0,z0]. Aborting!')
    if operation in ['mom0']:  #if the datacube must be clipped
        specunits=kwargs.get('specunits',None) #store the spectral units from the input paramaters
        spectralres=kwargs.get('spectralres',None) #store the spectral resolution from the input paramaters
        usemask=with_mask #store the use mask switch from the input parameters
        if usemask: #if a 2D mask is used
            maskcube=__read_string(cubedo_mask,'maskcube',**kwargs) if type(cubedo_mask) in [str,type(None)] else  cubedo_mask #store the 3D mask from the input parameters if the 3D mask is a string
    if operation in ['shuffle']: #if the datacube must be shuffled
        spectralres=kwargs.get('spectralres',None) #store the spectral resolution from the input paramaters
        v0=kwargs.get('v0',None) #store the starting velocity from the input paramaters
    if write_fits: #if the data must be written into a fits file
        output=cubedo_out #store the output directory/name from the input parameters
        if output in ['',None]: #if no output is provided
            raise ValueError('ERROR: no output name set: aborting')
        if output[0]=='.': #if the output name start with a . means that it contains a path
            path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
            if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    
    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #load the datacube
        
    #CHECK THE CHANNELS#
    if operation in ['blank','cut','mom0']: #if the operation to be done is blank, cut or mom0
        if chanmin is None or chanmin < 1: #if the lower channel is not given or less than 1
            warnings.warn(f'Starting channel wrongly set. You give {chanmin} but should be at least 1. Set to 1')
            chanmin=1 #set it to 1
        if chanmax is None or chanmax > data.shape[0]: #if the upper channel is not set or larger than the data size
            warnings.warn(f'Last channel wrongly set. You give {chanmax} but should be at maximum {data.shape[0]}. Set to {data.shape[0]}.')
            chanmax=data.shape[0]+1 #select until the last channel
        if chanmax < chanmin: #if the higher channel is less than the lower
            warnings.warn(f'Last channel ({chanmax}) lower than starting channel ({chanmin}). Last channel set to {data.shape[0]}.')
            chanmax=data.shape[0]+1 #select until the last channel
        chanmin-=1 #remove 1 from the first channel to account for python 0-counting. We do not have to do this for the chanmax as the indexing is exclusive, meaning that if chanmax=10, the operation will be done until pythonic channel 9, which is channel 10 when 1-counting
        
    #CHECK THE SPATIAL BOX
    if operation in ['blank','cut']: #if the operation to be done is blank or cut
        xmin,xmax,ymin,ymax=__load_box(data,inbox) #load the spatial box
    
    #------------   BLANK     ------------#  
    if operation == 'blank': #if the datacube must be blanked
        data[chanmin:chanmax,ymin:ymax,xmin:xmax]=np.nan #blank the data
    
    #------------   CLIP     ------------# 
    if operation == 'clip': #if the datacube must be clipped
        clip_cube=np.zeros(data.shape) #initialize the clipped cube as zeros
        if usemask: #if a mask is used
            mask,_=__load(mask2d) #load the mask
            if mask.shape[0] != data.shape[1] or  mask.shape[1] != data.shape[2]: #if the 2D mask has different spatial sizes than the data cube
                raise ValueError(f'ERROR: mask and data cube has different spatial shapes: {mask.shape} the mask and ({data.shape[1]},{data.shape[2]}) the data cube. Aborting!')
            else:
                x=np.where(mask>0)[0] #store the x coordinate of the non-masked pixels
                y=np.where(mask>0)[1] #store the y coordinate of the non-masked pixels
        else:
            x=np.where(~np.isnan(data))[1] #store the x coordinate of the non-masked pixels
            y=np.where(~np.isnan(data))[2] #store the y coordinate of the non-masked pixels
        for i in tqdm(zip(x,y),desc='Spectra clipped',total=len(x)): #run over the pixels
            spectrum=data[:,i[0],i[1]] #extract the spectrum
            peak=np.nanmax(spectrum) #define the peak of the gaussian as the maximum of the spectrum
            spectrum[spectrum<(peak*threshold)]=0 #clip the spectrum at the threshold of the max 
            clip_cube[:,i[0],i[1]]=spectrum.copy() #store the result in the clipped cube
        data=clip_cube.copy() #copy the clipped cube into the data cube
            
    #------------   CROP     ------------#
    if operation == 'crop': #if the datacube must be cropped
        xlim=np.where(~np.isnan(data))[2] #select the extreme x coordinates of non-NaN values
        ylim=np.where(~np.isnan(data))[1] #select the extreme y coordinates of non-NaN values
        data=data[:,np.min(ylim):np.max(ylim),np.min(xlim):np.max(xlim)] #crop the data
        if header is not None: #if the header is None means the input data is not a fits file
            wcs=WCS(header)[:,np.min(ylim):np.max(ylim),np.min(xlim):np.max(xlim)] #store the wcs
            newheader=wcs.to_header() #write the wcs into a header
            header['CRPIX1']=newheader['CRPIX1'] #update the header
            header['CRPIX2']=newheader['CRPIX2'] #update the header

    #------------   CUT     ------------#   
    if operation == 'cut': #if the datacube must be cutted
        if header is not None: #if the header is None means the input data is not a fits file
            if 'CRVAL3' in header and 'CDELT3' in header: #if the spectral keywords exist
                header['CRVAL3']=header['CRVAL3']+chanmin*header['CDELT3'] #recalculate the spectral axis 
            else:
                raise ValueError('ERROR: no spectral keywords in the header. Cannot recalculate the spectral axis: aborting') 
        data=data[chanmin:chanmax,ymin:ymax,xmin:xmax] #extract the  subcube
        wcs=WCS(header)[chanmin:chanmax,ymin:ymax,xmin:xmax] #store the wcs
        newheader=wcs.to_header() #write the wcs into a header
        header['CRPIX1']=newheader['CRPIX1'] #update the header
        header['CRPIX2']=newheader['CRPIX2'] #update the header
    
    #------------   EXTEND     ------------# 
    if operation == 'extend': #if the datacube must be cutted
        if value == 'blank': #if value is blank
            value=np.nan #set the value to nan
        if header is not None: #if the header is None means the input data is not a fits file
            if 'CRPIX3' in header: #if the spectral keyword exists
                header['CRPIX3']=header['CRPIX3']+abs(addchan) #recalculate the spectral axis
            else:
                raise ValueError('ERROR: no spectral keywords in the header. Cannot recalculate the spectral axis: aborting')
        if addchan < 0: #if the number of channels is less than 0
            newplane=np.ones((abs(addchan),data.shape[1],data.shape[2]))*value #create the new plane
            data=np.concatenate((newplane,data)) #concatenate the plane to the left
        else:
            newplane=np.ones((addchan,data.shape[1],data.shape[2]))*value #create the new plane
            data=np.concatenate((data,newplane)) #concatenate the plane to the left
    
    #------------   MIRROR     ------------# 
    if operation == 'mirror': #if the datacube must be mirrored
        if rot_center is None: #if no rotation center is given
            x0=data.shape[1]/2 #the center is the axis center
            y0=data.shape[2]/2 #the center is the axis center
            z0=data.shape[0]//2 #the center is the axis center
        else:
            x0=rot_center[0]-1 #store the x-axis rotation central pixel. -1 accounts for python 0-counting
            y0=rot_center[1]-1 #store the y-axis rotation central pixel. -1 accounts for python 0-counting
            z0=rot_center[2]-1 #store the z-axis rotation central channel. -1 accounts for python 0-counting
        for k in range(z0): #run over the channels
            for i in range(data.shape[1]): #run over the rows
                for j in range(data.shape[2]): #run over the columns
                    data[k,i,j]=data[2*z0-k,round(2*y0)-i-1,round(2*x0)-j-1] #do the mirroring
    
    #------------   MOM0     ------------# 
    if operation == 'mom0': #if the moment 0 map must be computed
        if header is not None: #if the header is None means the input data is not a fits file
            if spectralres is None: #if no spectral resolution is given
                if 'CDELT3' in header: #if the header has the starting spectral value
                    spectralres=header['CDELT3'] #store the starting spectral value
                else:
                    raise ValueError('ERROR: no spectral resolution was found: aborting')
            if spectralres < 0: #if the spectral resolution is negative
                spectralres=-spectralres #convert to positive
            if specunits is None: #if no spectral units are given
                if 'CUNIT3' in header: #if the spectral unit is in the header
                    specunits=header['CUNIT3'] #store the spectral unit
                else:
                    specunits='m/s' #set the first channel to 0 km/s
                    if verbose:
                        warnings.warn('No spectral unit was found: spectral unit set to m/s!')
            wcs=WCS(header,naxis=2) #store the WCS
            header=wcs.to_header() #convert the WCS into a header
            if 'BUNIT' in header: #if the spectral unit is in the header
                header['BUNIT']=header['BUNIT']+'*'+specunits #store the units of the moment 0 map
            else:
                header['BUNIT']=f'Jy/beam*{specunits}'
                if verbose:
                    warnings.warn(f'No flux unit was found: flux density unit set to Jy/beam*{specunits}!')
        else:
            if spectralres is None:
                raise ValueError('The spectral resolution is not provided. Aborting!')
        if usemask: #if a mask is used
            mask,_=__load(maskcube) #open the 3D mask
            if mask.shape[0] != data.shape[0] or mask.shape[1] != data.shape[1] or mask.shape[2] !=data.shape[2]: #if the mask cube has different size than the data cube
                raise ValueError(f'ERROR: mask cube and data cube has different shapes: {mask.shape} the mask cube and {data.shape} the data cube. Aborting!')
            mask[mask>1]=1 #convert to a binary mask
            data=data*mask #apply the mask
        data=np.nansum(data[chanmin:chanmax,:],axis=0)*spectralres #calculate the moment 0 map
        
    #------------   SHUFFLE     ------------# 
    if operation == 'shuffle': #if the cube must be shuffled
        nchan=data.shape[0] #store the number of channels
        cen=nchan//2 #define the central channel
        if header is not None: #if the header is None means the input data is not a fits file
            if spectralres is None: #if no spectral resolution is given
                if 'CDELT3' in header: #if the header has the starting spectral value
                    spectralres=header['CDELT3'] #store the starting spectral value
                else:
                    raise ValueError('ERROR: no spectral resolution was found: aborting')
            if v0 is None: #if no starting velocity is provided
                if 'CRVAL3' in header and 'CRPIX3' in header: #if the header has the starting spectral value
                    v0=header['CRVAL3']+(header['CRPIX3']-1)*np.abs(spectralres) #store the starting spectral value
                else:
                    raise ValueError('No starting channel value found: Aborting!')
            header['CRPIX3']=cen+1 #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred. +1 is needed for account the stupid python 0-counting
            header['CRVAL3']=0. #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred
        else:
            if spectralres is None:
                raise ValueError('The spectral resolution is not provided. Aborting!')
            if v0 is None:
                raise ValueError('The starting velocity of the spectral axis is not provided. Aborting!')
        mom1,_=__load(vfield) #store the velocity field
        shuffle_data=data.copy()*np.nan #initialize the shuffled cube
        v=np.arange(v0,v0+nchan*spectralres,spectralres) if spectralres>0 else np.flip(np.arange(v0+(nchan-1)*spectralres,v0-spectralres,-spectralres)) #define the spectral axis
        x=np.where(~np.isnan(mom1))[0] #store the non-NaN x coordinates
        y=np.where(~np.isnan(mom1))[1] #store the non-NaN y coordinates
        for i in tqdm(zip(x,y),desc='Spectra shuffled',total=len(x)): #run over the pixels
            if not np.all(np.isnan(data[:,i[0],i[1]])): #if the spectrum is not nan
                loc=np.argmin(abs(v-mom1[i[0],i[1]])) #define the spectral channel center of shuffle  
                shuffle_data[:,i[0],i[1]]=np.roll(data[:,i[0],i[1]],cen-loc,axis=0)
        data=shuffle_data.copy() #store the shuffle data
        
    #------------   TOINT     ------------#     
    if operation == 'toint': #if the datacube must be converted into an integer cube
        data[np.isnan(data)]=0 #set to 0 the nans
        data=data.astype(int) #convert the data to integer values
        
    if write_fits: #if the data must be write into a fits file
        hdu=fits.PrimaryHDU(data,header=header) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        hdul.writeto(output+'.fits',overwrite=True) #write the data into a fits file
        
    else:
        return data
                        
#############################################################################################
def cubestat(datacube='',nsigma=3,cs_statistics='mad',cs_fluxrange='negative',params_dict=None,**kwargs):
    """Calculate the detection limit of a data cube and optionally its root mean square (rms), spectral resolution, beam major axis, beam minor axis, beam position angle, and beam area. It also computes the errors on rms and sensitivity.

    Args:
        datacube (str or ndarray): The name or path+name of the FITS data cube.
        nsigma (float): The sigma of the detection limit in terms of nsigma*rms.
        cs_statistics (str): The statistic to be used in the noise measurement. Possible values are 'std' for standard deviation and 'mad' for median absolute deviation. Standard deviation is faster but less robust against emission and artifacts. Median absolute deviation is more robust in the presence of strong, extended emission or artifacts.
        cs_fluxrange (str): Flux range to be used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artifacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux.
        params_dict (dict): Dictionary to be updated with the information retreived by this function.

    Kwargs:
        path (str): The path to the data cube if 'datacube' is a name and not a path+name.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        fluxunits (str): The flux units. 
        pixelres (float): The cube spatial resolution in 'pixunits'. Default is None.
        spectralres (float): The cube spectral resolution in 'specunits'. Default is None.
        bmaj (float): The beam major axis in arcsec in 'pixunits'. Default is None.
        bmin (float): The beam minor axis in arcsec in 'pixunits'. Default is None.
        bpa (float): The beam position angle in degrees. Default is None.
        rms (float): The root mean square (rms) of the data cube in 'fluxunits'. Default is None.
        verbose (bool): If True, messages will be printed to the terminal. Default is False.

    Returns:
        dict: A Python dictionary with the cube statistics.

    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no path is provided.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    #CHECK THE KWARGS#
    pixunits=params_dict.get('pixunits',None) #store the spatial units from the optional dictionary
    if pixunits is None: #if not given
        pixunits=kwargs.get('pixunits',None) #store the spatial units from the kwargs
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    specunits=params_dict.get('specunits',None) #store the spectral units from the optional dictionary
    if specunits is None: #if not given
        specunits=kwargs.get('specunits',None) #store the spectral units from the kwargs
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')   
    fluxunits=params_dict.get('fluxunits',None) #store the flux units from the optional dictionary
    if fluxunits is None: #if not given
        fluxunits=kwargs.get('fluxunits',None) #store the flux units from the kwargs
    pixelres=params_dict.get('pixelres',None) #store the spatial resolution from the optional dictionary
    if pixelres is None: #if not given
        pixelres=kwargs.get('pixelres',None) #store the spatial resolution from the kwargs
    spectralres=params_dict.get('spectralres',None) #store the spectral resolution from the optional dictionary
    if spectralres is None: #if not given
        spectralres=kwargs.get('spectralres',None) #store the spectral resolution from the kwargs
    bmaj=params_dict.get('bmaj',None) #store the beam major axis from the optional dictionary
    if bmaj is None: #if not given
        bmaj=kwargs.get('bmaj',None) #store the beam major axis from the kwargs
    bmin=params_dict.get('bmin',None) #store the beam minor axis from the optional dictionary
    if bmin is None: #if not given
        bmin=kwargs.get('bmin',None) #store the beam minor axis from the kwargs
    bpa=params_dict.get('bpa',None) #store the beam position angle from the optional dictionary
    if bpa is None: #if not given
        bpa=kwargs.get('bpa',None) #store the beam position angle from the kwargs
    rms=params_dict.get('rms',None) #store the rms from the optional dictionary
    if rms is None: #if not given
        rms=kwargs.get('rms',None) #store the rms from the kwargs
    
    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #open the data cube
    
    #CHECK WHICH INFORMATION ARE ALREADY GIVEN#
    #------------   SPATIAL UNITS     ------------#
    if pixunits is None and 'CUNIT1' in header: #if no spatial units are given and the keyword is in the header
        pixunits=header['CUNIT1']#store the spatial units from the cube header
    elif pixunits is None and 'CUNIT2' in header: #if no spatial units are given and the keyword is in the header
        pixunits=header['CUNIT2']#store the spatial units from the cube header
    elif pixunits is None:
        pixunits='deg' #set them to deg
        if verbose:
            warnings.warn('No spatial units were found: spatial units set to deg!')
    #------------   SPECTRAL UNITS     ------------#
    if specunits is None and 'CUNIT3' in header: #if no spectral units are given and the keyword is in the header
        specunits=header['CUNIT3'] #store the spectral units from the cube header
    elif specunits is None:
        specunits='m/s' #set the first channel to 0 km/s
        if verbose:
            warnings.warn('No spectral unit was found: spectral unit set to m/s!')
    #------------   FLUX UNITS     ------------#        
    if fluxunits is None and 'BUNIT' in header: #if no spectral units are given and the keyword is in the header
        fluxunits=header['BUNIT'] #store the spectral units from the cube header
    elif fluxunits is None:
        fluxunits='Jy/beam' #set the first channel to 0 km/s
        if verbose:
            warnings.warn('No flux unit was found: flux unit set to Jy/beam!')
    #------------   SPATIAL RESOLUTION     ------------#
    if pixelres is None and 'CDELT1' in header: #if the spatial resolution is not given and the keyword is in the header
        pixelres=header['CDELT1'] #store the spatial resolution from the cube header
    elif pixelres is None and 'CDELT2' in header: #if the spatial resolution is not given and the keyword is in the header
        pixelres=header['CDELT2'] #store the spatial resolution from the cube header
    elif pixelres is None:
        if verbose:
            warnings.warn('No spatial resolution was found: please set it manually in the parameter file/dictionary!')
    #------------   SPECTRAL RESOLUTION     ------------#
    if spectralres is None and 'CDELT3' in header: #if the spectral resolution is not given and the keyword is in the header
        spectralres=header['CDELT3'] #store the spectral resolution from the cube header
    elif spectralres is None:
        if verbose:
            warnings.warn('No spectral resolution was found: please set it manually in the parameter file/dictionary!')
    #------------   BEAM     ------------#
    if bmaj is None and 'BMAJ' in header: #if the beam major axis is not given and the keyword is in the cube header
        bmaj=header['BMAJ'] #get the value from the cube header
    elif bmaj is None:
        if verbose:
            warnings.warn('No beam major axis was found: please set it manually in the parameter file/dictionary!')
    if bmin is None and 'BMIN' in header: #if the beam minor axis is not given and the keyword is in the cube header
        bmin=header['BMIN'] #get the value from the cube header
    elif bmin is None:
        if verbose:
            warnings.warn('No beam minor axis was found: please set it manually in the parameter file/dictionary!')
    if bpa is None and 'BPA' in header: #if the beam major axis is not given and the keyword is in the cube header
        bpa=header['BPA'] #get the value from the cube header
    elif bpa is None:
        if verbose:
            warnings.warn('No beam position angle was found: please set it manually in the parameter file/dictionary!')
    #------------   RMS     ------------#    
    if rms is None: #if the rms is not given
        rms=__rms(data,cs_statistics,cs_fluxrange) #calculate the rms
            
    #CALCULATE THE DETECTION LIMIT#
    if bmaj is not None and bmin is not None: #if the beam axes are available
        beamarea=1.13*(bmin*bmaj) #calculate the beam area
        sensitivity=converttoHI(nsigma*rms,fluxunits=fluxunits,beamarea=beamarea,pixunits=pixunits,spectralres=spectralres,specunits=specunits) #the detection limit is the rms
    else:
        if verbose:
            warnings.warn('Unable to calculate the beam area. Cannot compute the sensitivity')
                
    #UPDATE THE INPUT PARAMETER DICTIONARY#
    if params_dict is not None: #if an input parameter dictionary is provided
        if 'pixunits' in params_dict and params_dict['pixunits'] is None: params_dict['pixunits']=pixunits #update the pixel units if are in the input parameter dictionary but not set
        if 'specunits' in params_dict and params_dict['specunits'] is None: params_dict['specunits']=specunits #update the spectral units if are in the input parameter dictionary but not set
        if 'fluxunits' in params_dict and params_dict['fluxunits'] is None: params_dict['fluxunits']=fluxunits #update the flux units if are in the input parameter dictionary but not set
        if 'pixelres' in params_dict and params_dict['pixelres'] is None: params_dict['pixelres']=pixelres #update the pixel resolution if is in the input parameter dictionary but not set
        if 'spectralres' in params_dict and params_dict['spectralres'] is None: params_dict['spectralres']=spectralres #update the spectral resolution  
        if 'bmaj' in params_dict and params_dict['bmaj'] is None: params_dict['bmaj']=bmaj #update the beam major axis if is in the input parameter dictionary but not set
        if 'bmin' in params_dict and params_dict['bmin'] is None: params_dict['bmin']=bmin #update the beam minor axis if is in the input parameter dictionary but not set
        if 'bpa' in params_dict and params_dict['bpa'] is None: params_dict['bpa']=bpa #update the beam position angle if is in the input parameter dictionary but not set
        if 'rms' in params_dict and params_dict['rms'] is None: params_dict['rms']=rms #update the rms if is in the input parameter dictionary but not set
    
    #PRINT THE INFORMATION TO THE TERMINAL#
    if verbose: #if print-to-terminal option is true
        name=datacube.split('/')[-1] #extract the cube name from the datacube variable
        coldenunits='cm\u207B\u00B2' #units of the column density
        print(f'Noise statistic of the cube {name}:')
        print(f'The median rms per channel is: {rms:.1e} {fluxunits}')
        if bmaj is not None and bmin is not None: #if the beam axes are available
            print(f'The {int(nsigma)}\u03C3 1-channel detection limit is: {nsigma*rms:.1e} {fluxunits} i.e., {sensitivity:.1e} {coldenunits}')
            
    if params_dict is not None: #if an input parameter dictionary is provided    
        return params_dict

#############################################################################################
def fitsarith(path='',fits1='',fits2='',do=None,fits_out='',**kwargs):
    """Perform arithmetic operations (sum, subtraction, multiplication, division) between two FITS files.
    
    Args:
        path (str): Path to the FITS files.
        fits1 (str): Name or path+name of the first FITS file.
        fits2 (str): Name or path+name of the second FITS file.
        do (str): Operation to perform:
            - 'sum': Sum the two FITS files.
            - 'sub': Subtract the second FITS file from the first FITS file.
            - 'mul': Multiply the two FITS files.
            - 'div': Divide the first FITS file by the second FITS file (zeros in the second FITS file will be blanked).
        fits_out (str): The output folder/file name.
    
    Returns:
        str: Path to the resulting FITS file.
    
    Raises:
        ValueError: If one or both FITS files are not provided.
        ValueError: If no operation is specified.
        ValueError: If the specified operation is not supported.
        ValueError: If no output name is given.
        ValueError: If the shapes of the two FITS files do not match.
    """
    #CHECK THE INPUT#
    if path == '' or path is None: #if the path to the fits files is not given
        path=os.getcwd()+'/'
    if fits1 == '' or fits1 is None or fits2 == '' or fits2 is None:
        raise ValueError('ERROR: one or both fits file are not set: aborting')
    if fits1[0] != '.': #if the first fits name start with a . means that it is a path to the fits (so differs from path parameter)
        fits1=path+fits1
    if fits2[0] != '.': #if the second fits name start with a . means that it is a path to the fits (so differs from path parameter)
        fits2=path+fits2
    if do is None: #if no operation is set
        raise ValueError('ERROR: no operation set: aborting')
    if do not in ['sum','sub','mul','div']: #if wrong operation is given
        raise ValueError("ERROR: wrong operation. Accepted values: ['sum','sub','mul','div']. Aborting")
    output=fits_out #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
 
    #---------------   START THE FUNCTION   ---------------#
    data1,header=__load(fits1) #open the first fits file
    data2,_=__load(fits2) #open the second fits file
            
    #CHECK THAT THE TWO DATASET HAVE SAME DIMENSIONS#
    if len(data1.shape) != len(data2.shape): #if the two fits file have different dimensions
        raise ValueError(f'!ERROR! The two fits files have different dimensions {data1.shape} and {data2.shape}. Aborting!!!')
    for i in range(len(data1.shape)): #for each dimension of the fits
        if data1.shape[i] != data2.shape[i]: #if the two fits file have different shapes
            raise ValueError(f'!ERROR! The two fits files have different {i}-dimension shapes ({data1.shape[i]} and {data2.shape[i]}). Aborting!!!')
            
    #DO THE SELECTED OPERATION#
    data=data1+data2 if do == 'sum' else data1-data2 if do == 'sub' else data1*data2 if do == 'mul' else data1/data2 #do the requested operation
    
    #WRITE THE OUTPUT#    
    hdu=fits.PrimaryHDU(data.astype('float32'),header=header) #create the primary HDU
    hdul=fits.HDUList([hdu]) #make the HDU list
    hdul.writeto(output+'.fits',overwrite=True) #write the data into a fits file
        
#############################################################################################
def fixmask(refcube='',masktofix='',fix_out='',**kwargs):
    """Fix a 3D detection mask by setting the voxels corresponding to negative detections to 0, based on a reference data cube. It is assumed that a value > 0 in the mask indicates a detection.
    
    Args:
        refcube (str/ndarray): Name or path+name of the FITS data cube used as the reference for fixing the mask.
        masktofix (str): Name or path+name of the FITS 3D mask to be fixed.
        fix_out (str): The output folder/file name.
    
    Kwargs:
        datacube (str/ndarray): Name or path+name of the FITS data cube, used if refcube is not provided.
        maskcube (str/ndarray): Name or path+name of the FITS 3D mask cube, used when chanmask=True.
        path (str): Path to the data cube if datacube is specified as a name and not a path+name.
        verbose (bool): Option to print messages to the terminal if set to True. (Default: False)
    
    Returns:
        The fixed mask as a FITS file.
    
    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no path is provided.
        ValueError: If no mask cube is provided.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    datacube=__read_string(refcube,'datacube',**kwargs) #store the path to the data cube
    maskcube=__read_string(masktofix,'maskcube',**kwargs) #store the path to the mask cube
    output=fix_out #store the output directory/name from the input parameters
    if output==masktofix or output in ['',None]: #if the output is the same of the input mask or not given
        mode='update' #set the fits open mode to 'update'
    elif output[0]=='.': #if the output name start with a . means that it contains a path
        mode='readonly' #set the fits open mode to 'readonly'
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist     
    else: #if the output name does not contain a path, nor is equal to the input mask, nor is empty
        mode='readonly' #set the fits open mode to 'readonly'
        
    #---------------   START THE FUNCTION   ---------------#
    data,_=__load(datacube) #open the data cube
    with fits.open(maskcube,mode=mode) as Maskcube: #open the mask cube
        mask=Maskcube[0].data.copy() #store the mask data
        mask[data<0]=0 #fix the mask by setting to 0 the pixel where the emission is negative (hence, no source)
        Maskcube[0].data=mask #overwrite the mask data
        if mode == 'readonly': #if the open mode is read only
            Maskcube.writeto(output+'.fits',overwrite=True) #write the new mask
    del Maskcube[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
        
#############################################################################################
def gaussfit(cubetofit='',gaussmask='',components=1,linefwhm=15,amp_thresh=0,p_reject=1,
             clipping=False,clip_threshold=0.5,errors=False,write_field=False,write_fits=False,gauss_out='',**kwargs):
    """Perform a Gaussian fit on a spectral cube. This function fits a single Gaussian profile to each spaxel (spatial pixel) in the spectral cube.
    
    Args:
        cubetofit (str): Name or path+name of the FITS data cube to be fitted.
        gaussmask (str): Name or path+name of the FITS 2D mask to be used in the fit.
        components (int): Number of components (1 or 2) to fit.
        linefwhm (float): Initial guess for the full width at half maximum (FWHM) of the line profile in km/s.
        amp_thresh (float): Amplitude threshold for the fit. If a profile peak is below the threshold, the fit will not be performed on that spectrum.
        p_reject (float): p-value threshold for fit rejection. If a best-fit has p-value greater than p_reject, it will be rejected.
        clip_threshold (bool): If True, clip the spectrum to a percentage of the profile peak.
        threshold (float): Clip threshold as a percentage of the peak (0.5 represents 50%). This parameter is used only if clipping is True.
        errors (bool): If True, compute the errors on the best-fit.
        write_field (bool): If True, compute the velocity field.
        write_fits (bool): If True, store the output in a FITS file; if False, return a variable.
        gauss_out (str): The output folder/file name
    
    Kwargs:
        datacube (str/ndarray): Name or path+name of the FITS data cube if cubetofit is not given.
        mask2d (str): Name or path+name of the FITS 2D mask if gaussmask is not given.
        path (str): Path to the data cube if the datacube is a name and not a path+name.
        specunits (str): String specifying the spectral units for operations mom0 and shuffle. Default: "m/s". Accepted values:
            - 'km/s'
            - 'm/s'
            - 'Hz'
        spectralres (float): Data spectral resolution in specunits. This value will be taken from the cube header if not provided.
        verbose (bool): If True, print messages to the terminal. Default: False.
    
    Returns:
        Best-fit model cube as a FITS file.
    
    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no path is provided.
        ValueError: If the mask cube and data cube dimensions do not match.
        ValueError: If no spectral information is available through the input arguments or the cube header.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    datacube=__read_string(cubetofit,'datacube',**kwargs) if type(cubetofit)==str or cubetofit is None else cubetofit #store the path to the data cube if the datacube is a string
    mask2d=__read_string(gaussmask,'mask2d',**kwargs) if gaussmask is not None and (type(gaussmask) is str and gaussmask != '') else gaussmask #if a 2D mask is provided
         #store the path to the 2D mask if the 2D mask is a string
    output=gauss_out #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    #CHECK THE KWARGS#
    specunits=kwargs.get('specunits',None) #store the spectral units
    spectralres=kwargs.get('spectralres',None) #store the spectral resolution
    v0=kwargs.get('v0',None) #store the starting velocity from the input paramaters
    cs_statistics=kwargs.get('cs_statistics','mad') #store the statistics to be use for rms calculation
    
    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #open the data cube
    if mask2d == '' or mask2d is None: #if no mask is given
        mask=np.ones(data.shape[1:3]) #create a dummy mask
    else:
        mask,_=__load(mask2d) #open the mask
        mask[mask!=0]=1 #set to 1 the non-nan voxel
        if mask.shape != data.shape[1:3]: #if the mask shape is not the same of the data cube
            mask=np.ones(data.shape[1:3]) #create a dummy mask
            if verbose:
                warnings.warn(f'Mask spatial shape {mask.shape} and data spatial shape {data.shape[1:3]} mismatch. Cannot apply the detection mask; the fit will be done over the whole field of view.')
    model_cube=np.zeros(data.shape) #initialize the model cube as zeros
    if write_field: #if the velocity field must be computed
        field=np.empty(data.shape[1:3])*np.nan #initialize the velocity field
        
    #CHECK FOR THE RELEVANT INFORMATION#
    #------------   CUBE PROPERTIES    ------------#
    if header is not None:
        prop=[specunits,spectralres] #list of cube properties values
        prop_name=['specunits','spectralres'] #list of cube properties names
        if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if one or more cube parameters are not given
            if verbose:
                not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
                warnings.warn(f'I am missing some information: {not_found}. Running cubestat to retrieve them!')
            stats=cubestat(datacube,params_dict=dict(zip(prop_name,prop)),cs_statistics=cs_statistics,verbose=False) #calculate the statistics of the cube
            if specunits is None: #if spectral units are not given
                specunits=stats['specunits'] #take the value from the cubestat results
            if spectralres is None: #if spectral resolution is not given
                spectralres=stats['spectralres'] #take the value from the cubestat results
        prop=[specunits,spectralres] #update the list of cube properties values
        if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if still one or more cube parameters are not given
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            raise ValueError(f'I am still missing some information: {not_found}. I cannot proceed with the fitting!')
    elif spectralres is None: #if the spectral resolution is not provided
        raise ValueError('ERROR: no spectral resolution provided. Aborting!')    
    if header is not None: #if the header is None means the input data is not a fits file    
        if 'CRVAL3' in header and 'CRPIX3' in header: #if the header has the starting spectral value
            v0=header['CRVAL3']-(header['CRPIX3']-1)*spectralres #store the starting spectral value
        else:
            raise ValueError('ERROR: no spectral value for starting channel was found. Aborting!')
    elif v0 is None: #if the starting channel value is not provided
        raise ValueError('ERROR: no spectral value for starting channel provided. Aborting!')
        
    if specunits == 'm/s': #if the spectral units are m/s
        spectralres/=1000 #convert the spectral resolution to km/s
        v0/=1000 #convert the starting velocity to km/s
        
    #PREPARE THE SPECTRAL AXIS#
    nchan=data.shape[0] #store the number of channels
    v=np.arange(v0,v0+nchan*spectralres,spectralres) if spectralres>0 else np.flip(np.arange(v0+(nchan-1)*spectralres,v0-spectralres,-spectralres)) #define the spectral axis
    if len(v) > nchan: #!! sometimes an additional channel is created. For the moment, this is a workaround
        v=v[:-1]
        
    #PREPARE FOR THE FIT#
    width=linefwhm/np.abs(spectralres) #define the first guess fwhm of the line in km/s
    x=np.where(mask>0)[0] #store the x coordinate of the non-masked pixels
    y=np.where(mask>0)[1] #store the y coordinate of the non-masked pixels
    
    #START THE FITTING ROUTINE#
    if verbose:
        print('Starting the Gaussian fit with the following parameters:\n'
        f'Spectral resolution: {np.abs(spectralres)} km/s\n'
        f'Starting velocity: {v0} km/s\n'
        f'First-guess FWHM: {linefwhm} km/s\n'
        f'Amplitude threshold: {amp_thresh}\n'
        f'p-value for rejection: {p_reject}')
      
    for i in tqdm(zip(x,y),desc='Spectra fitted',total=len(x)): #run over the pixels
        spectrum=data[:,i[0],i[1]].copy() #extract the spectrum
        peak=np.nanmax(spectrum) #define the peak of the gaussian as the maximum of the spectrum        
        if peak > amp_thresh: #if the peak is above the threshold
            vpeak=v[np.nanargmax(spectrum)] #define the central velocity as the velocity of the peak
            if clipping: #if data must be clipped
                v_fit=v[np.where(spectrum>=peak*clip_threshold)] #extract the velocities of the clipped data
                spectrum=spectrum[np.where(spectrum>=peak*clip_threshold)] #clip the data
            else:
                v_fit=v[np.where(~np.isnan(spectrum))] #extract the velocities corresponding to non-nan in the spectrum
                spectrum=spectrum[np.where(~np.isnan(spectrum))] #extract the not-nan values
            if components==1: #if only one component is fitted
                model=models.Gaussian1D(amplitude=peak,mean=vpeak,stddev=width*2.355) #built the gaussian model
                model.amplitude.bounds=(0,2*peak) #bound the amplitude of the fit !!! This tolerance might be set as input !!!
                model.mean.bounds=(vpeak-2*np.abs(spectralres),vpeak+2*np.abs(spectralres)) #bound the velocity of the fit to be +- 2 channels. !!! This tolerance might be set as input !!!
            else:
                model=models.Gaussian1D(amplitude=peak,mean=vpeak,stddev=width*2.355)+models.Gaussian1D(amplitude=peak/2,mean=vpeak,stddev=width*2.355) #built the gaussian model
                model.amplitude_0.bounds=(0,2*peak) #bound the amplitude of the fit !!! This tolerance might be set as input !!!
                model.mean_0.bounds=(vpeak-2*np.abs(spectralres),vpeak+2*np.abs(spectralres)) #bound the velocity of the fit to be +- 2 channels. !!! This tolerance might be set as input !!!
                model.amplitude_1.bounds=(0,peak) #bound the amplitude of the fit !!! This tolerance might be set as input !!!
            fitter=fit(calc_uncertainties=errors) #define the fitter
            if len(spectrum)>3*components: #fit only if you have at least 3 data points per model
                total_fit=fitter(model,v_fit,spectrum) #do the fit. Weights are 1/error. They are used only if calc_uncertainties=True
                v_fit=v_fit[np.where(spectrum!=0)] #extract the velocities corresponding to non-zero in the spectrum
                spectrum=spectrum[np.where(spectrum!=0)] #extract the non-zero values
                chi2=np.sum(((spectrum-total_fit(v_fit))**2)/(spectrum/10)**2) #calculate the chi2
                dof=len(spectrum-3) #calculate the degrees of freedom
                p_value=statchi2.cdf(chi2,dof) #calculate the p-value
                if p_value<=p_reject: #if the p-value is less than p
                    model_cube[:,i[0],i[1]]=total_fit(v) #store the result in the model cube
                    if write_field and components==1: #if the velocity field must be computed and only one component is fitted
                        field[i[0],i[1]]=total_fit.mean.value #store the best-fit peak value
                    elif write_field and components==2: #if the velocity field must be computed and two components are fitted
                        field[i[0],i[1]]=total_fit.mean_0.value if total_fit.amplitude_0.value > total_fit.amplitude_1.value else total_fit.mean_1.value #use the best-fit peak velocity of the brightest peak
    
    if write_field: #if the velocity field must be computed
        wcs=WCS(header).dropaxis(2) #store the wcs
        h=wcs.to_header() #convert the wcs into a header
        h['BUNIT']=(specunits,'Best-fit peak velocities') #add the BUNIT keyword
        if specunits == 'm/s': #if the spectral units are m/s
            field=field*1000 #reconvert the velocity into m/s
        hdu=fits.PrimaryHDU(field.astype('float32'),header=h) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        hdul.writeto(output.replace('.fits','_vfield.fits'),overwrite=True) #save the velocity field
          
    if write_fits: #if the data must be write into a fits file
        hdu=fits.PrimaryHDU(model_cube.astype('float32'),header=header) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        hdul.writeto(output+'.fits',overwrite=True) #write the data into a fits file
    else:
        return model_cube.astype('float32')
            
#############################################################################################
def getpv(pvcube='',pvwidth=None,pvpoints=None,pvangle=None,pvchmin=1,pvchmax=None,pv_statistics='mad',pv_fluxrange='negative',pv_out='',write_fits=False,plot=False,saveplot=False,**kwargs):
    """Extract a position-velocity slice from a given data cube along a path defined by specified points, angle, and width. Optionally, it can plot the slice and save it to a FITS file.
    
    Args:
        pvcube (str/ndarray): Name or path+name of the FITS data cube.
        pvwidth (float): Width of the slice in arcseconds. If not provided, it will be set to the beam size.
        pvpoints (array-like): ICRS RA-DEC comma-separated coordinates of the slice points in decimal degrees. If only two points are given ([x, y]), they are assumed to be the center of the slice. Otherwise, the points need to be the starting and ending coordinates ([xmin, xmax, ymin, ymax]).
        pvangle (float): Position angle of the slice in degrees when two points are given. If not provided, it will use the object position angle.
        pvchmin (int): First channel of the slice.
        pvchmax (int): Last channel of the slice.
        pv_statistics (str): Statistic to be used in the noise measurement. Possible values are 'std' for standard deviation or 'mad' for median absolute deviation. The standard deviation algorithm is faster but less robust against emission and artifacts in the data. The median absolute deviation algorithm is more robust in the presence of strong, extended emission or artifacts.
        pv_fluxrange (str): The flux range used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artifacts from affecting the noise measurement. If set to any other value, all pixels will be used in the noise measurement irrespective of their flux.
        pvoutdir (str): Output folder name as a string.
        write_fits (bool): Saves the slice as a FITS file if True.
        fitsoutname (str): Output FITS file name as a string.
        plot (bool): Plots the slice if True.
        pv_out (str): The output folder/file name.
        
    Kwargs:
        datacube (str/ndarray): Name or path+name of the FITS data cube if pvcube is not provided.
        path (str): Path to the data cube if the datacube is a name and not a path+name.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        pixelres (float): Cube spatial resolution in pixunits (Default: None).
        bmaj (float): Beam major axis in arcseconds.
        pa (float): Object position angle in degrees.
        pvsig (float): Lowest contour level in terms of pvsig*rms.
        vsys (float): Object systemic velocity in m/s.
        rms (float): RMS of the data cube in Jy/beam as a float. If not provided (None), the function tries to calculate it.
        asectokpc (float): Arcseconds to kiloparsecs conversion factor to plot the spatial scale.
        lim (array-like): List or array of plot x and y limits as [xmin, xmax, ymin, ymax]. These values will replace the default limits.
        contours (array-like): Contour levels in units of rms. These values will replace the default levels (Default: None)
        ctr_width (float): Line width of the contours.
        cmap (str/ndarray): Name of the colormap to be used. Accepted values are those of matplotlib.colors.Colormap. (Default: 'Greys')
        ctrmap (str/ndarray): Name of the colormap to be used for the contour levels. Accepted values are those of matplotlib.colors.Colormap. (Default: 'hot').
        left_space (float): Width of the left margin in axis units (from 0 to 1) (Default: 0.025).
        upper_space (float): Width of the upper margin in axis units (from 0 to 1) (Default: 0.025).
        lower_space (float): Width of the lower margin in axis units (from 0 to 1) (Default: 0.975).
        position (int): Position of the subplot in the figure as a triplet of integers (e.g., 111 = nrow 1, ncol 1, index 1).
        plot_format (str): File format of the plots (Default: pdf).
        verbose (bool): Prints messages and plots to the terminal if True.
    
    Returns:
        Position-velocity (PV) slice of a cube as a FITS file or PrimaryHDU object.
    
    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no path is provided.
        ValueError: If no starting and/or ending points are given.
        ValueError: If the number of given points is incorrect.
        ValueError: If the points are not given as a list or tuple.
        ValueError: If no position angle is provided.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf') #store the plot output format
    if type(pvcube)==str or pvcube is None: #if the datacube is a string
        datacube=__read_string(pvcube,'datacube',**kwargs) #store the path to the data cube
    else: #the datacube is an array
        datacube=pvcube
    if pvwidth is None: #if width is not given
        if verbose:
            warnings.warn('Pv width not set. Trying to set it to the beam major axis!')
        pvwidth=kwargs.get('bmaj',None) #set it to the beam major axis
    if pvpoints is None: #if no coordinates are given for the center
        raise ValueError('ERROR: No points provided for the pv slice: aborting!')
    if type(pvpoints) != list:
        raise ValueError('ERROR: Pv points should be given as a list of tuples [(x,y)] or [(xmin,ymin),(xmax,ymax)]: aborting!')
    if len(pvpoints)==2: #if two points are given, it is interpreted as the coordinates of the slice center
        from_center=True #extract the slice from its center
        pvpoints=SkyCoord(ra=pvpoints[0]*u.degree,dec=pvpoints[1]*u.degree,frame='icrs') #define the points in icsr coordinates
    elif (len(pvpoints)%2) ==0: #if even points are given it is interpreted as a set of points along which the slice will be extracted
        from_center=False #extract the slice between them
        ra=[] #initialize the ra list
        dec=[] #initialize the dec list
        for i in range(0,len(pvpoints),2): #run over the path points
            ra.append(pvpoints[i])
            dec.append(pvpoints[i+1])
        pvpoints=ICRS(ra*u.degree,dec*u.degree) #convert the points into ICRS coordinates
    else:
        raise ValueError(f'ERROR: wrong number of coordinates given {len(pvpoints)}, required an even number. Aborting!')
    if from_center: #if the slice is defined from its center
        if pvangle is None: #if angle is not given
            pvangle=kwargs.get('pa',None) #set it to the position angle of the galaxy
            if pvangle is None: #if the position angle is not given
                raise ValueError('ERROR: no position angle is set: aborting!')
        pvangle=(180+pvangle)*u.degree #convert it due to position angle definition of pvextractor
    chmin=pvchmin #store the lower channel from the input parameters
    chmax=pvchmax #store the upper channel from the input parameters
    output=pv_out+plot_format #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
        
    #CHECK THE KWARGS#
    pixunits=kwargs.get('pixunits',None) #store the spatial units
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')   
    pixelres=kwargs.get('pixelres',None) #store the spatial resolution
    rms=kwargs.get('rms',None) #store the rms
    vsys=kwargs.get('vsys',None) #store the systemic velocity
    if vsys is None: #if not given
        vsys=0 #set it to 0
        warnings.warn('No systemic velocity is given: set it to 0 m/s!')
    cs_statistics=kwargs.get('cs_statistics','mad') #store the statistics to be use for rms calculation
    asectokpc=kwargs.get('asectokpc',None) #store the arcseconds to kpc conversion
    ctr=kwargs.get('contours',None) #store the contour levels
    sigma=kwargs.get('pvsig',3) #store the lowest contour in terms of rms*sigma
    ctr_width=kwargs.get('ctr_width',2) #store the contour width
    lim=kwargs.get('lim',None) #store the plot limits
    cmap=kwargs.get('cmap','Greys') #store the colormap
    ctrmap=kwargs.get('ctrmap','hot') #store the contours colormap
    left_space=kwargs.get('left_space',0.025) #store the left margin for annotations position
    lower_space=kwargs.get('lower_space',0.025) #store the bottom margin for annotations position
    upper_space=kwargs.get('upper_space',0.975) #store the top margin for annotations position
    position=kwargs.get('position',None) #store the position of the subplot

    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #open the data cube
    wcs=WCS(header) #store the wcs information
    
    #CHECK FOR THE RELEVANT INFORMATION#
    #------------   CUBE PROPERTIES    ------------#
    if header is not None:
        prop=[pvwidth,pixunits,pixelres,rms] #list of cube properties values
        prop_name=['bmaj','pixunits','pixelres','rms'] #list of cube properties names
        if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if one or more cube parameters are not given
            if verbose:
                not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
                warnings.warn(f'I am missing some information: {not_found}. Running cubestat to retrieve them!')
            stats=cubestat(datacube,params_dict=dict(zip(prop_name,prop)),cs_statistics=cs_statistics,verbose=False) #calculate the statistics of the cube
            if pvwidth is None: #if the width of the slice is not given
                pvwidth=stats['bmaj'] #take the value from the cubestat results
            if pixunits is None: #if the spatial units are not given
                pixunits=stats['pixunits'] #take the value from the cubestat results
            if pixelres is None: #if the spatial resolution is not given
                pixelres=stats['pixelres'] #take the value from the cubestat results
            if rms is None: #if the rms is not given
                rms=stats['rms'] #take the value from the cubestat results
        prop=[pixelres] #update the list of cube properties values
        if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if still one or more cube parameters are not given
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            raise ValueError(f'I am still missing some information: {not_found}. I cannot proceed!')
    elif pixelres is None: #if the spectral resolution is not provided
            raise ValueError('ERROR: no spatial resolution provided. Aborting!')
    if pvwidth is None: #if also the beam major axis is not given
        warnings.warn('Beam major axis not set. Pv width will be 1 pixel!')
        pvwidth=1 #set it to 1 pixel
    if pixunits is None:
        pixunits='deg' #set them to deg
        if verbose:
            warnings.warn('No spatial units were found: spatial units set to deg!')
     
    #CONVERT THE VALUES IN STANDARD UNITS#        
    if pixunits == 'deg': #if the spatial unit is degree
        pixelres*=3600 #convert into arcsec
        pvwidth*=3600*u.arcsec #convert into arcsec
    elif pixunits == 'arcmin': #if the spatial unit is arcmin
        pixelres*=60 #convert into arcsec
        pvwidth*=60*u.arcsec #convert into arcsec
    elif pixunits == 'arcsec': #if the spatial unit is arcsec
        pvwidth*=u.arcsec #do nothing 
    if pixelres < 0: #if the pixel resolution is negative
        pixelres=-pixelres #convert to positive  
        
    if from_center: #if the slice type is from the center
        length=np.sqrt((data.shape[0]*pixelres)**2+(data.shape[1]*pixelres)**2)*u.arcsec #define the length of the slice as the size of the image
        pvpath=PathFromCenter(center=pvpoints,length=length,angle=pvangle,width=pvwidth) #define the path for the pv slice
    else:
        pvpath=Path(pvpoints,width=pvwidth) #define the path for the pv slice  
        
    #CHECK THE CHANNELS#
    if chmin is None or chmin < 1: #if the lower channel is not given or less than 1
        warnings.warn(f'Starting channel wrongly set. You give {chmin} but should be at least 1. Set to 1')
        chmin=1 #set it to 1
    if chmax is None or chmax > data.shape[0]: #if the upper channel is not set or larger than the data size
        warnings.warn(f'Last channel wrongly set. You give {chmax} but should be at maximum {data.shape[0]}. Set to {data.shape[0]}.')
        chmax=data.shape[0]+1 #select until the last channel
    if chmax < chmin: #if the higher channel is less than the lower
        warnings.warn(f'Last channel ({chmax}) lower than starting channel ({chmin}). Last channel set to {data.shape[0]}.')
        chmax=data.shape[0]+1 #select until the last channel
    chmin-=1 #remove 1 from the first channel to account for python 0-counting. We do not have to do this for the chmax as the indexing is exclusive, meaning that if chmax=10, the operation will be done until pythonic channel 9, which is channel 10 when 1-counting
    
    if verbose:
        if from_center:
            print('Extracting the pv slice with the following parameters:\n'
            f'Spatial resolution: {pixelres:.1f} arcsec\n'
            f'Path length: {length:.1f}\n'
            f'Path position angle: {pvangle:.1f} deg\n'
            f'Path width: {pvwidth:.1f}\n'
            f'From channel {chmin+1} to channel {chmax-1}\n'
            '-------------------------------------------------')
        else:
            print('Extracting the pv slice with the following parameters:\n'
            f'Spatial resolution: {pixelres:.1f} arcsec\n'
            f'Path width: {pvwidth:.1f}\n'
            f'From channel {chmin+1} to channel {chmax-1}\n'
            '-------------------------------------------------')
            
    #EXTRACT THE PVSLICE AND REFER THE SPATIAL AXIS TO THE SLICE CENTER#    
    pv=extract_pv_slice(data[chmin:chmax,:,:],pvpath,wcs=wcs[chmin:chmax,:,:]) #extract the pv slice
    pv.header['CRPIX1']=round(pv.header['NAXIS1']/2)+1 #fix the header in order to have the distance from the center as spatial dimension
    
    #SAVE THE SLICE IF NEEDED#
    if write_fits: #if the slice must be saved
        pv.writeto(output.replace(plot_format,'.fits'),overwrite=True) #write the pv as fits file
    
    #DO THE PLOT IF NEEDED#
    if plot: #if the slice must be plotted
        #PREPARE THE DATA#
        data=pv.data #store the pv data
        if rms is None: #if the rms is not given
            if verbose:
                warnings.warn('No rms is given: calculating the rms on the pv data')
            rms=__rms(data,pv_statistics,pv_fluxrange) #calculate from the data
        #PREPARE THE FIGURE#
        if position is None: #if a plot position is not given, means the figure must be created
            nrows=1 #number of rows in the figure
            ncols=1 #number of columns in the figure
            fig=plt.figure(figsize=(8*ncols,8*nrows)) #create the figure
            ax=fig.add_subplot(nrows,ncols,1,projection=WCS(pv.header)) #create the subplot
        #PREPARE THE SUBPLOT#
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=WCS(pv.header)) #create the subplot
        #DEFINE THE LIMITS#
        if lim is None: #if no axes limits are given
            xlim=[np.min(np.where(~np.isnan(data))[1]),np.max(np.where(~np.isnan(data))[1])] #set the x limit by to the first and last non-nan
            ylim=[np.min(np.where(~np.isnan(data))[0]),np.max(np.where(~np.isnan(data))[0])] #set the y limit by to the first and last non-nan
        else: #get it from the input
            xlim=[lim[0],lim[1]]
            ylim=[lim[2],lim[3]]
        if pv.header['CDELT2'] < 0: #if the spectral separation is negative
            ylim=np.flip(ylim) #flip the y-axis
        left_space=0.025 #left margin for annotations position
        lower_space=0.025 #bottom margin for annotations position
        upper_space=0.95 #top margin for annotations position
        #DEFINE THE NORMALIZATION#
        vmin=0.1*np.nanmin(data) #set the lower limit for the normalization
        vmax=0.7*np.nanmax(data) #set the upper limit for the normalization
        norm=cl.PowerNorm(gamma=0.3,vmin=vmin,vmax=vmax)
        #DEFINE THE CONTOURS#
        if ctr is None: #if no contour levels are provided
            ctr=np.power(sigma,np.arange(1,10,2)) #4 contours level between nsigma and nsigma^9
        else:
            ctr=np.array(ctr) #convert the list of contours into an array
        if verbose:
            print(f'Contours level: {ctr*rms} Jy/beam')
        #DO THE PLOT#
        im=ax.imshow(data,cmap=cmap,norm=norm,aspect='auto') #plot the pv slice
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        if pv.header['CUNIT2']=='m/s': #if the spectral units are m/s
            ax.coords[1].set_format_unit(u.km/u.s) #convert to km/s
        ax.coords[0].set_format_unit(u.arcmin) #convert x units to arcmin
        ax.set_xlabel('Offset from center [arcmin]') #set the x-axis label
        ax.set_ylabel('Velocity [km/s]') #set the y-axis label
        ax.axvline(x=pv.header['CRPIX1']-1,ls='-.',c='black') #draw the galactic center line.-1 is due to python stupid 0-couting
        #ADD THE CONTOURS#
        ax.contour(data/rms,levels=ctr,cmap=ctrmap,linewidths=ctr_width,linestyles='solid') #add the positive contours
        ax.contour(data/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width,linestyles='dashed') #add the negative contours
        #ADD ANCILLARY INFORMATION#
        if vsys is not None: #if the systemic velocity is given
            if pv.header['CUNIT2']=='km/s': #if the spectral units are km/s
                vsys=vsys*1000 #convert the systemic velocity into m/s
            ax.axhline(y=((vsys-pv.header['CRVAL2'])/pv.header['CDELT2'])+pv.header['CRPIX2']-1,ls='--',c='black') #draw the systemic velocity line. -1 is due to python stupid 0-couting
        else:
            ax.axhline(y=((0-pv.header['CRVAL2'])/pv.header['CDELT2'])-pv.header['CRPIX2'],ls='--',c='black') #draw the systemic velocity line
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,left_space,upper_space) #draw the 10-kpc line

        #SAVE AND PRINT THE PLOT#
        fig.subplots_adjust(wspace=0.0,hspace=0.0) #fix the position of the subplots in the figure   
        if saveplot: #if the save switch is true
            fig.savefig(output+plot_format,dpi=300,bbox_inches='tight') #save the figure
        if position is None: #if the figure was created internally
            if verbose: #if print-to-terminal option is true
                plt.show() #show the figure
            plt.close(fig) #close the figure
    else: #if no plot has to be made
        return pv #return the pvslice
        
#############################################################################################
def lines_finder(spectrum=None,spectrum_rms=None,smooth_kernel=None,sc_threshold=3,sc_replace=2,sc_fluxrange='negative',sc_statistics='mad',link_kernel=3,min_size=3,rel_threshold=0,rel_kernel=0.4,rel_snrmin=3,keep_negative=True,gaussianity=False,write_catalogue=False,finder_out='',**kwargs):
    """One-dimensional source finding algorithm for spectral lines detection. It resembles the The Source Finding Application, SoFiA (Serra et al. 2015, Westmeier et al. 2021).

    Args:
        spectrum (array-like): An n-dimensional array or list of lists containing the data. Each dimension of the array or nested list represents a single spectrum.
        spectrum_rms (array-like/float): The root mean square (rms) of the spectra. If a single number is provided, it is assumed to be the same for all input spectra. Otherwise, it must be a 1-dimensional array or list with the same size as the number of input spectra.
        smooth_kernel (int/list of int): The size of the boxcar smoothing kernel (or a list of kernel sizes) to apply. Each kernel size must be an odd integer value of 3 or greater, representing the full width of the Boxcar filter used to smooth the spectrum. Set to None or 1 to disable smoothing.
        sc_threshold (float): The flux threshold used by the source finder relative to the measured rms in each smoothing iteration. Recommended values range from about 3 to 5, with lower values requiring the use of the reliability filter to reduce false detections.
        sc_replace (float): Before smoothing the spectrum during each source finder iteration, any flux value that was already detected in a previous iteration will be replaced by this value multiplied by the original noise level in the non-smoothed data cube. The original sign of the data value is preserved. Specify a value less than 0 to disable this feature.
        sc_fluxrange (str): The flux range used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artifacts from affecting the noise measurement. If set to any other value, all pixels will be used in the noise measurement irrespective of their flux.
        sc_statistics (str): The statistic used in the noise measurement process of the source finder. Possible values are 'std' for standard deviation and 'mad' for median absolute deviation. Standard deviation is faster but less robust in the presence of emission or artifacts. Median absolute deviation is more robust in such cases.
        link_kernel (int): The minimum size of sources in channels. Sources that fall below this limit will be discarded by the linker.
        min_size (int): The minimum number of channels a source can cover.
        rel_threshold (float): The reliability threshold in the range of 0 to 1. Sources with a reliability below this threshold will be discarded.
        rel_kernel (float): Scaling factor for the size of the Gaussian kernel used when estimating the density of positive and negative detections in the reliability parameter space.
        rel_snrmin (float): The lower signal-to-noise limit for reliable sources. Detections below this threshold will be classified as unreliable and discarded. The integrated signal-to-noise ratio (SNR) of a source is calculated as SNR = Fsum / (RMS * sqrt(N)), where Fsum is the summed flux density, RMS is the local RMS noise level (assumed to be constant), and N is the number of channels of the source. The spectral resolution is assumed to be equal to the channel width.
        keep_negative (bool): If set to False, detections with negative flux will be discarded at the end of the process.
        gaussianity (bool): If set to True, two statistical tests (Anderson-Darling and Kanekar) will be performed to assess the gaussianity of the noise in the input spectra.
        write_catalogue (bool): If True, a CSV file containing the sources catalogue will be written to disk.
        finder_out (str): The output folder/file name.
    
    Kwargs:
        finder_p_value (float): The p-value threshold of the Anderson-Darling test to reject non-Gaussian spectra. A spectrum is considered to have Gaussian noise if the p-value is greater than this threshold. (Default: 0.05)
        kanekar_threshold (array-like): An array-like object of two float elements representing the minimum and maximum values of the Kanekar test to reject non-Gaussian spectra. A spectrum has Gaussian noise if the test value falls within this range. (Default: [0.8, 1.16])
        objname (str): The name of the object. (Default: '')
        plot_format (str): The format type of the plots (e.g., pdf, jpg, png). (Default: pdf)
        ctr_width (float): The line width of the contours. (Default: 2)
        verbose (bool): If True, messages and plots will be printed to the terminal.
    
    Returns:
        A detection mask and the rms of the last smoothing iteration.
    
    Raises:
        ValueError: If no spectrum or its associated rms are provided.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf') #store the plot format
    if spectrum is None or spectrum_rms is None: #!!! ATTENZIONE: this can be improved. It can tell if the spectrum or the rms are missing. Also, can be avoided by running a warning and return None for the mask and the smooth_rms!!!
        raise ValueError(f'ERROR: No spectrum or its associated rms are provided. Cannot run the source finder!')
    if type(spectrum_rms) in [list,np.ndarray]: #if the rms is a list of list or an ndarray
        if len(spectrum_rms)!=len(spectrum): #if the input spectrum array has different size than the input rms array
            raise ValueError(f'ERROR: Input spectrum and rms arrays have different sizes ({len(spectrum)} and {len(spectrum_rms)}). Cannot run the source finder!')
    elif any(isinstance(i,list) for i in spectrum) or any(isinstance(i,np.ndarray) for i in spectrum): #if the rms is a number and the input spectrum is an ndarray or a list of list
        spectrum_rms=np.repeat(spectrum_rms,len(spectrum)) #make an array with the same size of the input spectral list/array
    else: #if the rms is a number and the input spectrum is a 1D array or a single list
        spectrum_rms=[spectrum_rms] #convert the rms into a nested list
    if not any(isinstance(i,list) for i in spectrum) and not any(isinstance(i,np.ndarray) for i in spectrum): #if the input spectrum is not a nested list
        spectrum=[spectrum] #convert into a nested list
    output=finder_out #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    else:
        path=os.getcwd() #get the current working directory
    #CHECK THE KWARGS#    
    p_value=kwargs.get('finder_p_value',0.05) #store the Anderson-Darling p-value threshold
    kanekar_threshold=kwargs.get('kanekar_threshold',[0.79,1.16]) #store the Kanerkar value threshold
    objname=kwargs.get('objname','') #store the object name
    ctr_width=kwargs.get('ctr_width',2) #store the contour width   
    
    #---------------   START THE FUNCTION   ---------------#
    if verbose:
        print(f'Running the source finder with the following parameters:\nFlux threshold: {sc_threshold}*rms\nDetections replacement: {sc_replace}*rms\nSmoothing kernel(s): {smooth_kernel} channel(s)\nLinker kernel(s): {link_kernel} channel(s)\nReliability threshold: {rel_threshold}\n----------------------------------------------')
    source_ids=[] #initialize the list of the sources ID
    mask_list=[] #initialize the list for the masks
    smooth_rms_list=[] #initialize the list for the rms of the smoothed spectra
    anderson_test_list=[] #initialize the list of the anderson test result
    g_test_list=[] #initialize the list of the gaussianity test result
    for i in tqdm(range(len(spectrum)),desc='Spectra analysed',total=len(spectrum)): #for each input spectrum
        #S+C FINDER#
        if smooth_kernel is None or smooth_kernel==1:
            mask=__source_finder(spectrum[i],sc_threshold*spectrum_rms[i]) #run the source finding
            smooth_rms=spectrum_rms[i] #set the smoothed rms to the one of the spectrum
        else:
            smooth_rms=[] #initialize the rms list after each smoothing
            mask=np.zeros(len(spectrum[i])) #initialize the spectral mask
            for ker in smooth_kernel: #for each smoothing kernel
                dummy=spectrum[i].copy() #create a copy of the original spectrum
                if sc_replace>0:
                    dummy[mask>0]=sc_replace*spectrum_rms[i]*np.sign(dummy[mask>0]) #replace the already detected channels with a value equal to 'sc_replace'*rms and conserving the original sign
                kernel=conv.Box1DKernel(ker) #define the smoothing boxcar kernel
                smooth_spectrum=conv.convolve(dummy,kernel) #smooth the spectrum
                dummy=smooth_spectrum.copy() #create a copy of the smoothed spectrum for the new rms calculation
                if sc_fluxrange == 'negative':
                    dummy[dummy>=0]=np.nan #blank the positive flux values for rms calculation
                elif sc_fluxrange=='positive':
                    dummy[dummy<=0]=np.nan
                rms=__rms(dummy,sc_statistics,sc_fluxrange) #calculate the rms
                smooth_rms.append(rms) #append the smooth rms to the rms list
                mask=np.add(mask,__source_finder(smooth_spectrum,sc_threshold*rms)) #run the source finding
            
        #LINKER#
        mask=__source_linker(spectrum[i],mask,link_kernel,min_size) #run the linker
        source_ids.append(mask.copy()) #append the current mask, which cointains the sources IDs, to the sources IDs list
        #GAUSSIANITY TESTS
        if np.all(mask==0): #if no sources have been found
            mask[mask==0]=np.nan #blank the non-detections
            if verbose:
                warnings.warn('No sources have been found.')
            if gaussianity:
                anderson_test_list.append(np.nan)
                g_test_list.append(np.nan)
        else:
            if gaussianity:
                #ANDERSON-DARLING GAUSSIANITY#
                dummy=spectrum[i].copy() #make a copy of the spectrum
                dummy=dummy[mask==0] #select the non-detected channels (i.e., where only noise is present)
                anderson_test=anderson(dummy) #perform the anderson test 
                if anderson_test.statistic>=.6:
                    p=np.exp(1.2937-5.709*anderson_test.statistic+.0186*anderson_test.statistic**2)
                elif anderson_test.statistic>=.34:
                    p=np.exp(0.9177-4.279*anderson_test.statistic-1.38*anderson_test.statistic**2)
                elif anderson_test.statistic>=.2:
                    p=1-np.exp(-8.318+42.796*anderson_test.statistic-59.938*anderson_test.statistic**2)
                else:
                    p=1-np.exp(-13.436+101.14*anderson_test.statistic-223.73*anderson_test.statistic**2) #store the p-value of the test. The p-value gives the probability that the spectrum is gaussian. If p>0.05, the spectrum can be considered gaussian
                anderson_test_list.append(p)
                
                #GAUSSIANITY#
                dummy=spectrum[i].copy() #make a copy of the spectrum
                rms_0=__rms(dummy,sc_statistics,sc_fluxrange) #calculate the rms before smmothing
                dummy[mask!=0]=np.nan #blank the detections
                kernel=conv.Box1DKernel(10) #define the smoothing boxcar kernel
                dummy=conv.convolve(dummy,kernel) #convolve the spectrum
                test_rms=__rms(dummy,sc_statistics,sc_fluxrange) #calculate the rms after smmothing
                g_test=test_rms*np.sqrt(10)/rms_0 #we expect that the rms of the spectrum is reduced by a factor of sqrt(10) (see https://www.researchgate.net/publication/3635815_Effect_of_smoothing_window_length_on_RMS_EMG_amplitude_estimates). Based on mock gaussian spectra, 95% of the gaussian spectra have 1.16 >= g_test >= 0.79. So, a threshold of 0.79 and 1.16 could be a good value for rejecting non gaussian spectra (i.e., a spectrum is non gaussian if 0.79 < g_test < 1.16).
                g_test_list.append(g_test)
                    
            #NEGATIVE FILTERING#
            if not keep_negative: #if the negative sources must be descarded
                mask[mask<=0]=np.nan #blank the negative sources
            else:
                mask[mask==0]=np.nan #blank the non-detections
                mask[mask<0]=-1 #set to -1 the negative sources
            mask[mask>0]=1 #set to 1 the positive sources 
            
        #OUTPUT VARIABLE
        mask_list.append(mask)
        smooth_rms_list.append(smooth_rms)
        
    #RELIABILITY#
    if rel_threshold>0: #if the reliability must be computed
        #first we need to get all the detections. To do that, we take all the spectra and put them on a 1D array
        if verbose:
            print('Splitting positive and negative sources\n----------------------------------------------\n----------------------------------------------')
        pos=[] #initialize the list of the positive sources
        pos_spec_id=[] #initialize the list of the spectrum IDs of the positive sources. This is needed because I need to know at which spectrum each positive source correspond, so that later I can find the spectra IDs corresponding to reliable sources
        neg=[] #initialize the list of the negative sources
        for i in range(len(spectrum)): #run over the spectra and extract the sources
            if not np.all(np.isnan(source_ids[i])): #if there are sources in the spectrum i
                for j in range(int(np.nanmin(source_ids[i])),int(np.nanmax(source_ids[i]))+1): #run over the source indexes of the i-spectrum
                    if j < 0: #if j is negative
                        neg.append(spectrum[i][source_ids[i]==j]/spectrum_rms[i]) #append to the negative list the flux values divided by the spectrum rms
                    elif j > 0: #if j is positive
                        pos.append(spectrum[i][source_ids[i]==j]/spectrum_rms[i]) #append to the positive list the flux values divided by the spectrum rms
                        pos_spec_id.append(i) #append the spectrum ID of the positive source
        pos=np.array(pos) #convert to array
        pos_spec_id=np.array(pos_spec_id) #convert to array
        neg=np.array(neg) #convert to array
        
        if verbose:
            print(f'Total sources: {len(pos)+len(neg)}\nPositive sources: {len(pos)}\nNegative sources: {len(neg)}\n----------------------------------------------\n----------------------------------------------')
        if verbose:
            print(f'Rejecting unreliable sources\n----------------------------------------------')
        plt.ioff()
        reliability=__reliability(pos,neg,rel_snrmin,rel_threshold,rel_kernel,outdir=path,objname=objname,plot_format=plot_format,ctr_width=ctr_width) #calculate the reliability of the positive sources
        plt.ion()
        
        reliable_ids=pos_spec_id[np.where(reliability>=rel_threshold)[0]] #the spectrum IDs having reliable sources are given by the pos_spec_id element corresponding to where the reliability array is greater than the threshold
        
        if verbose:
            print(f'Detected reliable sources : {len(reliable_ids)}')
        ## THIS PART DOESN'T WORK. YOU HAVE TO DECIDE IF YOU WANT TO SHOW IN THE STACKED SPECTRA PLOTS ONLY THE RELIABLE SOURCES (IN WHICH CASE YOU HAVE TO FIX THIS PART) OR ALL THE DETECTION (IN WHICH CASE REMOVE THIS PART)
        #non_rel_idx=np.where(reliability<rel_threshold)[0]+1 #find the index of the non reliable sources. +1 is needed because python counts from 0. reliability is an array of length equal to the number of positive sources. The first element is the first positive source, which has an index of 1.
        #for idx in non_rel_idx: #blank the non reliable sources
        #    mask[np.where(mask==idx)[0]]=np.nan #the mask contains positive values (indexes of the positive sources) and negative values. I know the indexes of the non-reliable positive values, hence I can blank them
    else:
        reliable_ids=None
        
    if write_catalogue: #if a catalogue must be written
        chans=np.arange(len(spectrum[0]))
        #to efficiently create the catalogue, we create each row as a list and append all rows to a list. 
        #Then, we convert the list of list, where list[0] is the row 0, into a dataframe
        catalogue=[] #initialize the catalogue as list (will become a list of list)
        colnames=['Spectrum ID','Source ID','Channel min','Channel max','Channels','Flux','Rms','Flux density','Mean','Median'] #names of the catalogue columns
        if gaussianity:
            colnames.append('Anderson test')
            colnames.append('Kanekar test')
        if rel_threshold>0:
            colnames.append('Reliability')
            k=0 #initialize the reliabile source counter
        for i in range(len(spectrum)): #run over the spectra and extract the sources
            if not np.all(np.isnan(source_ids[i])): #if there are sources in the spectrum i
                for j in range(int(np.nanmin(source_ids[i])),int(np.nanmax(source_ids[i]))+1): #run over the source indexes of the i-spectrum
                    if j!=0: #we have to omit 0 because it is not a source id
                        row=[] #create a new row
                        row.append(i+1) #spectrum ID
                        row.append(j) #source ID
                        chanmin=np.nanmin(np.where(source_ids[i]==j)[0])+1
                        row.append(chanmin) #source channel min
                        chanmax=np.nanmax(np.where(source_ids[i]==j)[0])+1
                        row.append(chanmax) #source channel max
                        row.append(chanmax-chanmin+1) #source channels. +1 is needed: if a source starts at channel 4 and ends at channel 6, it covers  6-4+1=3 channels, not 6-4=2 channels
                        row.append(np.nansum(spectrum[i][source_ids[i]==j])) #source flux
                        row.append(spectrum_rms[i]) #source rms
                        row.append(np.nansum(spectrum[i][source_ids[i]==j])/(np.sqrt((chanmax-chanmin+1))*spectrum_rms[i])) #source flux density: total flux / (N channels*rms)
                        row.append(np.nanmean(spectrum[i])) #spectrum mean
                        row.append(np.nanmedian(spectrum[i])) #spectrum median
                        if gaussianity:
                            row.append(anderson_test_list[i]) #anderson test value
                            row.append(g_test_list[i]) #gaussianity test value
                        if rel_threshold>0:
                            if j>0: #if a positive source
                                row.append(reliability[k+j-1]) #append its reliability value. -1 is needed because at the beginning, j=1 and k=0 so j+k=1 and not 0 as it should be.
                            else: #if a negative source
                                row.append(0) #append 0
                        catalogue.append(row) #append the row to the catalogue
                if rel_threshold>0:
                    if j>0: #if the last value of j is positive (can be negative if spectrum i has only negative sources
                        k+=j #increment the relibility counter by the maximum positive sources of the spectrum i
        catalogue=pd.DataFrame(catalogue,columns=colnames) #convert the catalogue into a dataframe
        filtered_catalogue=catalogue.copy() #create a copy of the catalogue
        if rel_threshold > 0: #filter unreliable sources
            filtered_catalogue=filtered_catalogue.drop(filtered_catalogue[filtered_catalogue['Reliability']<rel_threshold].index) #remove the non reliable sources
        if gaussianity: #filter non-Gaussian spectra
            filtered_catalogue=filtered_catalogue.drop(filtered_catalogue[(filtered_catalogue['Anderson test']<p_value) & ((filtered_catalogue['Kanekar test'] > kanekar_threshold[1]) | (filtered_catalogue['Kanekar test'] < kanekar_threshold[0]))].index)
        catalogue.to_csv(finder_out+'_sources_catalogue.csv',index=False) #remove sources in non-Gaussian spectra
        filtered_catalogue.to_csv(finder_out+'_sources_catalogue_filtered.csv',index=False)   

    return mask_list,smooth_rms_list,reliable_ids

#############################################################################################
def noise_variations(datacube='',nv_statistics='mad',nv_fluxrange='negative',noise_out='',**kwargs):
    """Determine the noise variations of a data cube along the spatial and spectral axes.

    Args:
        datacube (str or ndarray): Name or path+name of the FITS data cube.
        nv_statistics (str): Statistic to be used in the noise measurement. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is the fastest algorithm but the least robust one with respect to emission and artifacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artifacts.
        nv_fluxrange (str): The flux range used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artifacts from affecting the noise measurement. If set to any other value, all pixels will be used in the noise measurement irrespective of their flux.
        noise_out (str): The output folder/file name.
        
    Kwargs:
        path (str): Path to the data cube if the datacube is a name and not a path+name.
        fluxunits (str): Flux units of the data (Default: None).
        plot_format (str): File format of the plots (Default: pdf).
        verbose (bool): If True, print messages to the terminal (Default: False).
    
    Returns:
        Plots of the root mean square (RMS) in each channel and each pixel. The pixel map is also returned as a FITS file if the input data cube is a FITS file.
    
    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no path is provided.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf') #store the plot output format
    datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    output=noise_out #store the output directory from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    #CHECK THE KWARGS#
    specunits=kwargs.get('specunits',None) #store the spectral units
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')   
    spectralres=kwargs.get('spectralres',None) #store the spectral resolution
    fluxunits=kwargs.get('fluxunits',None) #store the flux units

    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #open the data cube
    #CHECK WHICH INFORMATION ARE ALREADY GIVEN#
    #------------   FLUX UNITS    ------------#        
    if fluxunits is None and 'BUNIT' in header: #if no spectral units are given and the keyword is in the header
        fluxunits=header['BUNIT'] #store the spectral units from the cube header
    elif fluxunits is None:
        fluxunits='Jy/beam' #set the first channel to 0 km/s
        if verbose:
            warnings.warn('No flux unit was found: flux unit set to Jy/beam!')
        
    #DO THE NOISE ANALYSIS#
    #------------   SPECTRAL   ------------#
    if verbose:
        print('Calculating the rms spectral variation\n----------------------------------------------') 
    channels=np.arange(data.shape[0])+1 #channels axis
    spectral_rms=[] #initialize the spectral rms list
    for i in tqdm(np.arange(data.shape[0]),desc='Channel processed',total=data.shape[0]): #run through the spectral axis
        spectral_rms.append(__rms(data[i,:,:],nv_statistics,nv_fluxrange)) #for each channel map calculate the rms
    spectral_rms=np.array(spectral_rms) #convert the list to an array
    
    #------------   SPATIAL   ------------#
    if verbose:
        print('Calculating the rms spatial variation\n----------------------------------------------') 
    spatial_rms=np.zeros((data.shape[1],data.shape[2])) #initialize the spatial rms map 
    with tqdm(desc='Pixel processed',total=data.shape[1]*data.shape[2]) as pbar:   
        for i in np.arange(data.shape[1]): #run over the pixels
            for j in np.arange(data.shape[2]):
                spatial_rms[i,j]=__rms(data[:,i,j],nv_statistics,nv_fluxrange) #for each pixel calculate the rms of its spectrum
                pbar.update(1)
    spatial_rms[spatial_rms==0]=np.nan #sometimes a pixel is glitchy and has a value in a single channel. This will lead to a zero rms. This type of pixel must be blanked
                
    #------------   OUTPUTS   ------------#
    if verbose:
        print('Writing the outputs\n----------------------------------------------') 
    if header is not None: #if a header is provided
        #SPATIAL VARIATION#
        wcs=WCS(header).dropaxis(2)
        hdu=fits.PrimaryHDU(spatial_rms.astype('float32'),header=wcs.to_header()) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        hdul.writeto(f'{output}_spatial.fits',overwrite=True) #write the data into a fits
        hdul.close() #close the file to realease the memory
        #SPECTRAL VARIATION#
        df=pd.DataFrame(columns=[f'CHANNEL',f'FLUX [{fluxunits}]']) #create the dataframe
        df[f'CHANNEL']=np.arange(len(spectral_rms))+1 #store the channel list
        df[f'FLUX [{fluxunits}]']=spectral_rms #store the rms list
        df.to_csv(f'{output}_spectral.csv',index=False) #convert into a csv
        #NOISE CUBE#
        #to create the 3D noise cube we have to normalize the 2D-array or the 1D-array representing the spatial and spectral noise variations. It doesn't matter which get normalized. Then, we multiply the normalized array per the other to get the 3D noise cube. So we normalize the spatial variation with spatial_rms/np.nanmax(spatial_rms) and multiply this 2D array by the spectral variation 1D array
        hdu=fits.PrimaryHDU(np.multiply.outer(spectral_rms,spatial_rms/np.nanmax(spatial_rms)).astype('float32'),header=header) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        hdul.writeto(f'{output}_cube.fits',overwrite=True) #write the data into a fits
        hdul.close() #close the file to realease the memory
                
    #------------   PLOTS    ------------#
    if verbose:
        print('Doing some fancy plots\n----------------------------------------------') 
    fig=plt.figure(figsize=(8,4)) #create the figure for the frequency-dependence of the rms
    ax=fig.add_subplot() #create the subplot for the rms
    exponent=int(np.nanmean(np.log10(spectral_rms[~np.isnan(spectral_rms)]))) #mean power of the rms values
    ax.plot(channels,spectral_rms/10**exponent,c='blue',lw=2)
    ax.set_xlabel('Channel') #set the x-axis label
    ax.set_ylabel(fr'RMS [$\mathdefault{{10^{{{exponent}}}}}$ $\mathdefault{{{fluxunits}}}$]') #set the y-axis label
    ax.set_xlim([0,len(spectral_rms)]) #set the x-axis limit
    fig.savefig(f'{output}_spectral{plot_format}',dpi=300,bbox_inches='tight')
    
    fig=plt.figure(figsize=(7,7)) #create the figure for the frequency-dependence of the rms
    if header is not None: #if a header is provided        
        ax=fig.add_subplot(projection=wcs) #create the subplot for the rms
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label 
    else:
        ax=fig.add_subplot() #create the subplot for the rms
        ax.set_xlabel('Pixel') #set the x-axis label
        ax.set_ylabel('Pixel') #set the y-axis label 
    norm=cl.PowerNorm(gamma=0.3,vmin=0.1*np.nanmax(spatial_rms),vmax=0.9*np.nanmax(spatial_rms)) #define the normalization from negative detection limit to 20% of the maximum
    im=ax.imshow(spatial_rms,cmap='Greys_r',norm=norm,aspect='equal',origin='lower')
    exponent=int(np.nanmean(np.log10(spatial_rms[~np.isnan(spatial_rms)]))) #mean power of the rms values
    cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label=fr'RMS [$\mathdefault{{10^{{{exponent}}}}}$ {fluxunits}]',fraction=0.0476) #add the colorbar on top of the plot
    cb.ax.tick_params(direction='in',length=5) #change the ticks of the colorbar from external to internal and made them longer
    cb.set_ticks(norm.inverse(np.linspace(0.2,0.8,4))) #set the ticks of the colobar to the levels of the contours
    cb.ax.set_xticklabels([f'{i/10**exponent:.2f}' for i in cb.get_ticks()]) #set ticks in a nice format
    fig.savefig(f'{output}_spatial{plot_format}',dpi=300,bbox_inches='tight')
    
    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    plt.close()

#############################################################################################
def radial_profile():
    return None
             
#############################################################################################
def plotmom(which='all',mom0map='',mom1map='',mom2map='',plotmom_out='',**kwargs):
    """Plot the given moment maps. All (moment 0, moment 1, and moment 2) can be plotted simultaneously or only one.

    Args:
        which (str): Specifies what to plot:
            - 'all': Plot moment 0, moment 1, and moment 2 maps in a single 1-row 3-column figure.
            - 'mom0': Plot moment 0 map.
            - 'mom1': Plot moment 1 map.
            - 'mom2': Plot moment 2 map.
        mom0map (str/ndarray): Name or path+name of the FITS moment 0 map.
        mom1map (str/ndarray): Name or path+name of the FITS moment 1 map.
        mom2map (str/ndarray): Name or path+name of the FITS moment 2 map.
        plotmom_out (str): The output folder/file name.
        save (bool): Saves the plot if True.

    Kwargs:
        path (str): Path to the moment map if momXmap is a name and not a path+name.
        pbcorr (bool): Applies the primary beam correction if True. Note that in that case, you must supply a beam cube. (Default: False)
        beamcube (str/ndarray): Name or path+name of the FITS beam cube if pbcorr is True.
        datacube (str/ndarray): Name or path+name of the FITS data cube to be used to retrieve necessary information.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        spectralres (float): Cube spectral resolution in km/s. (Default: None)
        bmaj (float): Beam major axis in arcsec. (Default: None)
        bmin (float): Beam minor axis in arcsec. (Default: None)
        bpa (float): Beam position angle in degrees. (Default: None)
        nsigma (float): Lowest contour level in terms of nsigma*rms. (Default: 3)
        vsys (float): Object systemic velocity in km/s. (Default: None)
        pixelres (float): Pixel resolution of the data in arcsec. (Default: None)
        asectokpc (float): Arcsec to kpc conversion to plot the spatial scale. (Default: None)
        objname (str): Name of the object. (Default: '')
        wcs (astropy.WCS): WCS as astropy.WCS object to use in the plot. It will replace the default one. (Default: None)
        position (int): Position of the subplot in the figure as a triplet of integers (111 = nrow 1, ncol 1, index 1). (Default: 111)
        rms (float): RMS of the data cube in Jy/beam as a float. If not given (None), the function tries to calculate it. (Default: None)
        mom0_ctr (array-like): List or array of moment 0 contour levels in units of 10^18. They will replace the default levels. (Default: None)
        mom1_ctr (array-like): List or array of moment 1 contour levels in units of km/s. They will replace the default levels. (Default: None)
        mom2_ctr (array-like): List or array of moment 2 contour levels in units of km/s. They will replace the default levels. (Default: None)
        lim (array-like): List or array of plot x and y limits as [xmin, xmax, ymin, ymax]. They will replace the default limits. (Default: None)
        mom0_cmap (str/ndarray): Name of the colormap to be used for the moment 0 map. Accepted values are those of matplotlib.colors.Colormap. (Default: 'Greys')
        mom1_cmap (str/ndarray): Name of the colormap to be used for the moment 1 map. Accepted values are those of matplotlib.colors.Colormap. (Default: 'jet')
        mom2_cmap (str/ndarray): Name of the colormap to be used for the moment 2 map. Accepted values are those of matplotlib.colors.Colormap. (Default: 'YlGnBu')
        ctr_width (float): Line width of the contours. (Default: 2)
        mom0_ctrmap (str/ndarray): Name of the colormap to be used for the moment 0 map contour levels. Accepted values are those of matplotlib.colors.Colormap. (Default: 'hot')
        mom2_ctrmap (str/ndarray): Name of the colormap to be used for the moment 2 map contour levels. Accepted values are those of matplotlib.colors.Colormap. (Default: 'RdPu_r')
        plot_format (str): File format of the plots (Default: pdf).
        verbose (bool): Option to print messages and plot to terminal if True. (Default: False)

    Returns:
        None

    Raises:
        ValueError: If the 'which' argument does not match accepted values.
        ValueError: If no moment 0 or moment 1 or moment 2 is given.
        ValueError: If no path is provided.
        ValueError: If no output folder is set.
        ValueError: If no data cube is given when use_cube is True.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf') #store the plot output format
    if which not in ['all','mom0','mom1','mom2']: #if wrong data is given
        raise ValueError("ERROR: wrong data selection. Accepted values: ['all','mom0','mom1','mom2']. Aborting")
    if which in ['all','mom0']: #if all moment maps must be plotted or only the moment 0
        if type(mom0map)==str or mom0map is None: #if the moment 0 map is a string
            mom0map=__read_string(mom0map,'mom0map',**kwargs) #store the path to the moment 0 map
        #------------   PB CORRECTION     ------------#
        pbcorr=kwargs.get('pbcorr',False) #store the apply pb correction option
        if pbcorr: #if the primary beam correction is applied
            beamcube=kwargs.get('beamcube',None) #store the data cube path from the input parameters
            if beamcube is None: #if the beam cube is not provided
                if verbose:
                    warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')
                pbcorr=False #set the pbcorr option to False
            elif type(beamcube)==str: #if the beam cube is a string
                beamcube=__read_string(beamcube,'beamcube',**kwargs) #store the path to the beam cube
    if which in ['all','mom1']: #if all moment maps must be plotted or only the moment 1 map
        if type(mom1map)==str or mom1map is None: #if the moment 1 map is a string
            mom1map=__read_string(mom1map,'mom1map',**kwargs) #store the path to the moment 1 map
        vsys=kwargs.get('vsys',None) #store the systemic velocity from the input kwargs
    if which in ['all','mom2']: #if all moment maps must be plotted or only the moment 2 map
        if type(mom2map)==str or mom2map is None: #if the moment 2 map is a string
            mom2map=__read_string(mom2map,'mom2map',**kwargs) #store the path to the moment 2 map
    output=plotmom_out+plot_format #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    #CHECK THE KWARGS#
    pixunits=kwargs.get('pixunits',None) #store the spatial units
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    specunits=kwargs.get('specunits',None) #store the spectral units
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')   
    pixelres=kwargs.get('pixelres',None) #store the spatial resolution
    spectralres=kwargs.get('spectralres',None) #store the spectral resolution
    bmaj=kwargs.get('bmaj',None) #store the beam major axis
    bmin=kwargs.get('bmin',None) #store the beam minor axis
    bpa=kwargs.get('bpa',None) #store the beam position angle
    rms=kwargs.get('rms',None) #store the rms
    nsigma=kwargs.get('nsigma',3) #store the lowest contour in terms of rms*sigma
    cs_statistics=kwargs.get('cs_statistics','mad') #store the statistics to be use for rms calculation
    asectokpc=kwargs.get('asectokpc',None) #store the arcseconds to kpc conversion
    objname=kwargs.get('objname','') #store the object name
    wcs=kwargs.get('wcs',None) #store the optional wcs
    position=kwargs.get('position',None) #store the position of the subplot
    mom0_ctr=kwargs.get('mom0_ctr',None) #store the moment 0 contour levels
    mom1_ctr=kwargs.get('mom1_ctr',None) #store the moment 1 contour levels
    mom2_ctr=kwargs.get('mom2_ctr',None) #store the moment 2 contour levels
    lim=kwargs.get('lim',None) #store the moment 0 contour levels
    mom0_cmap=kwargs.get('mom0_cmap','Greys') #store the moment 0 colormap
    mom1_cmap=kwargs.get('mom1_cmap','mom1') #store the moment 1 colormap
    mom2_cmap=kwargs.get('mom2_cmap','mom2') #store the moment 2 colormap
    ctr_width=kwargs.get('ctr_width',2) #store the contour width
    mom0_ctrmap=kwargs.get('mom0_ctrmap','hot') #store the moment 0 contour colors
    mom2_ctrmap=kwargs.get('mom2_ctrmap','mom2_ctr') #store the moment 2 contour colors
    left_space=kwargs.get('left_space',0.025) #store the left margin for annotations position
    lower_space=kwargs.get('lower_space',0.025) #store the bottom margin for annotations position
    upper_space=kwargs.get('upper_space',0.975) #store the top margin for annotations position        

    #---------------   START THE FUNCTION   ---------------#
    #CHECK FOR THE RELEVANT INFORMATION#
    #------------   CUBE PROPERTIES    ------------#
    prop=[pixunits,specunits,pixelres,spectralres,bmaj,bmin,bpa,rms] #list of cube properties values
    prop_name=['pixunits','specunits','pixelres','spectralres','bmaj','bmin','bpa','rms'] #list of cube properties names
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if one or more cube parameters are not given
        if verbose:
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            warnings.warn(f'I am missing some information: {not_found}. Checking if the correspondig cube is provided to retrieve them!')
        datacube=kwargs.get('datacube',None) #store the data cube from the input kwargs
        if datacube is None: #if the data cube is not provided
            if verbose:
                warnings.warn(f'Cube not found. I cannot display additional information on the plot!')
        elif type(datacube)==str: #if the data cube is a string
            datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
            if verbose:
                warnings.warn(f'Cube found!!! Running cubestat to retrieve the missing information!')
        stats=cubestat(datacube,params_dict=dict(zip(prop_name,prop)),cs_statistics=cs_statistics,verbose=False) #calculate the statistics of the cube
        if pixunits is None: #if the spatial units are not given
            pixunits=stats['pixunits'] #take the value from the cubestat results
        if specunits is None: #if spectral units are not given
            specunits=stats['specunits'] #take the value from the cubestat results
        if pixelres is None: #if pixel resolution is not given
            pixelres=stats['pixelres'] #take the value from the cubestat results
        if spectralres is None: #if spectral resolution is not given
            spectralres=stats['spectralres'] #take the value from the cubestat results
        if bmaj is None: #if bmaj is not given
            bmaj=stats['bmaj'] #take the value from the cubestat results
        if bmin is None: #if bmin is not given
            bmin=stats['bmin'] #take the value from the cubestat results
        if bpa is None: #if bpa is not given
            bpa=stats['bpa'] #take the value from the cubestat results
        if rms is None: #if rms is not given
            rms=stats['rms'] #take the value from the cubestat results
    prop=[pixunits,specunits,pixelres,spectralres,bmaj,bmin,bpa,rms] #update the list of cube properties values
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if still one or more cube parameters are not given
        if verbose:
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            warnings.warn(f'I am still missing some information: {not_found}. I cannot display additional information on the plot!')
   
    #---------------   PREPARE THE DATA   ---------------#
    #PREPARE THE MOMENT 0 MAP#
    if which == 'all' or which == 'mom0': #if all moment maps or moment 0 map must be plotted
        #GET THE DATA AND DO A SIMPLE CONVERSION#
        mom0,mom0header=__load(mom0map) #open the moment 0 map
        #TRY TO GET THE PIXEL RESOLUTION#
        if mom0header is not None: #if the header is None means the input data is not a fits file
            if pixelres is None and 'CDELT1' in mom0header: #if the pixelres is not given and is in the header
                pixelres=mom0header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if mom0header is not None: #if the header is None means the input data is not a fits file
            if pixunits is None and 'CUNIT1' in mom0header: #if the spatial unit is in the header
                pixunits=mom0header['CUNIT1'] #store the spatial unit
        #DO THE PB CORRECTION IF NEEDED#    
        if pbcorr: #if the primary beam correction is applied
            with fits.open(beamcube) as pb_cube: #open the primary beam cube
                pb_slice=pb_cube[0].data[np.array(pb_cube[0].shape[0]/2).astype(int)] #extract the central plane
            del pb_cube[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
            mom0/=pb_slice #apply the pb correction
            if mom0header is not None: #if the header is None means the input data is not a fits file
                momunit=mom0header['BUNIT'].replace('/beam'.casefold(),'')
        elif mom0header is not None: #if the header is None means the input data is not a fits file:
            momunit=mom0header['BUNIT']
        #TRY TO CONVERT INTO HI COLUMNS DENSITY# 
        beamarea=1.13*(bmin*bmaj) if bmaj is not None and bmin is not None else None #calculate the beam area if the beam major and minor axis are given
        if beamarea is not None and spectralres is not None: #if we have the beamarea and spectral resolution
            mom0=converttoHI(mom0,fluxunits=momunit,beamarea=beamarea,pixunits=pixunits,spectralres=np.abs(spectralres),specunits=specunits) #convert the moment 0 map into an HI column density map
            momunit='cm$^{-2}$' #define the column density unit
            if rms is not None:
                sens=converttoHI(rms*nsigma,fluxunits='Jy/beam',beamarea=beamarea,pixunits=pixunits,spectralres=np.abs(spectralres),specunits=specunits) #calculate the sensitivity and normalize it
            else:
                if verbose:
                    warnings.warn('The sensitivity is refering to the minimum of the moment 0 map.')
                sens=np.nanmin(mom0[mom0>0]) #the sensitivity is the min of the moment 0 map
        if mom0header is not None: #if the header is None means the input data is not a fits file
            if mom0header['BUNIT'].casefold()=='jy/beam*m/s' or mom0header['BUNIT'].casefold()=='jy*m/s': #if the units are in m/s
                mom0=mom0/1000 #convert to km/s
        mom0[mom0==0]=np.nan #convert the moment 0 zeros into nan
        
    #PREPARE THE MOMENT 1 MAP#
    if which == 'all' or which == 'mom1': #if all moment maps or moment 1 map must be plotted
        #GET THE DATA AND DO A SIMPLE CONVERSION#
        mom1,mom1header=__load(mom1map) #open the moment 1 map
        if mom1header is not None and mom1header['BUNIT']=='m/s': #if the header is None means the input data is not a fits file
            mom1=mom1/1000 #convert to km/s
        #TRY TO GET THE PIXEL RESOLUTION 
        if mom1header is not None: #if the header is None means the input data is not a fits file
            if pixelres is None and 'CDELT1' in mom1header: #if the pixelres is not given and is in the header
                pixelres=mom1header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if mom1header is not None: #if the header is None means the input data is not a fits file
            if pixunits is None and 'CUNIT1' in mom1header: #if the spatial unit is in the header
                pixunits=mom1header['CUNIT1'] #store the spatial unit
        #REMOVE THE SYSTEMIC VELOCITY                                         
        vsys=np.nanmedian(mom1) if vsys is None else vsys/1000 #calculate the systemic velocity if the systemic velocity is not given, else convert it in km/s
        mom1=mom1-vsys #subract the result to the moment 1 map
        
    #PREPARE THE MOMENT 2 MAP             
    if which == 'all' or which == 'mom2': #if all moment maps or moment 2 map must be plotted
        #GET THE DATA AND DO A SIMPLE CONVERSION#
        mom2,mom2header=__load(mom2map) #open the moment 2 map
        if mom2header is not None and mom2header['BUNIT']=='m/s': #if the header is None means the input data is not a fits file
            mom2=mom2/1000 #convert to km/s
        #TRY TO GET THE PIXEL RESOLUTION 
        if mom2header is not None: #if the header is None means the input data is not a fits file
            if pixelres is None and 'CDELT1' in mom2header: #if the pixelres is not given and is in the header
                pixelres=mom2header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if mom2header is not None: #if the header is None means the input data is not a fits file
            if pixunits is None and 'CUNIT1' in mom2header: #if the spatial unit is in the header
                pixunits=mom2header['CUNIT1'] #store the spatial unit
        #CALCULATE THE FWHM           
        disp=np.nanmedian(mom2) #calculate the median velocity dispersion
    
    #CONVERT THE BEAM TO ARCSEC
    if pixunits == 'deg': #if the spatial units are deg
        bmaj*=3600 #convert the beam major axis in arcsec
        bmin*=3600 #convert the beam minor axis in arcsec
        if which == 'all' or which == 'mom0': #if all moment maps or moment 0 map must be plotted
            beamarea*=3600**2 #convert the beam area in arcsec
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        bmaj*=60 #convert the beam major axis in arcsec
        bmin*=60 #convert the beam minor axis in arcsec
        if which == 'all' or which == 'mom0': #if all moment maps or moment 0 map must be plotted
            beamarea*=60**2 #convert the beam area in arcsec 
            
    #CONVERT THE SPATIAL DIMENSION TO ARCSEC
    if pixelres is not None and pixunits == 'deg': #if the spatial unit is degree
        pixelres*=3600 #convert into arcsec
    elif pixelres is not None and pixunits == 'arcmin': #if the spatial unit is arcmin
        pixelres*=60 #convert into arcsec
    if pixelres<0: #if the spatial resolution is negative
        pixelres=-pixelres #convert it to positive

    #---------------   DO THE PLOT   ---------------#
    #PREPARE THE FIGURE#
    if which == 'all': #if all moment maps must be plotted
        nrows=1 #number of rows in the atlas
        ncols=3 #number of columns in the atlas        
        fig=plt.figure(figsize=(6*ncols,6*nrows)) #create the figure
            
    #PREPARE THE MOMENT 0 SUBPLOT#
    if which == 'all' or which == 'mom0': #if all moment maps must be plotted or only the moment 0 map
        if wcs is None: #if no wcs is given
            wcs=WCS(mom0header) #calculate the wcs
        if which == 'all': #if all moment maps must be plotted
            ax=fig.add_subplot(nrows,ncols,1,projection=wcs) #create the subplot
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=wcs) #create the subplot
        if lim is None: #if no axes limits are given
            xmin=np.min(np.where(~np.isnan(mom0))[1]) #set xmin to the first non-nan pixel
            xmax=np.max(np.where(~np.isnan(mom0))[1]) #set xmax to the last non-nan pixel
            ymin=np.min(np.where(~np.isnan(mom0))[0]) #set ymin to the first non-nan pixel
            ymax=np.max(np.where(~np.isnan(mom0))[0]) #set ymax to the last non-nan pixel
            xlim=[xmin*(1-left_space),xmax*(1+left_space)] #extend a little the xlim
            ylim=[ymin*(0.8-lower_space),ymax*(2-upper_space)] #extend the ylim to place the ancillary information. 0.8 and 2 are for the text size
            #it is highly raccomended to have an equal aspect ration for the moment maps. So we have to extend the shorter axis to match the larger
            if ylim[1]-ylim[0] > xlim[1]-xlim[0]: #if the y-axis is bigger than the x-axis
                extend=(ylim[1]-ylim[0]-xlim[1]+xlim[0])/2 #calculate how much to extend
                xlim[1]=xlim[1]+extend
                xlim[0]=xlim[0]-extend
            else: #if the x-axis is larger
                extend=(xlim[1]-xlim[0]-ylim[1]+ylim[0])/2 #calculate how much to extend
                ylim[1]=ylim[1]+extend
                ylim[0]=ylim[0]-extend
        else: #get it from the input
            xlim=[lim[0],lim[1]]
            ylim=[lim[2],lim[3]]
        #DEFINE THE NORMALIZATION#
        norm=cl.PowerNorm(gamma=0.3,vmin=1.1*np.nanmin(np.abs(mom0)),vmax=0.9*np.nanmax(mom0)) #define the normalization
        #DEFINE THE CONTOURS#
        ctr=norm.inverse(np.linspace(0,1,7))[1:-1] if mom0_ctr is None else np.array(mom0_ctr) #if no contours are provided 5 contours levels between the min and max of the normalized moment 0, else use the user-defined levels
        #DO THE PLOT#
        im=ax.imshow(mom0,cmap=mom0_cmap,norm=norm,aspect='equal') #plot the moment 0 map
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        #ADD THE CONTOURS#
        ax.contour(mom0,levels=ctr,cmap=mom0_ctrmap,linewidths=ctr_width,linestyles='solid',norm=norm)
            
        if pbcorr: #if primary beam correction has been applied
            pb_slice[pb_slice==0]=np.nan #blank the zeroes
            ax.contour(pb_slice,levels=[np.nanmin(pb_slice)*1.02],colors='gray',linewidths=ctr_width,alpha=0.5) #add the sensitivity cutoff contours
        #ADD THE COLORBAR#
        if momunit == 'cm$^{-2}$':
            exponent=int(np.nanmean(np.log10(ctr))) #exponent of the column density
        cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,fraction=0.0476) if which == 'all' else fig.colorbar(im,ax=ax,location='top',pad=0.0,fraction=0.0476) #add the colorbar on top of the plot
        cb.set_label(fr'HI column density [$\mathdefault{{10^{{{exponent}}}}}$ $\mathdefault{{cm^{{-2}}}}$]') if momunit == 'cm$^{-2}$' else f'Flux [{momunit}]' #set the colorbar labeld
        cb.set_ticks(ctr) #set the ticks of the colobar to the levels of the contours
        cb.ax.set_xticklabels([f'{i/10**exponent:.1f}' for i in cb.get_ticks()]) #set ticks in a nice format
        for tick,color in zip(np.power(cb.get_ticks(),norm.gamma)*norm.vmax**(1-norm.gamma), plt.cm.get_cmap(mom0_ctrmap)(norm(cb.get_ticks()))): #we want to assign the color of the countour levels to the correspondong tick. We use np.power(...) to convert the tick into the normalized value. We cannot call norm() because in that case vmax is hardcoded to 1. Hence, we have to manually calculate it. Then, plt.cm.get_cmap(...) is used to retrieve the color of the colormap corresponding to the position of the tick
            cb.ax.axvline(tick,c=color,lw=ctr_width) #create the line
        #ADD ANCILLARY INFORMATION#
        if momunit == 'cm$^{-2}$': #if the moment 0 is column density
            ax.text(left_space,lower_space,f'Detection limit: {sens:.1e} {momunit}',transform=ax.transAxes) #add the information of the detection limit
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,left_space+0.05,upper_space-0.01) #draw the 10-kpc line
    
    #PREPARE THE MOMENT 1 SUBPLOT#
    if which == 'all' or which == 'mom1': #if all moment maps must be plotted or only the moment 0 map
        if wcs is None: #if no wcs is given
            wcs=WCS(mom1header)
        if which == 'all': #if all moment maps must be plotted
            ax=fig.add_subplot(nrows,ncols,2,projection=wcs) #create the subplot
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=wcs) #create the subplot
        if lim is None: #if no axes limits are given
            xmin=np.min(np.where(~np.isnan(mom1))[1]) #set xmin to the first non-nan pixel
            xmax=np.max(np.where(~np.isnan(mom1))[1]) #set xmax to the last non-nan pixel
            ymin=np.min(np.where(~np.isnan(mom1))[0]) #set ymin to the first non-nan pixel
            ymax=np.max(np.where(~np.isnan(mom1))[0]) #set ymax to the last non-nan pixel
            xlim=[xmin*(1-left_space),xmax*(1+left_space)] #extend a little the xlim
            ylim=[ymin*(0.8-lower_space),ymax*(2-upper_space)] #extend the ylim to place the ancillary information. 0.8 and 2 are for the text size
            #it is highly raccomended to have an equal aspect ration for the moment maps. So we have to extend the shorter axis to match the larger
            if ylim[1]-ylim[0] > xlim[1]-xlim[0]: #if the y-axis is bigger than the x-axis
                extend=(ylim[1]-ylim[0]-xlim[1]+xlim[0])/2 #calculate how much to extend
                xlim[1]=xlim[1]+extend
                xlim[0]=xlim[0]-extend
            else: #if the x-axis is larger
                extend=(xlim[1]-xlim[0]-ylim[1]+ylim[0])/2 #calculate how much to extend
                ylim[1]=ylim[1]+extend
                ylim[0]=ylim[0]-extend
        else: #get it from the input
            xlim=[lim[0],lim[1]]
            ylim=[lim[2],lim[3]]
        #DEFINE THE NORMALIZATION#
        norm=cl.CenteredNorm()
        #DEFINE THE CONTOURS#
        if mom1_ctr is None: #if the contours levels are not given
            ctr_res=np.floor(0.85*np.nanmax(np.abs(mom1))/15)*5 #we want the contours to be multiple of 5
            ctr_pos=np.arange(0,0.85*np.nanmax(np.abs(mom1)),ctr_res)[1:].astype(int) #positive radial velocity contours
            ctr_neg=np.flip(-ctr_pos) #negative radial velocity contours
        else:
            mom1_ctr=np.array(mom1_ctr) #convert the input contour levels into an array
            ctr_pos=mom1_ctr[mom1_ctr>0] #get the positive values
            ctr_neg=mom1_ctr[mom1_ctr<0] #get the negative values
        #DO THE PLOT#
        im=ax.imshow(mom1,cmap=mom1_cmap,norm=norm,aspect='equal') #plot the moment 1 map with a colormap centered on 0
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if which == 'all': #if it is an atlas
            ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
        #ADD THE CONTOURS#
        ax.contour(mom1,levels=ctr_pos,colors='black',linewidths=ctr_width,linestyles='solid') #add the contours
        ax.contour(mom1,levels=ctr_neg,colors='silver',linewidths=ctr_width,linestyles='dashed') #add the contours contours
        #ADD THE COLORBAR#
        cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='Radial velocity [km/s]',fraction=0.0476) if which == 'all' else plt.colorbar(im,ax=ax,location='top',pad=0.0,label='Radial velocity [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        cb.ax.tick_params(direction='in',length=5) #change the ticks of the corobar from external to internal and made them longer
        cb.set_ticks(np.concatenate((ctr_neg,0,ctr_pos),axis=None)) #set the ticks of the colobar to the levels of the contours
        for tick in ctr_pos: #for each positive contour
            cb.ax.axvline(tick,c='black',lw=ctr_width) #create the line for each contour level
        for tick in ctr_neg: #for each negative contour
            cb.ax.axvline(tick,c='silver',lw=ctr_width,ls='dashed') #create the line for each contour level
        #ADD ANCILLARY INFORMATION#
        ax.text(left_space,lower_space,f'Systemic velocity: {vsys:.1f} km/s',transform=ax.transAxes) #add the information of the systemic velocity
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,left_space+0.05,upper_space-0.01) #draw the 10-kpc line
                
    #PREPARE THE MOMENT 2 SUBPLOT#
    if which == 'all' or which == 'mom2': #if all moment maps must be plotted or only the moment 0 map
        if wcs is None: #if no wcs is given
            wcs=WCS(mom2header)
        if which == 'all': #if all moment maps must be plotted
            ax=fig.add_subplot(nrows,ncols,3,projection=wcs) #create the subplot
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=wcs) #create the subplot
        if lim is None: #if no axes limits are given
            xmin=np.min(np.where(~np.isnan(mom2))[1]) #set xmin to the first non-nan pixel
            xmax=np.max(np.where(~np.isnan(mom2))[1]) #set xmax to the last non-nan pixel
            ymin=np.min(np.where(~np.isnan(mom2))[0]) #set ymin to the first non-nan pixel
            ymax=np.max(np.where(~np.isnan(mom2))[0]) #set ymax to the last non-nan pixel
            xlim=[xmin*(1-left_space),xmax*(1+left_space)] #extend a little the xlim
            ylim=[ymin*(0.8-lower_space),ymax*(2-upper_space)] #extend the ylim to place the ancillary information. 0.8 and 2 are for the text size
            #it is highly raccomended to have an equal aspect ration for the moment maps. So we have to extend the shorter axis to match the larger
            if ylim[1]-ylim[0] > xlim[1]-xlim[0]: #if the y-axis is bigger than the x-axis
                extend=(ylim[1]-ylim[0]-xlim[1]+xlim[0])/2 #calculate how much to extend
                xlim[1]=xlim[1]+extend
                xlim[0]=xlim[0]-extend
            else: #if the x-axis is larger
                extend=(xlim[1]-xlim[0]-ylim[1]+ylim[0])/2 #calculate how much to extend
                ylim[1]=ylim[1]+extend
                ylim[0]=ylim[0]-extend
        else: #get it from the input
            xlim=[lim[0],lim[1]]
            ylim=[lim[2],lim[3]]
        #DEFINE THE CONTOURS#
        ctr=np.around(np.linspace(disp,0.9*np.nanmax(mom2),5),1) if mom2_ctr is None else np.array(mom2_ctr) #if not supplied by the user, 5 contours level from the median dispersion to the 90% of the max and convert to 1-decimal float, else use the user-defined levels
        #DO THE PLOT#
        im=ax.imshow(mom2,cmap=mom2_cmap,aspect='equal') #plot the moment 2 map with a square-root colormap and in units of velocity dispersion
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if which == 'all': #if all moment maps must be plotted
            ax.coords[1].set_ticklabel_position('r') #move the y-axis label to the right
            ax.coords[1].set_axislabel_position('r') #move the y-axis tick labels to the right
        #ADD THE CONTOURS#
        ax.contour(mom2,levels=ctr,cmap=mom2_ctrmap,linewidths=ctr_width,linestyles='solid') #add the contours
        #ADD THE COLORBAR#
        cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='Velocity dispersion [km/s]',fraction=0.0476) if which == 'all' else plt.colorbar(im,ax=ax,location='top',pad=0.0,label='Velocity dispersion [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        cb.ax.tick_params(direction='in',length=5) #change the ticks of the colorbar from external to internal and made them longer
        cb.set_ticks(ctr) #set the ticks of the colobar to the levels of the contours
        for tick,color in zip(ctr,plt.cm.get_cmap(mom2_ctrmap)(__normalize(ctr,0,1))): #assign the color of the countour levels to the correspondong tick
            cb.ax.axvline(tick,c=color,lw=ctr_width) #create the line
        #ADD ANCILLARY INFORMATION#
        ax.text(left_space,lower_space,f'Median dispersion: {disp:.1f} km/s',transform=ax.transAxes) #add the information of the velocity dispersion
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,left_space+0.05,upper_space-0.01) #draw the 10-kpc line
                
    #---------------   FIX SUBPLOTS AND SAVE/SHOW   ---------------#
    if which == 'all': #if all moment maps must be plotted
        fig.subplots_adjust(wspace=-0.185) #fix the position of the subplots in the figure   
        fig.savefig(output+plot_format,dpi=300,bbox_inches='tight') #save the figure
            
        if verbose: #if print-to-terminal option is true
            plt.show() #show the figure
        plt.close()
    
#############################################################################################
def removemod(datacube='',modelcube='',maskcube=None,method='subtraction',blankthreshold=0,write_fits=False,removemod_out='',**kwargs):
    """Remove a model from a data cube using one of the five available methods: data-model (subtraction), data blanking (blanking), data-model after data blanking (b+s), data-model and negative residual blanking (negblank), and a combination of blanking and negblank (all).

    Args:
        datacube (str or ndarray): The name or path+name of the FITS data cube.
        modelcube (str or ndarray): The name or path+name of the FITS model cube.
        maskcube (str or ndarray): The name or path+name of the FITS 3D mask to be used in the removal.
        method (str): The method to remove the model. Accepted values are:
            - 'all': Apply blanking and negblank methods.
            - 'blanking': Blank the pixel in the data cube whose value in the model cube is > blankthreshold.
            - 'b+s': Same as blanking, but subtract the model from the data after the blanking.
            - 'negblank': Subtract the model from the data and blank the negative residuals.
            - 'subtraction': Subtract the model from the data.
        blankthreshold (float): The flux threshold for the 'all', 'blanking', and 'b+s' methods.
        write_fits (bool): If True, store the output in a FITS file. If False, return a variable. (default: True)
        removemod_out (str): The output folder/file name.
        
    Kwargs:
        path (str): The path to the data cube if datacube is a name and not a path+name.
        verbose (bool): If True, print messages to the terminal. (default: False)

    Returns:
        ndarray or None: The data cube with the model removed as a FITS file or None if write_fits is False.
        
    Raises:
        ValueError: If no data cube is provided.
        ValueError: If no model cube is provided.
        ValueError: If no path is provided.
        ValueError: If the method set does not match accepted values.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    if type(datacube)==str: #if the datacube is a string
        datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    if type(modelcube)==str: #if the modelcube is a string
        modelcube=__read_string(modelcube,'modelcube',**kwargs) #store the path to the model cube
    if maskcube is None: #if a mask cube is not given
        if verbose:
            warnings.warn('No mask cube provided: the removal will be done over the whole data cube')
    elif type(maskcube)==str: #if the maskcube is a string
        maskcube=__read_string(maskcube,'maskcube',**kwargs) #store the path to the mask cube
    if method not in ['all','blanking','b+s','negblank','subtraction']: #if wrong operation is given
        raise ValueError("ERROR: wrong method. Accepted values: ['all','blanking','b+s','negblank','subtraction']. Aborting")   
    threshold=blankthreshold #store the blanking threshold from the input parameters  
    output=removemod_out #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    #---------------   START THE FUNCTION   ---------------#    
    data,header=__load(datacube) #open the data cube
    model,_=__load(modelcube) #open the model cube
    mask,_=__load(maskcube) if maskcube is not None else np.ones(data.shape) #if a mask cube is not given open it or create a dummy mask cube
    if method in ['all','blanking','b+s']: #if all methods, or the blanking, or the blanking and subtraction must be performed
        data[np.where(model>threshold)]=np.nan #blank the data where the model is above the threshold
    if method in ['all','b+s','subtraction','negblank']: #if all methods, or the blanking and subtraction, or the subtraction or the negative blanking must be performed
        emission=np.where(mask>0) #store the emission coordinates
        data[emission]=data[emission]-model[emission] #subtract the model
    if method in ['all','negblank']: #if all methods, or the negative blanking must be performed
        mask[np.where(mask>0)]=1 #convert the mask into a 1/0 mask
        masked_data=data*mask #mask the data
        data[np.where(masked_data<0)]=np.nan #blank the negative data-model pixels
        
    if write_fits: #if the data must be write into a fits file
        hdu=fits.PrimaryHDU(data,header=header) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        hdul.writeto(output+'.fits',overwrite=True) #write the data into a fits file
        
    else:
        return data
        
#############################################################################################
def rotcurve(vfield='',pa=None,rotcenter=None,rotcurve_out='',save_csv=False,**kwargs):
    """Compute the rotation curve of a galaxy.
    
    Args:
        vfield (str): The name or path of the fits velocity field.
        pa (float): The position angle of the object in degrees.
        rotcenter (array-like): The x-y coordinates of the rotational center in pixels.
        rotcurve_out (str): The output folder/file name.
        save_csv (bool): If True, save the rotation curve as a CSV file.
    
    Kwargs:
        path (str): The path to the moment map if the momXmap is a name and not a path+name.
        vsys (float): The systemic velocity of the object in km/s.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        pixelres (float): The pixel resolution of the data in arcsec.
        asectokpc (float): The conversion factor from arcsec to kpc for plotting the spatial scale.
        objname (str): The name of the object.
        verbose (bool): If True, print messages and plot to the terminal.
    
    Returns:
        None
    
    Raises:
        ValueError: If no velocity field is provided.
        ValueError: If no path is provided.
        ValueError: If no galactic center is set.
        ValueError: If the galactic center is not given as x-y comma-separated coordinates in pixels.
        ValueError: If no output folder is provided.
    """    
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf')
    if vfield == '' or vfield is None: #if a moment 1 map is not given
        raise ValueError('ERROR: velocity field is not set: aborting!')
    if vfield[0] != '.': #if the moment 1 map name start with a . means that it is a path to the map (so differs from path parameter)
        path=kwargs.get('path',None) #if the path to the  moment 1 map is in kwargs
        if path == '' or path is None:
            raise ValueError('ERROR: no path to the velocity field is set: aborting!')
        else:
            vfield=path+vfield
    if pa is None: #if no position angle is given
        raise ValueError('ERROR: position angle is not set: aborting!')
    pa=np.radians(pa+90) #convert the position angle into effective angle in radians
    center=rotcenter #store the rotation center from the input parameters
    if center is None: #if no center is given
        raise ValueError('ERROR: no velocity field center is provided: aborting!')
    elif len(center) != 2: #if the center as wrong length
        raise ValueError('ERROR: wrongly velocity field center is provided. Use x-y comma-separated coordinates in pixel: aborting!')
    else: #store the rotation center from the input
        x0=center[0]-1 #convert the x-center into 0-indexing
        y0=center[1]-1 #convert the y-center into 0-indexing
    output=rotcurve_out+plot_format #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    #CHECK THE KWARGS#
    vsys=kwargs.get('vsys',None)
    pixunits=kwargs.get('pixunits',None)
    specunits=kwargs.get('specunits',None)
    pixelres=kwargs.get('pixelres',None)
    asectokpc=kwargs.get('asectokpc',None)
    objname=kwargs.get('objname','')

    #IMPORT THE DATA AND SETUP THE SPATIAL/SPECTRAL PROPERTIES#
    with fits.open(vfield) as V: #open the moment 1 map
        data=V[0].data #store the data
        header=V[0].header #store the header
    del V[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
    
    if vsys is None: #if the systemic velocity is not given
        vsys=np.nanmedian(data) #calculate the systemic velocity
    
    if pixunits is None: #if no spatial units are provided
        if 'CUNIT1'.casefold() in header: #check the spatial units keyword
            pixunits=header['CUNIT1'.casefold()] #store the spatial units
        else:
            pixunits='deg' #default to deg
            if verbose:
                warnings.warn('Cannot find the spatial units: set it to deg!')
                
    if specunits is None: #if no velocity units are provided
        if 'BUNIT'.casefold() in header: #check the velocity units keyword
            specunits=header['BUNIT'.casefold()] #store the velocity units
        else:
            specunits='m/s' #default to m/s
            if verbose:
                warnings.warn('Cannot find the velocity units: set it to m/s!')
                
    if pixelres is None: #if no pixel resolution is provided
        if 'CDELT1'.casefold() in header: #check the pixel resolution keyword
            pixelres=np.abs(header['CDELT1'.casefold()]) #store the pixel resolution
        else:
            if verbose:
                warnings.warn("Cannot find the spatial resolution: can't convert to kpc!")
    else:
        pixelres=np.abs(pixelres) #convert the pixel resolution to positive value
    #---------------   START THE FUNCTION   ---------------#
    #we want to extract the value of the velocity for each pixel (x,y) that falls on the major axis
    #we have to recover the range of x and the range of y covered by the major axis
    #we parameterize the major axis with a line y=tan(pa)*x+q
    #once we have the length r of the major axis on the full image (i.e., not limited to the size of the galaxy), we can recover the rotation curve by using x=r*cos(pa)+xmin and y=r*sin(pa)+ymin
    #from basic trigonometry, we can recover the x and y range
    if pa == np.radians(0) or pa == np.radians(180): #if the position angle is 'horizontal'
        xmin=0 #xmin is 0
        xmax=data.shape[1]-1 #xmax is the x-size of the image. -1 accounts for python 0-indexing
        ymin=y0 #ymin is the y-coordinate of the rotational center
        ymax=y0 #ymax is the y-coordinate of the rotational center
    elif pa in np.radians(np.arange(45,360,90)+90): #if the position angle is 45, 135, 225 or 315 deg
        xmin=0 #xmin is 0
        xmax=data.shape[1]-1 #xmax is the x-size of the image. -1 accounts for python 0-indexing
        ymin=0 #ymin is 0
        ymax=data.shape[0]-1 #ymax is the y-size of the image. -1 accounts for python 0-indexing
    elif pa == np.radians(90) or pa == np.radians(270): #if the position angle is 'vertical'
        xmin=x0 #xmin is the x-coordinate of the rotational center
        xmax=x0 #xmax is the x-coordinate of the rotational center
        ymin=0 #ymin is 0
        ymax=data.shape[0]-1 #ymax is y-size of the image. -1 accounts for python 0-indexing
    else: #we have to directly calculate the x and y range
        alpha=np.tan(pa) #calculate the slope of the major axis
        xmin=0 #xmin is 0
        xmax=data.shape[1]-1 #xmax is the x-size of the image. -1 accounts for python 0-indexing
        if pa < np.radians(45) or np.radians(180) < pa < np.radians(225):
            ymin=y0-x0*alpha #ymin is the intercept of the y-axis
            ymax=xmax*alpha+ymin #ymax is given by solving y=alpha*x+q
        elif pa > np.radians(315) or np.radians(135) < pa < np.radians(180):
            ymax=y0-x0*alpha #ymax is the intercept of the y-axis
            ymin=xmax*alpha+ymax #ymin is given by solving y=alpha*x+q
        else:
            q=y0-x0*alpha #calculate the intercept of the y-axis
            ymin=0 #ymin is 0
            ymax=data.shape[0]-1 #ymax is the y-size of the image. -1 accounts for python 0-indexing
            if np.radians(45) < pa < np.radians(90) or np.radians(225) < pa < np.radians(270):
                xmin=-q/alpha #xmin is given by solving y=alpha*x+q when y=0
                xmax=(ymax-q)/alpha #xmax is given by solving y=alpha*x+q when y=ymax
            else:
                xmin=(ymax-q)/alpha #xmin is given by solving y=alpha*x+q when y=ymax
                xmax=-q/alpha #xmax is given by solving y=alpha*x+q when y=0
    r=np.sqrt((xmax-xmin)**2+(ymax-ymin)**2) #calculate the length of the major axis line in the image
    x=np.arange(round(r))*np.abs(np.cos(pa))+xmin #define the x-axis
    y=np.arange(round(r))*np.abs(np.sin(pa))+ymin #define the y-axis
    if np.radians(90) < pa < np.radians (180) or np.radians(270) < pa < np.radians(360): #for that values of pa, y decreases as x increases; hence, flip the y-axis
        y=np.flip(y) #flip the y-axis to go from ymax to ymin

    rotcurve=data[y.astype(int)-1,x.astype(int)-1]-vsys #extract the rotation curve

    if pixelres is None: #if no pixel resolution is given
        radius=np.sqrt((x-x0)**2+(y-y0)**2) #radius from center in pixel
        units='pixel' #set the units to pixel
    else:
        radius=np.sqrt((x-x0)**2+(y-y0)**2)*pixelres  #radius from center in pixunits
        units=f'{pixunits}' #set the units to pixunits
    radius[rotcurve<=0]=-radius[rotcurve<=0] #set to negative values the radii of receding velocities
    
    recradius=radius[rotcurve>=0] #get the radius of the receding velocities
    appradius=radius[rotcurve<=0] #get the radius of the approaching velocities
    rec=rotcurve[rotcurve>=0] #store the receding rotation curve
    app=rotcurve[rotcurve<=0] #store the approaching rotation curve
    
    if np.cos(pa) < 0: #for those pa, the higher/lower receding/approaching velocity will be sampled first
        rec=np.flip(rec) #flip the receding velocities sothat the first element is 0
        recradius=np.flip(recradius) #flip the receding radii sothat the first element is 0
    else: #for those pa, the lower/higher receding/approaching velocity will be sampled first
        app=np.flip(app) #flip the approaching velocities sothat the first element is 0
        appradius=np.flip(appradius) #flip the approaching radii sothat the first element is 0
  
    if np.nanmin(np.abs(appradius)) < np.nanmin(recradius): #by definition, at radius = 0 the rotation velocity is 0.
        #However, it is almost impossible to have a 0 radius, hence, the lowest value of radius should be considered as 0.
        #This radius belongs to the approaching or receding side, consequently, to the other should be added a radius = 0.
        recradius=np.append(0,recradius)
        rec=np.append(0,rec)
    else:
        appradius=np.append(0,appradius)
        app=np.append(0,app)
      
    #we want to calculate the mean rotation curve, so we have to equal the size of the approaching and receding rotation curves
    if len(app) > len(rec): #if the approaching side is larger
        newapp=app #store a dummy variable used for csv
        diff=len(app)-len(rec) #calculate how much larger it is
        toadd=np.zeros(diff)*np.nan #create a nans array with size equal to the difference
        newrec=np.concatenate((rec,toadd)) #append the nans to the receding rotation curve
        newradius=appradius #the total radius is the one of the approaching side
    elif len(app) < len(rec): #if the receding side is larger
        newrec=rec #store a dummy variable used for csv
        diff=len(rec)-len(app) #calculate how much larger it is
        toadd=np.zeros(diff)*np.nan #create a nans array with size equal to the difference
        newapp=np.concatenate((app,toadd)) #append the nans to the approaching rotation curve
        newradius=recradius #the total radius is the one of the receding side
    else:
        newradius=np.nanmean((appradius,recradius),axis=0) #the radius is given by the mean of the approacing and receding radii
        newapp=app #copy the approaching rotation curve
        newrec=rec #copy the receding rotation curve

    total=np.nanmean((newrec,np.abs(newapp)),axis=0) #calculate the mean rotation velocity. flip is needed because the receding velocities are starting from the outermost value, while the approaching from the innermost
    
    if save_csv: #if the result must be saved to a csv file
        df=pd.DataFrame(columns=[f'RADIUS [{pixunits}]',f'VROT (APP) [{specunits}]',f'VROT (REC) [{specunits}]',f'VROT (TOT) [{specunits}]']) #create the dataframe
        df[f'RADIUS [{pixunits}]']=np.abs(newradius) #store the radius. abs needed since depending on which side is larger, the radius can be positive or negative
        df[f'VROT (APP) [{specunits}]']=newapp #store the approaching rotation curve
        df[f'VROT (REC) [{specunits}]']=newrec #store the receding rotation curve. flip needed since the  receding curve goes from larger to smaller values
        df[f'VROT (TOT) [{specunits}]']=total #store the global rotation curve        
        df.to_csv(output.replace(plot_format,'.csv'),index=False) #convert into a csv

    #---------------   DO THE PLOT   ---------------#
    if specunits=='m/s': #if the units are in m/s
        rec=rec/1000
        app=app/1000
        total=total/1000

    if pixelres is not None: #if the pixel resolution is given
        if pixunits == 'deg': #if the spatial unit is degree
            recradius=recradius*3600 #convert into arcsec
            appradius=appradius*3600 #convert into arcsec
            newradius=newradius*3600 #convert into arcsec
        elif pixunits == 'arcmin': #if the spatial unit is arcmin
            recradius=recradius*60 #convert into arcsec
            appradius=appradius*60 #convert into arcsec
            newradius=newradius*60 #convert into arcsec
        if asectokpc is not None: #if arcsec-to-kpc conversion is provided
            recradius=recradius*asectokpc #convert the radius from arcsec to kpc
            appradius=appradius*asectokpc #convert the radius from arcsec to kpc
            newradius=newradius*asectokpc #convert the radius from arcsec to kpc
            units='kpc' #set the units to kpc
            
    nrows=1 #number of rows in the atlas
    ncols=1 #number of columns in the atlas
    
    fig=plt.figure(figsize=(6*ncols,7*nrows)) #create the figure
    ax=fig.add_subplot(nrows,ncols,1) #create the subplot
    ax.plot(recradius,rec,c='red',label='receding side',lw=1.5,marker='o') #plot the receding rotation curve
    ax.plot(np.abs(appradius),np.abs(app),c='blue',label='Approaching side',lw=1.5,marker='o') #plot the approaching rotation curve
    ax.plot(np.abs(newradius),total,c='green',label='Global',lw=1.5,ls='--') #plot the total rotation curve
    ax.set_xlabel(f'Radius from center [{units}]') #set the x-axis label
    ax.set_ylabel('Velocity [km/s]') #set the y-axis label
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right') #set the legend
    
    fig.subplots_adjust(wspace=0.0,hspace=0.0) #fix the position of the subplots in the figure   
    fig.savefig(output,dpi=300,bbox_inches='tight') #save the figure
    
    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    else:
        plt.close()

#############################################################################################
def stacking(datacube='',mask2d='',vfield='',pa=None,inc=None,stackcenter=None,shape='cones',nregions=None,from_to=None,between_angles=None,weighting=None,stack_fluxrange='negative',stack_statistics='mad',diagnostic=False,periodicity=False,save_numpy=False,stack_out='',**kwargs):
    """Stack the spectra extracted from a given number of regions around a center starting from a minimum radius up to a maximum radius. Afterwards, it runs a source-finding algorithm on each stacked spectrum to check for detected lines. It optionally stores diagnostic plots of the stacking and the source-finding routines and also optionally stores relevant diagnostic fits file of the source-finding routine.

    Args:
        datacube (str/ndarray): Name or path+name of the fits data cube.
        mask2d (str): Name or path+name of the fits 2D emission mask. Set to None to disable the masking.
        vfield (str): Name or path+name of the fits velocity field to be used for the spectral alignment. Set to None to disable the alignment.
        pa (float): Object position angle in degrees.
        inc (float): Inclination of the object in degrees (0 deg means face-on).
        stackcenter (array-like): x-y comma-separated coordinates of the rotational center in pixels.
        shape (str): Shape of regions to stack. Available shapes:
            - cells: squares cells across the whole field.
            - cones: conic regions around a center and with width rmax-rmin.
            - concentric: concentric circles around a center with radius (rmax-rmin)/nregions.
        nregions (int): Number of regions from which the spectra are extracted and stacked.
        from_to (array-like): Comma-separated min and max radius from the galactic center in pixunits from which the spectra are extracted and stacked.
        between_angles (array-like): Comma-separated min and max angle from which the stacking regions are defined.
        weighting (str): Type of weight to apply during the stacking. Options:
            - None (default): The stacked spectrum will be averaged with the number of stacked spectra.
            - rms: The stacked spectrum will be averaged with the square of the rms of each stacked spectrum.
        stack_fluxrange (str): Flux range to be used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artifacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux.
        stack_statistics (str): Statistic to be used in the noise measurement process of the source finder. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is the fastest algorithm but the least robust one with respect to emission and artifacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artifacts.
        diagnostic (bool): Store all the diagnostic files and plots. Warning: the diagnostic might occupy large portions of the disk (default: False).
        save_numpy (bool): Store the stacked spectra and stacked rms array in the disk. Useful to calibrate the source finder without re-run each time the stacking (default: False).
        stack_out (str): The output folder/file name.
        
    Kwargs:
        path (str): Path to the data cube and the velocity field if one or both is a name and not a path+name.
        pbcorr (bool): Apply the primary beam correction if True. Note that in that case, you must supply a beam cube (Default: False).
        beamcube (str/ndarray): Name or path+name of the fits beam cube if pbcorr is True.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz  
        fluxunits (str): String with the flux units. Will be taken from the cube header if not provided.
        pixelres (float): Pixel resolution of the data. Will be taken from the cube header if not provided.
        spectralres (float): Data spectral resolution in specunits. Will be taken from the cube header if not provided.
        rms (float): RMS of the data cube as a float in fluxunits (Default: None).
        bmaj (float): Beam major axis in pixunits (Default: None).
        bmin (float): Beam minor axis in pixunits (Default: None).
        objname (str): Name of the object (Default: '').
        plot_format (str): Format type of the plots (pdf, jpg, png, ...) (Default: pdf).
        ctr_width (float): Line width of the contours (Default: 2).
        verbose (bool): Option to print messages and plot to terminal if True.
        **lines_finder_args: Arguments for the source finding (see the function lines_finder).

    Returns:
        A plot of the stacked spectra and the stacked regions, and arrays of each stacked spectrum, each stacked source, the stacked RMS, and the detection mask.
        
    Raises:
        ValueError: If mandatory inputs are missing.
        ValueError: If no galactic center is set.
        ValueError: If the galactic center is not given as x-y comma-separated coordinates in pixels.
        ValueError: If no output folder is provided.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf') #store the plot format
    datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    if mask2d is not None and (type(mask2d) is str and mask2d != ''): #if a 2D mask is provided
        mask2d=__read_string(mask2d,'mask2d',**kwargs) #store the path to the 2D mask
    if vfield is not None and (type(vfield) is str and vfield != ''): #if a velocity field is provided
        vfield=__read_string(vfield,'vfield',**kwargs) #store the path to the velocity field
    if shape in ['cones','concentric']: #if the stacking regions shape is cones or concentric
        center=stackcenter #store the rotation center from the input parameters
        inputs=np.array([pa,inc,center,nregions,from_to],dtype='object') #store the mandatory inputs
        inputsnames=['pa','inc','center','nregions','from_to'] #store the mandatory input names
    elif shape in ['cells']: #if the stacking regions shape is cells
        inputs=np.array([nregions],dtype='object') #store the mandatory inputs
        inputsnames=['nregions'] #store the mandatory input names
    if None in inputs: #check if one or more mandatory inputs are missing
        raise ValueError('ERROR: one or more mandatory inputs are missing ({}): aborting!'.format([inputsnames[i] for i in range(len(inputs)) if inputs[i]==None]).replace("'",""))
    if shape in ['cones','concentric']: #if the stacking regions shape is cones or concentric
        if len(center) != 2: #if the center as wrong length
            raise ValueError('ERROR: wrong galactic center is provided. Use x-y comma-separated coordinates in pixel: aborting!')
        else: #store the rotation center from the input
            x0=center[0]-1 #convert the x-center into 0-indexing
            y0=center[1]-1 #convert the y-center into 0-indexing
        if len(from_to) != 2: #if the min/max radius list has wrong length
            raise ValueError('ERROR: wrong min and max radius is provided. Use x-y comma-separated coordinates in pixunits: aborting!')
        else: #store the min/max radius from the input
            rmin=from_to[0] #store the min radius
            rmax=from_to[1] #store the max radius
    output=stack_out #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path):
            os.makedirs(path) #create the folder if the output folder does not exist
        plots_output=path+'stacked_spectra/' #create the directory for the stacked spectra plots
        if not os.path.exists(plots_output): #if the output folder does not exist
            os.makedirs(plots_output) #create the folder  
        if periodicity: #if the spectra periodicity must be computed
            periodicity_output=path+'periodicity_plots/' #create the directory for the stacked spectra plots
            if not os.path.exists(periodicity_output): #if the output folder does not exist
                os.makedirs(periodicity_output) #create the folder  
        if diagnostic: #if the diagnostic option is true
            diagnostic_output=path+'diagnostic/'
            if not os.path.exists(diagnostic_output): #if the output folder does not exist
                os.makedirs(diagnostic_output) #create the folder
    else: #if the output is a name
        plots_output=os.getcwd()+'stacked_spectra/' #create the directory for the stacked spectra plots
        if not os.path.exists(plots_output): #if the output folder does not exist
            os.makedirs(plots_output) #create the folder  
        if periodicity: #if the spectra periodicity must be computed
            periodicity_output=os.getcwd()+'periodicity_plots/' #create the directory for the stacked spectra plots
            if not os.path.exists(periodicity_output): #if the output folder does not exist
                os.makedirs(periodicity_output) #create the folder  
        if diagnostic: #if the diagnostic option is true
            diagnostic_output=os.getcwd()+'diagnostic/'
            if not os.path.exists(diagnostic_output): #if the output folder does not exist
                os.makedirs(diagnostic_output) #create the folder
        
    #CHECK THE KWARGS#
    pbcorr=kwargs.get('pbcorr',False) #store the apply pb correction option
    if pbcorr: #if the primary beam correction is applied
        beamcube=kwargs.get('beamcube',None) #store the data cube path from the input parameters
        if beamcube is None: #if the beam cube is not provided
            if verbose:
                warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')
            pbcorr=False #set the pbcorr option to False
        elif type(beamcube)==str: #if the beam cube is a string
            beamcube=__read_string(beamcube,'beamcube',**kwargs) #store the path to the beam cube
    pixunits=kwargs.get('pixunits',None) #store the spatial units
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    specunits=kwargs.get('specunits',None) #store the spectral units
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')  
    fluxunits=kwargs.get('fluxunits',None) #store the flux units
    pixelres=kwargs.get('pixelres',None) #store the spatial resolution
    spectralres=kwargs.get('spectralres',None) #store the spectral resolution
    bmaj=kwargs.get('bmaj',None) #store the beam major axis
    bmin=kwargs.get('bmin',None) #store the beam minor axis
    bpa=kwargs.get('bpa',None) #store the beam position angle
    rms=kwargs.get('rms',None) #store the rms
    objname=kwargs.get('objname','') #store the object name
    ctr_width=kwargs.get('ctr_width',2) #store the contour width       
    smooth_kernel=kwargs.get('smooth_kernel',None) #store the lines finder smoothing kernels
    sc_threshold=kwargs.get('sc_threshold',None) #store the lines finder threshold
    link_kernel=kwargs.get('link_kernel',None) #store the lines finder linker kernel
    rel_threshold=kwargs.get('rel_threshold',0) #store the lines finder reliability threshold
               
    #IMPORT THE DATA AND SETUP THE SPATIAL/SPECTRAL PROPERTIES#
    data,header=__load(datacube) #load the data cube
    if mask2d is not None and (type(mask2d) is str and mask2d != ''): #if a 2D mask is provided
        mask,_=__load(mask2d) #load the mask

    # CHECK FOR THE RELEVANT INFORMATION #
    #------------   CUBE PROPERTIES    ------------#
    prop=[pixunits,specunits,fluxunits,pixelres,spectralres,rms,bmaj,bmin] #list of cube properties values
    prop_name=['pixunits','specunits','fluxunits','pixelres','spectralres','rms','bmaj','bmin'] #list of cube properties names
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if one or more cube parameters are not given
        if verbose:
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            warnings.warn(f'I am missing some information: {not_found}. Running cubestat to retrieve them!')
        stats=cubestat(datacube,params_dict=dict(zip(prop_name,prop)),verbose=False) #calculate the statistics of the cube
        if pixunits is None: #if the spatial units are not given
            pixunits=stats['pixunits'] #take the value from the cubestat results
        if specunits is None: #if spectral units are not given
            specunits=stats['specunits'] #take the value from the cubestat results
        if fluxunits is None: #if flux units are not given
            fluxunits=stats['fluxunits'] #take the value from the cubestat results
        if pixelres is None: #if pixel resolution is not given
            pixelres=stats['pixelres'] #take the value from the cubestat results
        if spectralres is None: #if spectral resolution is not given
            spectralres=stats['spectralres'] #take the value from the cubestat results
        if rms is None: #if rms is not given
            rms=stats['rms'] #take the value from the cubestat results
        if bmaj is None: #if the beam major axis is not given
            bmaj=stats['bmaj'] #take the value from the cubestat results
        if bmin is None: #if the beam minor axis is not given
            bmin=stats['bmin'] #take the value from the cubestat results
    prop=[pixunits,specunits,fluxunits,pixelres,spectralres,rms,bmaj] #update the list of cube properties values
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if still one or more cube parameters are not given
        not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
        raise ValueError(f'ERROR: I am still missing some information: {not_found}. Please check the parameter!')
          
    #PREPARE THE KWARGS FOR THE RMS PLOT#
    converttoHI_kwargs={'pixelres':pixelres,'pixunits':pixunits,'spectralres':spectralres,'specunits':specunits,'beamarea':1.13*bmaj*bmin}
    
    #SETUP THE VELOCITY AXIS PROPERTIES#
    if 'CRVAL3' in header and 'CRPIX3' in header: #if the header has the starting spectral value
        v0=header['CRVAL3']-(header['CRPIX3']+1)*spectralres #store the starting spectral value
    else:
        raise ValueError('ERROR: no spectral value for starting channel was found. Aborting!')
    if specunits == 'm/s': #if the spectral units are m/s
        spectralres/=1000 #convert the spectral resolution to km/s
        v0/=1000 #convert the starting velocity to km/s
    nchan=data.shape[0] #store the number of channels
    if spectralres>0: #if the spectral resolution is positive
        v=np.arange(v0,v0+nchan*spectralres,spectralres) #define the spectral axis
    else:
        v=np.arange(v0+(nchan-1)*spectralres,v0-spectralres,-spectralres) #define the spectral axis
    if len(v) > nchan: #!! sometimes an additional channel is created. For the moment, this is a workaround
        v=v[:-1]
    if spectralres<0: #if the spectral resolution is negative
        flip=True #flip the spectra when stacking
    else:
        flip=False
                
    #---------------   START THE FUNCTION   ---------------#  
    #APPLY THE PRIMARY BEAM CORRECTION#
    if pbcorr: #if the primary beam correction must be applied
        if verbose:
            print('Applying the primary beam correction\n----------------------------------------------')
        beamcube,_=__load(beamcube) #load the pb cube
        if len(beamcube.shape)==2: #if it is a beam map
            beamcube=np.repeat(beamcube[None,:],data.shape[0],axis=0) #extend the beam map to mach the size of the datacube
        if beamcube.shape != data.shape: #if the pb cube shape is not the same of the data cube
            if verbose:
                warnings.warn(f'primary beam cube shape {beamcube.shape} and data cube shape {data.shape} mismatch. Cannot apply the primary beam correction; flux values will not be corrected for the primary beam.')
        else:
            data/=beamcube #apply the primary beam correction
            
    #MASK THE EMISSION IN THE CUBE#
    if mask2d is not None and (type(mask2d) is str and mask2d != ''): #if a 2D mask is provided
        if verbose:
            print('Masking the emission\n----------------------------------------------')
        if mask.shape != data.shape[1:3]: #if the mask shape is not the same of the data cube
            if verbose:
                warnings.warn(f'mask spatial shape {mask.shape} and data spatial shape {data.shape[1:3]} mismatch. Cannot apply the detection mask; make sure the supplied data cube is already masked from the emission.')
        else:
            mask=mask.astype('float32') #convert the mask into a float array to assign nan values
            mask[mask==0]=np.nan #set to nan the 0 voxel
            mask[~np.isnan(mask)]=1 #set to 1 the non-nan voxel
            mask=np.array([mask]*data.shape[0]) #transform the 2D mask into a cube
            data*=mask #mask the emission from the data
        
            if diagnostic:
                if verbose:
                    print('Writing the masked cube\n----------------------------------------------')
                hdu=fits.PrimaryHDU(data.astype('float32'),header=header) #create the primary HDU
                hdul=fits.HDUList([hdu]) #make the HDU list
                if objname != '': #if the object name is given
                    hdul.writeto(diagnostic_output+f'{objname}_masked_cube.fits',overwrite=True) #write the data into a fits
                else:
                    hdul.writeto(diagnostic_output+'masked_cube.fits',overwrite=True) #write the data into a fits 

    #SHUFFLE THE DATA TO ALIGN THE CUBE#
    if vfield is not None: #if a velocity field is provided
        if verbose:
            print('Aligning the spectra\n----------------------------------------------')
        if specunits == 'm/s': #if the spectral units are m/s
            data=cubedo(cubedo=data,operation='shuffle',vfield=vfield,v0=v0*1000,spectralres=spectralres*1000,verbose=True)
        else:
            data=cubedo(cubedo=data,operation='shuffle',vfield=vfield,v0=v0,spectralres=spectralres,verbose=True)
        aligned=True #set the aligned switch for the plots to True
        
        if diagnostic:
            if verbose:
                print('Writing the shuffled cube\n----------------------------------------------')
            header['CRPIX3']=(nchan//2)+1 #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred. +1 is needed for account the stupid python 0-counting
            header['CRVAL3']=0. #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred
            hdu=fits.PrimaryHDU(data.astype('float32'),header=header) #create the primary HDU
            hdul=fits.HDUList([hdu]) #make the HDU list
            if objname != '': #if the object name is given
                hdul.writeto(diagnostic_output+f'{objname}_shuffled_cube.fits',overwrite=True) #write the data into a fits
            else:
                hdul.writeto(diagnostic_output+'shuffled_cube.fits',overwrite=True) #write the data into a fits 
            
        #REDEFINE THE SPECTRAL AXIS#
        if spectralres>0: #if the spectral resolution is positive
            v0=-(nchan//2)*spectralres #redefine the starting velocity
            v=np.arange(v0,v0+nchan*spectralres,spectralres) #redefine the spectral axis
        else:
            v0=(nchan//2)*spectralres #redefine the starting velocity
            v=-np.arange(v0,v0-nchan*spectralres,-spectralres) #redefine the spectral axis
        if len(v) > nchan: #!! sometimes an additional channel is created. For the moment, this is a workaround
            v=v[:-1]
    else:
        aligned=False #set the aligned switch for the plots to True
    
    #ASSIGN THE PIXEL TO THE CORRECT REGION#
    if verbose:
        print('Assigning pixel to each stacking region\n----------------------------------------------')
    if shape=='cones':
        rmin/=np.abs(pixelres) #convert the max radius into pixel
        rmax/=np.abs(pixelres) #convert the max radius into pixel
        xvalid,yvalid=__assign_to_cones(data,nregions,pa,inc,rmin,rmax,x0,y0) #assign the pixel to the cones     
    elif shape=='concentric':
        rmin/=np.abs(pixelres) #convert the max radius into pixel
        rmax/=np.abs(pixelres) #convert the max radius into pixel
        xvalid,yvalid=__assign_to_concentric(data,nregions,pa,inc,rmin,rmax,between_angles,x0,y0) #assign the pixel to the concentric circles
    elif shape=='cells':
        xvalid,yvalid=__assign_to_cells(data.shape[2],data.shape[1],nregions) #assign the pixel to the cells
    
    #PREPARE THE FIGURES FOR THE RESULT#
    wcs=WCS(header,naxis=2) #store the wcs

    fig1=plt.figure(figsize=(16,8)) #create the figure for the stacked positions
    ax1=fig1.add_subplot(121,projection=wcs) #create the subplot for the stacked pixels
    ax1.imshow(np.nansum(data,axis=0),origin='lower',aspect='equal',cmap='gray_r',norm=cl.PowerNorm(gamma=0.3,vmin=0,vmax=0.75*np.nanmax(np.nansum(data,axis=0)))) #plot the image
    ax1.set_xlabel('RA') #set the x-axis label
    ax1.set_ylabel('DEC') #set the y-axis label 
    if shape in ['cones','concentric']:
        xmin=round(x0-rmax) #store the min x-coordinate
        xmax=round(x0+rmax) #store the max x-coordinate
        ymin=round(y0-rmax) #store the min y-coordinate
        ymax=round(y0+rmax) #store the max y-coordinate  
        xlim=(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin)) #set the xlim
        ylim=(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin)) #set the ylim
        ax1.set_xlim(xlim) #set the xlim
        ax1.set_ylim(ylim) #set the ylim 
           
    cm=cl.LinearSegmentedColormap.from_list("",['black','orange','lime','deeppink','blue','yellow','cyan','red'])
    colors=cm(np.linspace(0,1,nregions))
        
    #DO THE STACKING#
    if verbose and diagnostic:
        warnings.warn("You activated the diagnostic mode. Stacking may take a while and stores large quantity of plots in your disk! If there will be a serious memory leak, run 'import matplotlib' and matplotlib.use('agg') before using the stacking function.")
    if verbose and nregions>50:
        warnings.warn(f"You selected a large ({nregions}) number of regions. Stacking may take a while and stores large quantity of plots in your disk! If there will be a serious memory leak, run 'import matplotlib' and matplotlib.use('agg') before using the stacking function.")
    validmap=np.zeros((data.shape[1],data.shape[2]),dtype=int) #initialize a map for showing the stacked positions. We want to draw contours over it, so the idea is to set each valid x,y pixel of the region i to have the value of i and then draw the contours on the map with level equal to i
    for i in range(nregions):
        if diagnostic:
            diagnostic_plots_output=diagnostic_output+f'Region_{i+1}_spectra/'
        else:
            diagnostic_plots_output=None
        if xvalid[i] == [] and yvalid[i] == []: #if no valid pixels have been found
            warnings.warn(f'no valid pixel found for region {i+1}. Skip to the next region')
            spectrum=np.zeros(data.shape[0])*np.nan #set the output spectrum as nan
            rms=[np.nan] #set the output rms as nan
            mask=np.zeros(data.shape[0])*np.nan #set the output spectrum as nan
        else: #proceed
            validmap[yvalid[i],xvalid[i]]=i+1 #store the value of i (+1 accounts for 0-counting) in the validmap
            ax1.contour(validmap,levels=[i,i+1],colors=cl.rgb2hex(colors[i]),linewidths=0.5) #draw the contour to show which pixels belong to the i-region
            text=ax1.text((np.nanmin(xvalid[i])+np.nanmax(xvalid[i]))/2,(np.nanmin(yvalid[i])+np.nanmax(yvalid[i]))/2,f'{i+1}',ha='center',va='center',c=colors[i]) #add the spectrum ID to the cell center
            __auto_fit_fontsize(text,np.nanmax(xvalid[i])-np.nanmin(xvalid[i]),np.nanmax(yvalid[i])-np.nanmin(yvalid[i]),fig=fig1,ax=ax1)

            #CREATE THE STACKED SPECTRUM#
            if verbose:
                print(f'Computing the stacked spectrum for region {i+1}\n----------------------------------------------')
            spectrum,rms,exp=__stack(data,xvalid[i],yvalid[i],weighting,stack_fluxrange=stack_fluxrange,stack_statistics=stack_statistics,flip=flip,diagnostic=diagnostic,v=v,fluxunits=fluxunits,c=colors[i],aligned=aligned,outdir=diagnostic_plots_output,plot_format='.jpg') #do the stacking
            if np.all(np.isnan(spectrum)): #if the spectrum contains only nans
                rms=[np.nan] #set the rms to be nan
            #PREPARE THE VARIABLES FOR THE LINES FINDER
            if 'stacked_spectra' in locals(): #if not at the first iteration
                stacked_spectra.append(spectrum) #append the latest stacking result
                stacked_rms.append(rms[-1]) #append the rms value of the latest stacked spectrum
                stacked_rms_list.append(rms) #append the list of rms value for each stacking iteration
                stacked_exp.append(exp) #append the expected rms value
                n_spectra.append(len(rms)) #append the number of stacked spectra (needed later for plot the result)
            else:
                stacked_spectra=[spectrum] #store the stacking result
                stacked_rms=[rms[-1]] #store the rms value of the stacked spectrum
                stacked_rms_list=[(rms)] #store the list of rms value for each stacking iteration
                stacked_exp=[exp] #store the expected rms value
                n_spectra=[len(rms)] #store the number of stacked spectra (needed later for plot the result)
        
    #SOURCE FINDING#
    if verbose:
        print('Running the lines finder on the stacked spectra\n----------------------------------------------')                
    mask,smooth_rms,reliable_ids=lines_finder(stacked_spectra,stacked_rms,**kwargs) #run the source finder
    
    #ADD THE RESULT OF THE FINDER TO THE DETECTION MAP#
    if verbose:
        print('Creating the detection map and the output plots\n----------------------------------------------') 
    detection_map=np.zeros((data.shape[1],data.shape[2]))*np.nan #initialize the detection map where we are going to show if a region has detections in it
    cmap=cl.ListedColormap([(0.5,0,0),(0,0.5,0),(0.5,0.5,0)]) #define the colormap for the detections
    ax2=fig1.add_subplot(122,projection=wcs) #create the subplot for the stacked pixels
    ax2.imshow(np.nansum(data,axis=0),origin='lower',aspect='equal',cmap='gray_r',norm=cl.PowerNorm(gamma=0.3,vmin=0,vmax=0.75*np.nanmax(np.nansum(data,axis=0)))) #plot the image
    ax2.set_xlabel('RA') #set the x-axis label
    ax2.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
    if shape in ['cones','concentric']:
        ax2.set_xlim(xlim) #set the xlim
        ax2.set_ylim(ylim) #set the ylim 
    if rel_threshold > 0 and  rel_threshold is not None: #if the reliability was computed
        for j in reliable_ids:
            detection_map[yvalid[j],xvalid[j]]=1 #add to the detection map only the reliable sources
            
    if periodicity: #if the periodicity plots must be made
        mags_list=[] #create the list to store the periodicity amplitude of each spectrum and each smoothing iteration
        for k in range(len(smooth_kernel)): #for each smoothing kernel
            mags_list.append([]) #extend the periodicity amplitude list. We are going to store in each slot of the list the amplitudes of all the spectra for a given smoothing kernel
    for i in range(nregions): #reiter over the regions to plot and store the result of the line finder. I can't do that internally the first iteration because of the way the lines finder is written
        plt.ioff() #disable the interactive plotting 
        if not np.all(np.isnan(stacked_spectra[i])): #if the stacked spectrum is not empty         
            if rel_threshold <=0 or rel_threshold is None: #if the reliability was not computed
                dummy=mask[i].copy() #create a copy of the detection mask
                if np.all(np.isnan(dummy)): #if no sources are detected
                    pass
                else:
                    dummy=dummy[~np.isnan(dummy)] #get the sources
                    if np.all(dummy<0): #if they are all negative
                        detection_map[yvalid[i],xvalid[i]]=0
                    elif np.all(dummy>0): #if they are all positive
                        detection_map[yvalid[i],xvalid[i]]=1
                    else: #if the sources are positive and negative
                        detection_map[yvalid[i],xvalid[i]]=2
            
            #PLOT THE STACKED SPECTRA#
            smooth_rms[i]=smooth_rms[i][-1] if smooth_rms[i] is not None else None #get the final value of the smooth rms if it is not None (like when no smoothing is applied)
            nrows=1 #number of rows in the plot
            ncols=2 #number of columns in the plot    
            fig2=plt.figure(figsize=(8*ncols,4*nrows)) #create the figure for the stacked spectra
            fig2=__plot_stack_result(v,stacked_spectra[i],stacked_rms_list[i],stacked_exp[i][:n_spectra[i]],smooth_rms[i],fluxunits,mask=np.abs(mask[i]),nrows=nrows,ncols=ncols,color=colors[i],aligned=aligned,idx=1,smooth_kernel=smooth_kernel,sc_threshold=sc_threshold,link_kernel=link_kernel,**converttoHI_kwargs)
            fig2.savefig(plots_output+f'stacked_spectrum_{i+1}'+'.jpg',dpi=300,bbox_inches='tight')
            plt.close(fig=fig2)
            
            #DO THE PERIODICITY PLOTS#
            if periodicity: #if the periodicity plots must be made
                if verbose and i == 0:
                    warnings.warn("You activated the periodicity plots. Large quantity of plots may be stored in your disk! If there will be a serious memory leak, run 'import matplotlib' and 'matplotlib.use('agg')' before using the stacking function.")
                    print(f'Computing the periodicity of the stacked spectra\n----------------------------------------------')
                sampling=1/2 #sampling frequency for the FFT in order to convert frequency to channels when doing the plot
                freqs=1/(2*sampling*np.fft.rfftfreq(len(stacked_spectra[i]),sampling)[1:]) #create the channel axis from the time axis
                dummy_mask=mask[i].copy() #create a copy of the detection mask. We want to invert it in order to mask the detected emission
                dummy_mask[~np.isnan(dummy_mask)]=0 #set to 0 the non-NaN values
                dummy_mask[np.isnan(dummy_mask)]=1 #set to 1 the NaN values
                dummy_mask[dummy_mask==0]=np.nan #set to NaN the 0-values
                
                alpha=np.linspace(0,1,len(smooth_kernel)+1)[1:] if len(smooth_kernel)>1 else 1 #set the transparency of the plots based on the number of smoothing kernels
                lw=np.arange(len(smooth_kernel))+1 #set the linewidth
                
                period_fig=plt.figure(figsize=(10,10)) #create the figure
                
                period_ax1=period_fig.add_subplot(211) #add the subplot for the spectrum
                period_ax1.set_title('Spectrum') #add the title
                period_ax1.axhline(y=0,c='black',lw=1,ls='--')
                period_ax1.set_xlabel('Channel')
                period_ax1.set_ylabel(f'Amplitude [{fluxunits}]')
                period_ax2=period_fig.add_subplot(212) #add the subplot for the FFT
                period_ax2.set_title('FFT') #add the title
                period_ax2.set_xlabel('Periodicity [Channel]')
                period_ax2.set_ylabel('Amplitude [a.u.]')
                
                for k in range(len(smooth_kernel)): #for each kernel in the smoothing kernel
                    dummy_spec=conv.convolve(stacked_spectra[i],conv.Box1DKernel(smooth_kernel[k])) #convolve the stacked spectrum masked from the detected emission
                    mags=abs(np.fft.rfft(dummy_spec)[1:]) #compute the FFT and exclude the first element, which is the 0-channel periodicity. Also, take the absolute value beause we don't care about the phase
                    mags_list[k].append(mags) #append the amplitudes of the given smoothing kernel

                    #PERIODICITY PLOTS OF THE STACKED SPECTRUM i#
                    period_ax1.plot(dummy_spec,c=colors[i],label=f'{smooth_kernel[k]} channels smoothing',alpha=alpha[k],lw=lw[k]) #plot the spectrum
                    period_ax2.plot(freqs,mags,c=colors[i],alpha=alpha[k],lw=lw[k]) #plot the FFT
                
                period_ax1.legend(loc='upper right')
                period_ax1.set_xlim(0,len(stacked_spectra[i]))
                period_ax2.set_xlim(freqs[-1],freqs[0])
                period_ax2.set_xscale('log')
                period_fig.subplots_adjust(hspace=0.3)
                period_fig.savefig(periodicity_output+f'spectrum_periodicity_{i+1}.jpg',dpi=300,bbox_inches='tight')
                plt.close(fig=period_fig)
        
        #PREPARE THE VARIABLE TO RETURN#
        if 'source_spectra' in locals(): #if not at the first iteration
            source_spectra.append(np.abs(mask[i])*stacked_spectra[i]) #append the latest source finder result
            source_mask.append(mask[i]) #append the detection mask
        else:
            source_spectra=[np.abs(mask[i])*stacked_spectra[i]] #store the source finder result
            source_mask=[mask[i]] #store the detection mask
    plt.ion() #re-enable the interactive plotting 
    
    #PLOT THE DETECTION MAP#
    ## AT THE MOMENT THE FINAL DETECTION MAP IS NOT FILTERED FROM NON-GAUSSIAN SPECTRA ##
    im=ax2.imshow(detection_map,origin='lower',aspect='equal',cmap=cmap,alpha=0.75,vmin=0,vmax=2) #plot the detection map
    if rel_threshold<=0 or rel_threshold is None: #if the reliability was not computed
        #adding a colorbar will shrink the corresponding image. To keep the images to the same size, I have to create a new axes and store the colrobar in it
        cax=fig1.add_axes([0.9,0.125,0.02,0.755]) #create an axes on the right side of the figure
        cbar=fig1.colorbar(im,cax=cax) #add the colorbar to this new axes
        cbar.ax.get_yaxis().set_ticks([]) #blank the colorbar ticks
        for j,label in enumerate(['Only neg.','Only pos.','Pos. and neg.']): #for each type of detection
            cbar.ax.text(1,0.66*j+0.33,label,ha='center',va='center',rotation=90) #place the color label
            #1 e' (vmax-vmin)/2
            #0.66 e' (vmax-vmin)/(n_colors)
            #0.33 e' il numero sopra diviso 2
        cbar.ax.set_ylabel('For detection limit look at stacked spectra',rotation=90) #add the colorbar label
        cbar.ax.get_yaxis().labelpad=10 #fix the position of the colorbar
    fig1.subplots_adjust(wspace=0)
    hdu=fits.PrimaryHDU(detection_map.astype('float32'),header=wcs.to_header()) #create the primary HDU
    hdul=fits.HDUList([hdu]) #make the HDU list 
    
    #PLOT THE GLOBAL PERIODICITY PLOT#
    if periodicity:
        fig2=plt.figure(figsize=(10,5))
        ax=fig2.add_subplot()
        if objname != '': #if the object name is given
            ax.set_title(f'{objname} stacked spectra median FFT')
        for k in range(len(smooth_kernel)): #for each smoothing kernel
            c='b' if k == 0 else 'r' if k == 1 else 'g' #set the color of the plot
            ax.plot(freqs,np.nanmedian(np.array(mags_list[k]),axis=0),c=c,alpha=alpha[k],lw=lw[k],label=f'{smooth_kernel[k]} channels smoothing')
        ax.set_xlabel('Periodicity [Channel]')
        ax.set_ylabel('Amplitude [a.u.]')
        ax.set_xlim(freqs[-1],freqs[0])
        ax.set_xscale('log')
        ax.legend()
        fig2.savefig(periodicity_output+f'{objname}_total_fft'+plot_format,dpi=300,bbox_inches='tight') if objname != '' else fig2.savefig(periodicity_output+'Total_fft'+plot_format,dpi=300,bbox_inches='tight')
    
    #SAVE THE FIGURE AND FITS FILE#
    fig1.savefig(output+'_stacked_positions'+plot_format,dpi=300,bbox_inches='tight') #save the detection map figure
    hdul.writeto(output+'_detection_positions.fits',overwrite=True) #save the detection map as fits file

    if verbose:
        plt.show()
    plt.close('all')
    
    if save_numpy: #if the variables must be saved as numpy arrays in the disk
        np.save(output+'stacked_spectra',stacked_spectra) #save the stacked spectra
        np.save(output+'stacked_rms',stacked_rms) #save the stacked rms
    
    return stacked_spectra,source_spectra,stacked_rms,source_mask
                      
#############################################################################################
def velfi(vfield='',radii=None,vrot=None,pa=None,inc=None,vrad=None,vsys=None,vcenter=None,extend_only=False,correct=False,velfi_out='',**kwargs):
    """Compute a synthetic velocity field or extend a given velocity field.
    
    Args:
        vfield (str): Name or path+name of the measured velocity field.
        radii (float/array-like): Radii in pixel units at which the rotation velocity is measured.
        vrot (float/array-like): Rotation velocities (subtracted from the systemic velocities) in specunits for the radii.
        pa (float/array-like): Position angle of the object in degrees (counter-clockwise from the North to the most receding velocity).
        inc (float/array-like): Inclination of the object in degrees (0 deg means face-on).
        vrad (float/array-like): Expansion velocity for the radii.
        vsys (float/array-like): Systemic velocity for the radii.
        center (array-like): Comma-separated value for the x and y coordinates of the rotation center in pixels.
        extend_only (bool): Flag to extend a given velocity field (True) or write a new one from scratch (False).
        correct (bool): Flag to correct the input rotation velocities for the inclination angles (True) or not (False).
        velfi_out (str): The output folder/file name.
    
    Kwargs:
        path (str): Path to the FITS velocity field if 'vfield' is a name and not a path+name.
        pixelres (float): Pixel resolution of the data in pixel units.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        plot_format (str): Format type of the plots (pdf, jpg, png, ...) (Default: pdf).
        verbose (bool): Option to print messages and plots to the terminal if True (Default: False).
    
    Returns:
        Synthetic velocity field as a FITS file.
    
    Raises:
        ValueError: If mandatory inputs are missing.
        ValueError: If no path to the velocity field is provided.
        ValueError: If incorrect spatial units are provided.
        ValueError: If incorrect spectral units are provided.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    plot_format='.'+kwargs.get('plot_format','pdf') #store the plot format
    if vfield == '' or vfield is None: #if a moment 1 map is not given
        raise ValueError('ERROR: velocity field is not set: aborting!')
    center=vcenter #store the rotational center
    inputs=np.array([radii,vrot,pa,inc,vrad,vsys,center],dtype='object') #store the mandatory inputs
    inputsnames=['radii','vrot','pa','inc','vrad','vsys','center'] #store the mandatory input names
    if None in inputs: #check if one or more mandatory inputs are missing
        raise ValueError("ERROR: one or more mandatory inputs are missing ({}): aborting!".format([inputsnames[i] for i in range(len(inputs)) if inputs[i]==None]).replace("'",""))
    vfield=__read_string(vfield,'vfield',**kwargs) #store the path to the velocity field
    if len(center)!=2: #check that the rotation center has the correct format
        raise ValueError('ERROR: Please provide the rotation center in the format [x0,y0]. Aborting!')
    else: #store the rotation center from the input
        x0=center[0]-1 #convert the x-center into 0-indexing
        y0=center[1]-1 #convert the y-center into 0-indexing
    output=velfi_out #store the output directory/name from the input parameters
    if output in ['',None]: #if no output is provided
        raise ValueError('ERROR: no output name set: aborting')
    if output[0]=='.': #if the output name start with a . means that it contains a path
        path=''.join(a+'/' for a in output.split('/')[:-1]) #recover the path from the output string
        if not os.path.exists(path): os.makedirs(path) #create the folder if the output folder does not exist
    #CHECK THE KWARGS#
    pixunits=kwargs.get('pixunits',None) #store the spatial units
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    specunits=kwargs.get('specunits',None) #store the spectral units
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')  
    pixelres=kwargs.get('pixelres',None) #store the spatial resolution
        
    #PREPARE THE INPUT#
    nradii=len(radii) #get the number of radii
    #---------------   POSITION ANGLE  ---------------#
    if type(pa) is list: #if the position angles are given as list
        pa=np.array(pa) #convert into array
    if type(pa) is np.ndarray: #if the position angles are array
        if len(pa) < nradii: #if the number of position angles is less than the number of radii
            pa=np.concatenate((pa,np.repeat(pa[-1],nradii-len(pa)))) #extend pa by the last value to match the number of radii
    else:
        pa=np.repeat(pa,nradii) #extend pa to match the number of radii
    pa-=180 #FOR SOME REASONS I NEED THIS LINE, OTHERWISE THE FIELD IS INVERTED
    pa=np.radians(pa) #convert the position angle into radians
    #---------------   INCLINATION ANGLE   ---------------#
    if type(inc) is list: #if the inclination angles are given as list
        inc=np.array(inc) #convert into array
    if type(inc) is np.ndarray: #if the inclination angles are array
        if len(inc) < nradii: #if the number of inclination angles is less than the number of radii
            inc=np.concatenate((inc,np.repeat(inc[-1],nradii-len(inc)))) #extend inc by the last value to match the number of radii
    else:
        inc=np.repeat(inc,nradii) #extend inc to match the number of radii
    inc=np.radians(inc) #convert the inclination angle into radians
    #---------------   EXPANSION VELOCITY   ---------------#
    if type(vrad) is list: #if the expansion velocities are given as list
        vrad=np.array(vrad) #convert into array
    if type(vrad) is np.ndarray: #if the expansion velocities are array
        if len(vrad) < nradii:  #if the number of expansion velocities is less than the number of radii
            vrad=np.concatenate((vrad,np.repeat(vrad[-1],nradii-len(vrad)))) #extend vrad by the last value to match the number of radii
    else:
        vrad=np.repeat(vrad,nradii) #extend vrad to match the number of radii
    #---------------   SYSTEMIC VELOCITY   ---------------#
    if type(vsys) is list: #if the systemic velocities are given as list
        vsys=np.array(vsys) #convert into array
    if type(vsys) is np.ndarray: #if the systemic velocities are array
        if len(vsys) < nradii: #if the systemic velocities are array
            vsys=np.concatenate((vsys,np.repeat(vsys[-1],nradii-len(vsys)))) #extend vsys by the last value to match the number of radii
    else:
        vsys=np.repeat(vsys,nradii) #extend vsys to match the number of radii
    #---------------   ROTATION VELOCITY   ---------------#
    if type(vrot) is list: #if the rotation velocities are given as list
        vrot=np.array(vrot) #convert into array
    if type(vrot) is np.ndarray: #if the rotation velocities are array
        if len(vrot) < nradii: #if the number of rotation velocities is less than the number of radii
            vrot=np.concatenate((vrot,np.repeat(vrot[-1],nradii-len(vrot)))) #extend vrot by the last value to match the number of radii
    else:
        vrot=np.repeat(vrot,nradii) #extend vrot to match the number of radii
    if correct: #if the rotation velocities must be corrected for inc and vsys
        vrot=vrot/np.sin(inc)
    
    #IMPORT THE DATA AND SETUP THE SPATIAL/SPECTRAL PROPERTIES#
    field,header=__load(vfield) #load the velocity field fits file
        
    if pixunits is None: #if no spatial units are provided
        if 'CUNIT1'.casefold() in header: #check the spatial units keyword
            pixunits=header['CUNIT1'.casefold()] #store the spatial units
        else:
            pixunits='deg' #default to deg
            if verbose:
                warnings.warn('Cannot find the spatial units: set it to deg!')
    if specunits is None: #if no velocity units are provided
        if 'BUNIT'.casefold() in header: #check the velocity units keyword
            specunits=header['BUNIT'.casefold()] #store the velocity units
        else:
            specunits='m/s' #default to m/s
            if verbose:
                warnings.warn('Cannot find the velocity units: set it to m/s!')
    if pixelres is None: #if no pixel resolution is provided
        if 'CDELT1'.casefold() in header: #check the pixel resolution keyword
            pixelres=header['CDELT1'.casefold()]#store the pixel resolution
        else:
            raise ValueError('ERROR: No pixel resolution is found: aborting!')
            
    if extend_only: #if the velocity field must be extended
        result=field.copy() #initialize the output as a copy of the velocity field
    else:
        result=field.copy()*np.nan #initialize the output as nans
     
    #---------------   START THE FUNCTION   ---------------#
    rmax=radii[-1]/np.abs(pixelres) #convert the max radius into pixel
    xmin=round(x0-rmax) #store the min x-coordinate
    if xmin<0: #if xmin is less than 0
        xmin=0 #set to 0
    xmax=round(x0+rmax) #store the max x-coordinate
    if xmax>=field.shape[1]: #if xmax is greater than the x-shape of the data
        xmax=field.shape[1]-1 #set to the x-shape of the data
    ymin=round(y0-rmax) #store the min y-coordinate
    if ymin<0: #if ymin is less than 0
        ymin=0 #set to 0
    ymax=round(y0+rmax) #store the max y-coordinate
    if ymax>=field.shape[0]: #if xmax is greater than the y-shape of the data
        ymax=field.shape[0]-1 #set to the y-shape of the data   
    yrange=np.arange(ymin,ymax+1) #define the y-range
    xrange=np.arange(xmin,xmax+1) #define the x-range
    with tqdm(desc='Pixel computed',total=len(xrange)*len(yrange)) as pbar:
        for i in yrange:
            for j in xrange:
                if np.isnan(result[i,j]): #if the output in (i,j) is empty
                    x=(j-x0)*pixelres #convert the x-position of the pixel w.r.t. the center in arcsec
                    y=(i-y0)*pixelres #convert the y-position of the pixel w.r.t. the center in arcsec
                    r=np.sqrt(x**2+y**2) #calculate the distance from the center in arcsec
                    if r <= radii[nradii-1]: #if the distance is less or equal the outer radius
                        d1=__radius(radii[0],x,y,pa[0],inc[0]) #calculate the radius of (x,y) compared to the innermost radius
                        d2=__radius(radii[nradii-1],x,y,pa[nradii-1],inc[nradii-1]) #calculate the radius of (x,y) compared to the outermost radius
                        if d1*d2 <= 0: #if one of the distance is negative, means that the points resides between the innermost and outermost radius, hence, it is okay to calculate the rotation velocity at that point
                            for k in np.arange(1,nradii+1,1): #find the radius[i] corresponding to that point
                                d2=__radius(radii[k],x,y,pa[k],inc[k]) #keep calculate the distance between the radius of (x,y) and the k-radius
                                if d1*d2 > 0: #when both distances are positive, means that we have reached the outermost radius
                                    d1=d2 #set the two distances to be equal
                                else:
                                    break
                            R=(radii[k]*d1-radii[k-1]*d2)/(d1-d2)
                            w1=(radii[k-1]-R)/(radii[k-1]-radii[k]) #calculate the weight of the radius k-1
                            w2=1-w1 #calculate the weight of the radius k
                            posai=w1*pa[k]+w2*pa[k-1] #calculated the weighted position angle
                            incli=w1*inc[k]+w2*inc[k-1] #calculated the weighted inclination angle
                            vroti=w1*vrot[k]+w2*vrot[k-1] #calculated the weighted rotation velocity
                            vradi=w1*vrad[k]+w2*vrad[k-1] #calculated the weighted expansion velocity
                            vsysi=w1*vsys[k]+w2*vsys[k-1] #calculated the weighted systemic velocity
                            if R == 0:
                                cost=1
                                sint=0
                            else:
                                cost=(-x*np.sin(posai)+y*np.cos(posai))/R
                                sint=((-x*np.cos(posai)-y*np.sin(posai))/R)/np.cos(incli)
                            result[i,j]=vsysi+(vroti*cost+vradi*sint)*np.sin(incli) #calculate the rotation velocity
                            pbar.update(1)
    
    #DO THE PLOT#
    wcs=WCS(header)
    
    nrows=1
    ncols=2
    
    lim=np.where(~np.isnan(result)) #get the (x,y) values of not-nans in the output
    xmin=np.nanmin(lim[1]) #store xmin
    xmax=np.nanmax(lim[1]) #store xmax
    ymin=np.nanmin(lim[0]) #store ymin
    ymax=np.nanmax(lim[0]) #store ymax
    xlim=(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin)) #set the xlim
    ylim=(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin)) #set the ylim
    
    vmin=np.nanmin(result) #set the min value of normalization
    vmax=np.nanmax(result) #set the max value of normalization
    
    fig=plt.figure(figsize=(5*ncols,7*nrows)) #create the figure
    
    ax=fig.add_subplot(nrows,ncols,1,projection=wcs) #create the subplot
    ax.imshow(field,origin='lower',vmin=vmin,vmax=vmax,aspect='equal',cmap='jet') #plot the original field
    ax.set_xlim(xlim) #set the xlim
    ax.set_ylim(ylim) #set the ylim
    ax.set_xlabel('RA') #set the x-axis label
    ax.set_ylabel('DEC') #set the y-axis label
    ax.set_title('Original field',pad=20,fontsize=20)
    
    ax=fig.add_subplot(nrows,ncols,2,projection=wcs) #create the subplot
    ax.imshow(result,origin='lower',vmin=vmin,vmax=vmax,aspect='equal',cmap='jet') #plot the original field
    ax.set_xlim(xlim) #set the xlim
    ax.set_ylim(ylim) #set the ylim
    ax.set_xlabel('RA') #set the x-axis label
    ax.set_ylabel('DEC') #set the y-axis label
    ax.set_title('Extended Field',pad=20,fontsize=20)
    ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
    
    fig.savefig(velfi_out+plot_format,dpi=300,bbox_inches='tight') #save the figure
    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    else:
        plt.close()
                        
    hdu=fits.PrimaryHDU(result,header=header) #create the primary HDU
    output=fits.HDUList([hdu]) #make the HDU list
    output.writeto(velfi_out+'.fits',overwrite=True) #save the syntethic velocity field
        
################################ --- ANCILLARY FUNCTIONS --- ################################
#############################################################################################
def converttoHI(data,fluxunits='Jy/beam',beamarea=None,pixunits='deg',spectralres=None,specunits='m/s'):
    """Convert an array of flux values into HI column density.

    Args:
        data (str/array): Name or path+name of the fits data, or array of the data.
        fluxunits (str): String indicating the units of flux.
        beamarea (float): Area of the beam in pixel units squared.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        spectralres (float): Spectral resolution of the data in spectral units.
    
    Returns:
        Array of flux values converted into HI column density.
        
    Raises:
        ValueError: If incorrect spatial units are provided.
        ValueError: If incorrect spectral units are provided.
        ValueError: If no spectral resolution and beam area are provided when the data is given as an array.
        ValueError: If no beam information is provided.
    """
    #CHECK THE INPUT#
    if pixunits not in ['deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    if specunits not in ['km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')
    if type(data) is str: #if the data as a path-to-file
        darray,header=__load(data) #open the fits file
    elif type(data) is list: #if the data are a list
        darray=np.array(data) #convert to array
    else: #if the data is given as a numpy array
        if spectralres is None or beamarea is None: #if no additional information are given, abort
            raise ValueError('ERROR: Please provide the spectral resolution and the beam area. Aborting!')
        darray=data #store the data   
     
    #CHECK THE REQUIRED INFORMATION FOR FLUX CALCULATION#
    #------------   BEAM     ------------#
    if beamarea is None: #if the beam area is not given
        if 'BMAJ'.casefold() in header: #check the beam major axis keyword
            bmaj=header['BMAJ'.casefold()] #store the beam major axis
        if 'BMIN'.casefold() in header: #check the beam minor axis keyword
            bmin=header['BMIN'.casefold()] #store the beam minor axis
        else: #if no beam info have been found, abort
            raise ValueError('ERROR: No beam information are found: aborting!')
        beamarea=1.13*bmaj*bmin #calculate the beam area
    #------------   UNITS     ------------#
    if 'm/s'.casefold() in fluxunits or 'Hz'.casefold() in fluxunits: #if the units contains the spectral information
        spectralres=1 #spectralres is dummy variable
    #------------   SPECTRAL RESOLUTION     ------------         
    if spectralres is None: #if no units for the data are provided
        if type(data) is str: #if the data as a path-to-file
            if 'CDELT3'.casefold() in header: #check the spectral resolution keyword
                spectralres=header['CDELT3'.casefold()] #store the spectral resolution
            else:
                raise ValueError('ERROR: No spectral resolution is found: aborting!')
        else:
            raise ValueError('ERROR: No spectral resolution is found: aborting!')    
    if spectralres < 0: #if the spectral resolution is negative
        spectralres=-spectralres #convert to positive
    #------------   CONVERT THE VALUES IN STANDARD UNITS   ------------ 
    if pixunits == 'deg': #if the spatial units are deg
        beamarea=beamarea*3600*3600 #convert the beam area in arcsec^2
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        beamarea=beamarea*60*60 #convert the beam area in arcsec^2
    if specunits == 'm/s' and spectralres != 1: #if the spectralunits are m/s
        spectralres/=1000 #convert the spectral resolution in km/s
    HI=darray*spectralres*(1.25*10**24)/beamarea #convert to HI column density
    return HI

#############################################################################################
def cosmo(z,H0,omega_matter,omega_vacuum,verbose=True):
    """Given a cosmological model and the redshift of an object, this function calculates various cosmological quantities such as distances, age of the Universe, and light travel time for that object.

    Args:
        z (float): Redshift of the object.
        H0 (float): Hubble Parameter in km/s*Mpc.
        omega_matter (float): Omega for matter.
        omega_vacuum (float): Omega for vacuum.
        verbose (bool): If True, messages will be printed to the terminal.
    
    Returns:
        dict: A Python dictionary containing the cosmological quantities of the given object.
        
    Raises:
        None
    """
    c=299792.458 #velocity of light in km/sec
    Tyr = 977.8 #coefficent for converting 1/H into Gyr 
    h=H0/100. #little h
    omega_rad=4.165E-5/(h*h)   #includes 3 massless neutrino species, T0 = 2.72528
    WK=1-omega_matter-omega_rad-omega_vacuum #Omega curvaturve = 1-Omega_total
    az=1.0/(1+1.0*z) #1/(1+z(object))
    n=1000 #number of points in integrals
    age=0 #age of Universe in units of 1/H0
    DTT=0 #value of DTT in Gyr
    DCMR=0 #comoving radial distance in units of c/H0
    for i in range(n):
        a=az*(i+0.5)/n #1/(1+z), the scale factor of the Universe
        adot=np.sqrt(WK+(omega_matter/a)+(omega_rad/(a*a))+(omega_vacuum*a*a))
        age=age+1./adot
    zage=az*age/n #age of Universe at redshift z in units of 1/H0
    zage_Gyr=(Tyr/H0)*zage #value of zage in Gyr

    #do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a=az+(1-az)*(i+0.5)/n
        adot=np.sqrt(WK+(omega_matter/a)+(omega_rad/(a*a))+(omega_vacuum*a*a))
        DTT=DTT+1./adot 
        DCMR=DCMR+1./(a*adot) 

    DTT=(1.-az)*DTT/n
    DCMR=(1.-az)*DCMR/n
    age=DTT+zage
    age_Gyr=age*(Tyr/H0) #value of age in Gyr
    DTT_Gyr= (Tyr/H0)*DTT #value of DTT in Gyr
    DCMR_Gyr=(Tyr/H0)*DCMR
    DCMR_Mpc=(c/H0)*DCMR

    #tangential comoving distance
    x=np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio=0.5*(exp(x)-exp(-x))/x 
        else:
            ratio=sin(x)/x
    else:
        y=x*x
    if WK < 0:
        y=-y
    ratio=1.+y/6.+y*y/120.
    DCMT=ratio*DCMR
    DA=az*DCMT #angular size distance
    DA_Mpc=(c/H0)*DA
    kpc_DA=DA_Mpc/206.264806
    DA_Gyr=(Tyr/H0)*DA
    DL=DA/(az*az) #luminosity distance
    DL_Mpc=(c/H0)*DL
    DL_Gyr=(Tyr/H0)*DL #DL in units of billions of light years

    #comoving volume computation
    x=np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio=(0.125*(exp(2.*x)-exp(-2.*x))-x/2.)/(x*x*x/3.)
        else:
            ratio=(x/2.-sin(2.*x)/4.)/(x*x*x/3.)
    else:
        y=x*x
        if WK < 0:
            y=-y
        ratio=1.+y/5.+(2./105.)*y*y
    VCM=ratio*DCMR*DCMR*DCMR/3.
    V_Gpc=4.*np.pi*((0.001*c/H0)**3)*VCM

    if verbose:
        print(f'For H₀={H0:.1f}, \u03A9ₘ={omega_matter:.1f}, \u03A9ᵥ={omega_vacuum:.1f}, z={z}:')
        print(f'It is now {age_Gyr:.1f} Gyr since the Big Bang.')
        print(f'The age at redshift z was {zage_Gyr:.1f} Gyr.')
        print(f'The light travel time was {DTT_Gyr:.3f} Gyr.')
        print(f'The comoving radial distance, which goes into Hubbles law, is {DCMR_Mpc:.1f} Mpc or {DCMR_Gyr:.3f} Gly')
        print(f'The comoving volume within redshift z is {V_Gpc:.3f} Gpc^3.')
        print(f'The angular size distance Dₐ is {DA_Mpc:.1f} Mpc or {DA_Gyr:.3f} Gly.')
        print(f'This gives a scale of {kpc_DA:.3f} kpc/arcsec.')
        print(f'The luminosity distance Dₗ is {DL_Mpc:.1f} Mpc or {DL_Gyr:.3f} Gly.')
        print(f'The distance modulus, m-M, is {5*np.log10(DL_Mpc*10.**6)-5:.1f}')

    cosmology={} #initialize the cosmology dictionary
    cosmology['age']=age_Gyr
    cosmology['age at z']=zage_Gyr
    cosmology['light travel time']=DTT_Gyr
    cosmology['radial distance [Mpc]']=DCMR_Mpc
    cosmology['radial distance [Gly]']=DCMR_Gyr
    cosmology['comoving volume']=V_Gpc
    cosmology['angular distance [Mpc]']=DA_Mpc
    cosmology['angular distance [Gyr]']=DA_Gyr
    cosmology['angular scale']=kpc_DA
    cosmology['distance [Mpc]']=DL_Mpc
    cosmology['distance [Gyr]']=DL_Gyr
    
    return cosmology
    
#############################################################################################
def getHImass(data,fluxunits='Jy/beam*m/s',beamarea=None,pixelres=None,pixunits='deg',pbcorr=False,distance=None,box=None,**kwargs):
    """Convert an nD-array of flux densities into HI mass.

    Args:
        data (str or array): The name or path+name of the FITS data file, or an array of the data.
        fluxunits (str): String indicating the flux units.
        beamarea (float): Area of the beam in pixel units squared.
        pixelres (float): Pixel resolution of the data in pixel units.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        pbcorr (bool): Apply the primary beam correction if True. Note that in this case, you must supply a beam cube.
        distance (float): Distance to the object in megaparsecs (Mpc).
        box (array-like): Region to compute the flux as [xmin, xmax, ymin, ymax].
    
    Kwargs:
        beamcube (str or ndarray): The name or path+name of the FITS beam cube if pbcorr is True.
        path (str): The path to the beam cube if the beamcube is given as a name and not as a path+name.
        verbose (bool): Option to print messages to the terminal if True.   
    
    Returns:
        tuple: A tuple containing the HI mass over the given array as a float and the corresponding uncertainty.
    
    Raises:
        ValueError: If incorrect spatial units are provided.
        ValueError: If no spatial resolution and beam area are given when the data type is an array.
        ValueError: If data is not provided as a path-to-FITS file or as a NumPy ndarray.
        ValueError: If no beam information is found.
        ValueError: If no pixel resolution is found.
        ValueError: If no data units are found.
        ValueError: If data units are not in km/s or m/s.
        ValueError: If the distance is set.
        ValueError: If the size of the spatial box is not 4.
    """
    #CHECK THE INPUT#
    verbose=kwargs.get('verbose',False) #store the verbosity
    if pixunits not in ['deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    if type(data) is str: #if the data as a path-to-file
        with fits.open(data) as Data: #open the fits file
            header=Data[0].header #store the header
            darray=Data[0].data #store the data
        del Data[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
    elif type(data) is np.ndarray: #if the data is given as a numpy array
        if beamarea is None or pixelres is None: #if no additional information are given, abort
            raise ValueError('ERROR: Please provide the spatial resolution and the beam area. Aborting!')
        darray=data #store the data
    else:
        raise ValueError('ERROR: Please provide the data as path-to-fits file or as numpy.ndarray. Aborting!')
    if pbcorr: #if the primary beam correction is applied
        beamcube=kwargs.get('beamcube',None) #store the data cube path from the input parameters
        if beamcube is None: #if the beam cube is not provided
            if verbose:
                warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')
            pbcorr=False #set the pbcorr option to False
        elif type(beamcube)==str: #if the beam cube is a string
            beamcube=__read_string(beamcube,'beamcube',**kwargs) #store the path to the beam cube
    darray[darray<0]=0 #remove the negatives if any (like noise artifacts sneaked in the moment map    
    
    #CHECK THE REQUIRED INFORMATION FOR FLUX CALCULATION#
    #------------   BEAM     ------------#
    if beamarea is None: #if the beamarea is not given
        if 'BMAJ'.casefold() in header: #check the beam major axis keyword
            bmaj=header['BMAJ'.casefold()] #store the beam major axis
        if 'BMIN'.casefold() in header: #check the beam minor axis keyword
            bmin=header['BMIN'.casefold()] #store the beam minor axis
        else: #if no beam info have been found, abort
            raise ValueError('ERROR: No beam information are found: aborting!')
        beamarea=1.13*bmaj*bmin #calculate the beam area
    #------------   PIXEL RESOLUTION     ------------#
    if pixelres is None: #if no pixel resolution is provided
        if type(data) is str: #if the data as a path-to-file
            if 'CDELT1'.casefold() in header: #check the pixel resolution keyword
                pixelres=header['CDELT1'.casefold()]**2 #store the pixel resolution and convert it into pixelarea
            else:
                raise ValueError('ERROR: No pixel resolution is found: aborting!')
        else:
            raise ValueError('ERROR: No pixel resolution is found: aborting!') 
    pixelsize=pixelres**2 #convert to pixel size
    #------------   UNITS     ------------#
    if fluxunits is None: #if no flux units for the data are provided
        if type(data) is str: #if the data as a path-to-file
            if 'BUNIT'.casefold() in header: #check the data fluxu nits keyword
                fluxunits=header['BUNIT'.casefold()] #store the data units
            else:
                raise ValueError('ERROR: No flux units are found: aborting!')
        else:
            raise ValueError('ERROR: No flux units are found: aborting!')
    if 'km/s'.casefold() in fluxunits: #if the flux units have km/s
        pass #do nothing
    elif 'm/s'.casefold() in fluxunits: #if the flux units have m/s
        darray=darray/1000 #convert into km/s
    else:
        raise ValueError(f"ERROR: Wrong flux units provided: {fluxunits}. They must contain 'm/s' or 'km/s'. Aborting!")
    #------------   PB CORRECTION     ------------# 
    if pbcorr: #if the primary beam correction is applied
        with fits.open(beamcube) as pb_cube: #open the primary beam cube
            pb_slice=pb_cube[0].data[np.array(pb_cube[0].shape[0]/2).astype(int)] #extract the central plane
        del pb_cube[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
        darray/=pb_slice #apply the primary beam correction
        
    #------------   DISTANCE     ------------#
    if distance is None: #if no pixel resolution is provided
        raise ValueError('ERROR: No distance provided: aborting!')
    #------------   CONVERT THE VALUES IN STANDARD UNITS   ------------ 
    if pixunits == 'deg': #if the spatial units are deg
        beamarea=beamarea*3600*3600 #convert the beam area in arcsec^2
        pixelsize=pixelsize*3600*3600 #convert the pixel size in arcsec^2
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        beamarea=beamarea*60*60 #convert the beam area in arcsec^2
        pixelsize=pixelsize*60*60 #convert the pixel size in arcsec^2
    #------------   BOX     ------------         
    xmin,xmax,ymin,ymax=__load_box(darray,box) #load the spatial box
    
    #CALCULATE THE MASS# 
    HI_mass=(2.35*10**5)*(distance**2)*pixelsize*np.nansum(darray[ymin:ymax,xmin:xmax])/beamarea #compute the HI mass
    error=HI_mass/10 #the error on the mass is equal to the calibration error (typically of 10%)
    if verbose: #if the print-to-terminal option is True
        print(f'Total HI mass: {HI_mass:.1e} solar masses')
    return HI_mass,error #return the mass and its error

#############################################################################################
def flux(data,fluxunits='Jy/beam',beamarea=None,pixelres=None,pixunits='deg',spectralres=None,specunits='m/s',box=None,verbose=False):
    """Calculate the flux over a given region.

    Args:
        data (str/array): The input data. It can be either the name or path+name of the FITS data file, or an array of the data.
        fluxunits (str): The units of flux.
        beamarea (float): The area of the beam in square pixunits.
        pixelres (float): The pixel resolution of the data in pixunits.
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        spectralres (float): The spectral resolution of the data in specunits.
        box (array-like): The region to compute the flux as [xmin, xmax, ymin, ymax].
        verbose (bool): If True, print messages to the terminal.
    
    Returns:
        The flux over the specified array.
    
    Raises:
        ValueError: If incorrect spatial units are provided.
        ValueError: If incorrect spectral units are provided.
        ValueError: If no spatial resolution, spectral resolution, and beam area are given when the data type is an array.
        ValueError: If the data is not provided as a path-to-fits file or as a numpy.ndarray.
        ValueError: If no beam information is found.
        ValueError: If no pixel resolution is found.
        ValueError: If no data units are found.
        ValueError: If no spectral resolution is found.
        ValueError: If the size of the spatial box is not 4.
    """
    #CHECK THE INPUT#
    if pixunits not in ['deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    if specunits not in ['km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')
    if type(data) is str: #if the data as a path-to-file
        with fits.open(data) as Data: #open the fits file
            header=Data[0].header #store the header
            darray=Data[0].data #store the data
        del Data[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
    elif type(data) is np.ndarray: #if the data is given as a numpy array
        if spectralres is None or beamarea is None or pixelres is None: #if no additional information are given, abort
            raise ValueError('ERROR: Please provide the spatial and spectral resolution, and the beam area. Aborting!')
        darray=data #store the data
    else:
        raise ValueError('ERROR: Please provide the data as path-to-fits file or as numpy.ndarray. Aborting!')
    #------------   REMOVE THE NEGATIVE VALUES     ------------# 
    darray[darray<0]=0 #set to 0 the negative flux values    
        
    #CHECK THE REQUIRED INFORMATION FOR FLUX CALCULATION#
    #------------   BEAM     ------------#
    if beamarea is None: #if the beamarea is not given
        if 'BMAJ'.casefold() in header: #check the beam major axis keyword
            bmaj=header['BMAJ'.casefold()] #store the beam major axis
        if 'BMIN'.casefold() in header: #check the beam minor axis keyword
            bmin=header['BMIN'.casefold()] #store the beam minor axis
        else: #if no beam info have been found, abort
            raise ValueError('ERROR: No beam information are found: aborting!')
        beamarea=1.13*bmaj*bmin #calculate the beam area
    #------------   PIXEL RESOLUTION     ------------#
    if pixelres is None: #if no pixel resolution is provided
        if type(data) is str: #if the data as a path-to-file
            if 'CDELT1'.casefold() in header: #check the pixel resolution keyword
                pixelres=header['CDELT1'.casefold()]**2 #store the pixel resolution and convert it into pixelarea
            else:
                raise ValueError('ERROR: No pixel resolution is found: aborting!')
        else:
            raise ValueError('ERROR: No pixel resolution is found: aborting!')
    pixelsize=pixelres**2 #convert to pixel size
    #------------   UNITS     ------------#
    if fluxunits is None: #if no units for the data are provided
        if type(data) is str: #if the data as a path-to-file
            if 'BUNIT'.casefold() in header: #check the data units keyword
                fluxunits=header['BUNIT'.casefold()] #store the data units
            else:
                raise ValueError('ERROR: No data units are found: aborting!')
        else:
            raise ValueError('ERROR: No data units are found: aborting!')
    if 'm/s'.casefold() in fluxunits or 'Hz'.casefold() in fluxunits: #if the units contains the spectral information
        spectralres=1 #spectralres is dummy variable
    #------------   SPECTRAL RESOLUTION     ------------         
    if spectralres is None: #if no units for the data are provided
        if type(data) is str: #if the data as a path-to-file
            if 'CDELT3'.casefold() in header: #check the spectral resolution keyword
                spectralres=header['CDELT3'.casefold()] #store the spectral resolution
            else:
                raise ValueError('ERROR: No spectral resolution is found: aborting!')
        else:
            raise ValueError('ERROR: No spectral resolution is found: aborting!') 
    #------------   CONVERT THE VALUES IN STANDARD UNITS   ------------ 
    if pixunits == 'deg': #if the spatial units are deg
        beamarea=beamarea*3600*3600 #convert the beam area in arcsec^2
        pixelsize=pixelsize*3600*3600 #convert the pixel size in arcsec^2
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        beamarea=beamarea*60*60 #convert the beam area in arcsec^2
        pixelsize=pixelsize*60*60 #convert the pixel size in arcsec^2
    if specunits == 'm/s' and spectralres != 1: #if the spectralunits are m/s
        spectralres/=1000 #convert the spectral resolution in km/s
    #------------   BOX     ------------         
    xmin,xmax,ymin,ymax=__load_box(darray,box) #load the spatial box
    
    #CALCULATE THE FLUX# 
    flux=np.nansum(darray[ymin:ymax,xmin:xmax])*spectralres*pixelsize/beamarea #calculate the flux
    
    if verbose: #if the print-to-terminal option is True
        print(f'The flux is {flux:.1e} {fluxunits}')
    return flux #return the flux
            
#############################################################################################
def create_config(name='default_parameters'):
    """Create a default configuration file.

    Args:
        name (str): name of the configuration file.
        
    Returns:
        Default configuration file saved as 'name'.par.
        
    Raises:
        None.
    """
    with open(name+'.par',mode='w') as configfile: #equivalent of saying configfile=open()
        configfile.write(";configuration file for nicci. Use ';' or '#' to comment the parameters you don't need/know\n"
            '\n'
            '[GENERAL]\n'
            'verbose     =      #if True, all plots and ancillary information will be printed to terminal [True,False] (default: False)\n'
            '\n'
            '[INPUT]\n'
            'path        =      #path to the working directory (generally where the data are stored). (defatult: current directory)\n'
            '\n'
            '[COSMOLOGY]\n'
            'HO          =      #Hubble parameter (default: 69.6)\n'
            'Omegam      =      #Omega matter (default: 0.286)\n'
            'Omegav      =      #Omega vacuum (default: 0.714)\n'
            '\n'
            '[GALAXY]\n'
            'objname     =      #name of the object (default: None)\n'
            'distance    =      #distance of the object in Mpc (default: None)\n'
            'redshift    =      #redshift of the object (default: None)\n'
            'asectokpc   =      #arcsec to kpc conversion (default: None)\n'
            'vsys        =      #systemic velocity of the object in km/s (default: None)\n'
            'pa          =      #position angle of the object in deg (default: None)\n'
            'inc         =      #inclination angle of the object in deg (default: None)\n'
            '\n'
            '[CUBEPAR]\n'
            'pixunits    =      #spatial units (default: None)\n'
            'specunits   =      #spectral units (default: None)\n'
            'fluxunits   =      #flux units (default: None)\n'
            'pixelres    =      #pixel resolution in pixunits (default: None)\n'
            'spectralres =      #spectral resolution in specunits (default: None)\n'
            'rms         =      #root-mean-square value in fluxunits (default: None)\n'
            '\n'
            '[BEAM]\n'
            'bmaj        =      #beam major axis in pixunits (default: None)\n'
            'bmin        =      #beam minor axis in pixunits (default: None)\n'
            'bpa         =      #beam position angle in degrees (default: None)\n'
            '\n'
            '[CORRECTION]\n'
            'pbcorr      =      #apply the primary beam correction [True,False] (default: False)\n'
            '\n'
            '[FITS]\n'
            'datacube    =      #name of the fits file of the data cube including .fits (default: None)\n'
            'beamcube    =      #name of the fits file of the beam cube including .fits (default: None)\n'
            'maskcube    =      #name of the fits file of the mask cube including .fits (default: None)\n'
            'mask2d      =      #name of the fits file of the 2D mask including .fits (default: None)\n'
            'channelmap  =      #name of the fits file of the channel map including .fits (default: None)\n'
            'modelcube   =      #name of the fits file of the model cube including .fits (default: None)\n'
            'mom0map     =      #name of the fits file of the moment 0 map including .fits (default: None)\n'
            'mom1map     =      #name of the fits file of the moment 1 map including .fits (default: None)\n'
            'mom2map     =      #name of the fits file of the moment 2 map including .fits (default: None)\n'
            'vfield      =      #name of the fits file of the velocity field including .fits (default: None)\n'
            '\n'
            '[CHANMAP]\n'
            'chanmin     =      #starting channel to plot in the channel map (default: 1)\n'
            'chanmax     =      #ending channel to plot in the channel map (default: None)\n'
            'chansep     =      #channel separation to plot in the channel map (chamin,chanmin+chansep,chanmin+2*chansep,...,chanmax) (default: 1)\n'
            'box         =      #comma-separated pixel edges of the box to extract the channel map in the format [xmin,xmax,ymin,ymax] (default: None)\n'
            'nsigma      =      #rms threshold to plot the contours (lowest contours will be nsigma*rms) (default: 3)\n'
            'usemask     =      #use a mask in the channel map [True,False] (default: False)\n'
            'output      =      #output directory/name to save the plot. Cannot be empty or None (default: None)\n'
            '\n'
            '[CUBEDO]\n'
            'datacube    =      #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'operation   =      #operation to perform on the cube [blank,clip,crop,cut,extend,mirror,mom0,shuffle,toint] (default: None)\n'
            "chanmin     =      #first channel for the operations 'blank,cut,mom0' (default: 1)\n"
            "chanmax     =      #last channel for the operations 'blank,cut,mom0' (default: None)\n"
            "box         =      #comma-separated pixel edges of the box to extract for operation 'cut' in the format [xmin,xmax,ymin,ymax] (default: None)\n"
            "addchan     =      #number of channels to add in operation 'extend'. Negative values add lower channels, positive add higher channels (default: None)\n"
            "value       =      #value to assign to blank pixel in operation 'blank' (blank is np.nan) (default: blank)\n"
            "usemask     =      #use a 2D mask in the operation 'clip' [True,False] (default: False)\n"
            'mask        =      #name of the fits file of the 2D mask including .fits. If empty, is the same of [FITS] mask2d (default: None)\n'
            "cliplevel   =      #clip threshold as % of the peak (0.5 is 50%) for operation 'clip' (default: 0.5)\n"
            "center      =      #the rotation center as [x0,y0,z0] around which the cube is mirrored when operation is 'mirror' (default: None)\n"
            'output      =      #output directory/name to save the new cube. Cannot be empty or None (default: None)\n'
            '\n'
            '[CUBESTAT]\n'
            'nsigma      =      #rms threshold in terms of nsigma*rms for detection limit (default: 3)\n'
            "statistics  =      #statistic to be used in the noise measurement. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to emission and artefacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artefacts (default: 'mad')\n"
            "fluxrange   =      #flux range to be used in the noise measurement. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artefacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux (default: 'negative')\n"
            '\n'
            '[FITSARITH]\n'
            'fits1       =      #name of reference fits file including .fits (default: None)\n'
            'fits2       =      #name of second fits file including .fits (default: None)\n'
            'operation   =      #operation to do between the two fits [sum,sub,mul,div] (default:None)\n'
            'output      =      #output directory/name to save the new fits. Cannot be empty or None (default: None)\n'
            '\n'
            '[FIXMASK]\n'
            'datacube    =      #name of the fits file of the reference data cube including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'maskcube    =      #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube (default: None)\n'
            'output      =      #output directory/name to save the new cube. Cannot be empty or None (default: None)\n'
            '\n'
            '[GAUSSFIT]\n'
            'datacube    =      #name of the fits file of the data cube to fit including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'mask        =      #name of the fits file of the 2D mask including .fits. If empty, is the same of [FITS] mask2d (default: None)\n'
            'components  =      #number of components (1 or 2) to fit (default: 1)'
            'linefwhm    =      #first guess on the fwhm of the line profile in km/s (default: 15)\n'
            'amp_thresh  =      #amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum (default: 0)\n'
            'p_reject    =      #p-value threshold for fit rejection. If a best-fit as p>p_reject, it will be rejected (default: 1) \n'
            'clipping    =      #clip the spectrum to a % of the profile peak [True,False] (default: False)\n'
            'threshold   =      #clip threshold as % of the peak (0.5 is 50%) if clipping is True (default: 0.5)\n'
            'errors      =      #compute the errors on the best-fit [True,False] (default: False)\n'
            'write_field =      #compute the best-fit velocity field [True,False] (default: False)\n'
            'output      =      #output directory/name to save the new cube. Cannot be empty or None (default: None)\n'
            '\n'
            '[GETPV]\n'
            'datacube    =      #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'width       =      #width of the slice in arcsec. If not given, will be the beam size (default: None)\n'
            'points      =      #RA-DEC comma-separated coordinate of the slice points in decimal degree. If two are given ([x,y]), it is assumed they are the center of the slice. Else, they need to be the starting and ending coordinates ([xmin,xmax,ymin,ymax]) (default: None)\n'
            'angle       =      #position angle of the slice in degree. If not given, will be the object position angle (default: None)\n'
            'chanmin     =      #starting channel of the slice (default: 0)\n'
            'chanmax     =      #ending channel of the slice (default: None)\n'
            "statistics  =      #statistic to be used in the noise measurement. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to emission and artefacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artefacts (default: 'mad')\n"
            "fluxrange   =      #flux range to be used in the noise measurement. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artefacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux (default: 'negative')\n"
            'nsigma      =      #rms threshold to plot the contours (lowest contours will be nsigma*rms) (default: 3)\n'
            'output      =      #output directory/name to save the result. Cannot be empty or None (default: None)\n'
            '\n'
            '[LINESFINDER]\n'
            'smooth_ker  =      #boxcar kernel size (or list of kernel sizes) to apply. The individual kernel sizes must be odd integer values of 3 or greater and denote the full width of the Boxcar filter used to smooth the spectrum. Set to None or 1 to disable (default: None)\n'
            'threshold   =      #flux threshold to be used by the source finder relative to the measured rms in each smoothing iteration. Values in the range of about 3 to 5 have proven to be useful in most situations, with lower values in that range requiring use of the reliability filter to reduce the number of false detections (default: 3)\n'
            'replace     =      #before smoothing the spectrum during n source finder iteration, every flux value that was already detected in a previous iteration will be replaced by this value multiplied by the original noise level in the non-smoothed data cube, while keeping the original sign of the data value. This feature can be disabled by specifying a value of < 0. (default: 2)\n'
            "statistics  =      #statistic to be used in the noise measurement process of the source finder. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to emission and artefacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artefacts (default: 'mad')\n"
            "fluxrange   =      #flux range to be used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artefacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux (default: 'negative')\n"
            'link_ker    =      #minimum size of sources in channels. Sources that fall below this limit will be discarded by the linker (default: 3)\n'
            'min_size    =      #minimum number of channels a source can cover (default: 3)\n'
            'rel_thresh  =      #reliability threshold in the range of 0 to 1. Sources with a reliability below this threshold will be discarded (default: 0)\n'
            'rel_kernel  =      #scaling factor for the size of the Gaussian kernel used when estimating the density of positive and negative detections in the reliability parameter space (default: 0.4)\n'
            'rel_snrmin  =      #lower signal-to-noise limit for reliable sources. Detections that fall below this threshold will be classified as unreliable and discarded. The value denotes the integrated signal-to-noise ratio of the source. Note that the spectral resolution is assumed to be equal to the channel width (default: 3)\n'
            'negative    =      #if set to False, then the detections with negative flux will be discarded at the end of the process (default: True)\n'
            'gauss_tests =      #if set to True, two statistical tests (Anderson-Darling and Kanekar) will be performed to quantify the gaussianity of the noise in the input spectra (default: False)\n'
            'p-value     =      #p-value threshold of the Anderson-Darling test to reject non-Gaussian spectra (a spectrum has Gaussian noise if p>p_value) (default: 0.05)\n'  
            'kanekar     =      #two comma-separated floats representing the minimum and maximum value of the Kanekar test to reject non-Gaussian spectra (a spectrum has Gaussian noise if kanekar_min<test_value<kanekar_max) (default: None)\n'
            'catalogue   =      #if True, a csv file containing the sources catalogue will be written on the disk (default: False)\n'
            'output      =      #output directory/name to save the result. Cannot be empty or None (default: None)\n'
            '\n'
            '[NOISE VARIATIONS]\n'
            "statistics  =      #statistic to be used in the noise measurement. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to emission and artefacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artefacts (default: 'mad')\n"
            "fluxrange   =      #flux range to be used in the noise measurement. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artefacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux (default: 'negative')\n"
            'output      =      #output directory/name to save the new cube. Cannot be empty or None (default: None)\n'
            '\n'
            '[PLOTMOM]\n'
            'output      =      #output directory/name to save the plot. Cannot be empty or None (default: None)\n'
            '\n'
            '[REMOVEMOD]\n'
            'method      =      #method to remove the model [all,blanking,b+s,negblank,subtraction] (default: subtraction)\n'
            "threshold   =      #flux threshold for the 'all,blanking,b+s' methods in units of cube flux (default: 0)\n"
            'output      =      #output directory/name to save the new cube. Cannot be empty or None (default: None)\n'
            '\n'
            '[ROTCURVE]\n'
            'center      =      #x-y comma-separated coordinates of the rotational center in pixel (default: None)\n'
            'save_csv    =      #store the output in a csv file [True,False] (default: False)\n'
            'output      =      #output directory/name to save the result. Cannot be empty or None (default: None)\n'
            '\n'
            '[STACKING]\n'
            'center      =      #x-y comma-separated coordinates of the rotational center in pixel (default: None)\n'
            'nregions    =      #number of regions from which the spectra are extracted and stacked (default: None)\n'
            "shape       =      #shape of regions to stack between 'cells', cones' and 'concentric' (default: 'cones')\n"
            'between     =      #comma-separated min and max angle from which the stacking regions are defined. Set to None to disable (default: None)\n'
            'radii       =      #comma-separated min and max radius from the galactic center in pixunits from which the spectra are extracted and stacked (default: None)\n'
            "weighting   =      #type of weight to apply during the stacking between 'None' and 'rms' (default None)\n"
            "statistics  =      #statistic to be used in the noise measurement. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to emission and artefacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artefacts (default: 'mad')\n"
            "fluxrange   =      #flux range to be used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artefacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux (default: 'negative')\n"
            'output      =      #output directory/name to save the result. Cannot be empty or None (default: None)\n'
            '\n'
            '[VELFI]\n'
            'radii       =      #radii in pixunits at which the rotation velocity is measured (default: None)\n'
            'vrot        =      #rotation velocities (subtracted from the systemic velocities) in specunits for the radii (default: None)\n'
            'vrad        =      #expansion velocity for the radii (default: None)\n'
            'center      =      #comma-separated value for the x and y coordinates of the rotation center in pixel (default: None)\n'
            'extend_only =      #extend a given velocity field (True) or write a new one from scratch (False) (default: False)\n'
            'correct     =      #correct the input rotation velocities for the inclination angles (True) or not (False) (default: False)\n'
            'output      =      #output directory/name to save the new cube. Cannot be empty or None (default: None)\n'
            '\n'
            '[PLOTSTYLE]\n'
            'ctr_width   =      #width of the contours levels (default: 2)\n'
            'left_space  =      #width of the left margin in axis units (from 0 to 1) (default: 0.05)\n'
            'upper_space =      #width of the upper margin in axis units (from 0 to 1) (Default: 0.95)\n'
            'lower_space =      #width of the lower margin in axis units (from 0 to 1) (Default: 0.05)\n'
            'format      =      #file format for the plots (default: pdf)')

################################## --- PRIVATE FUNCTIONS --- #################################
#############################################################################################
def __assign_to_cells(xshape,yshape,nregions):
    """Given the length of a cell, divide the spatial dimension of a datacube in a grid of N cells and assign each pixel to a given cell.

    Args:
        xshape (int): shape of the data along the x-axis
        yshape (int): shape of the data along the y-axis
        nregions (int): number of cells
        
    Returns:
        lists of the x and y coordinate of the pixel in each cell
        
    Raises:
        None
    """
    #GET THE NUMBER OF CELLS PER DIMENSION#
    xstart=np.linspace(0,xshape,round(np.sqrt(nregions))+1) #starting x-coordinate of each cell
    ystart=np.linspace(0,yshape,round(np.sqrt(nregions))+1) #starting y-coordinate of each cell
     
    #DO THE ASSIGNMENT#
    xvalid=[] #initialize the x-coordinates list of the pixel in the cells
    yvalid=[] #initialize the y-coordinates list of the pixel in the cells
    with tqdm(desc='Assigned cells') as pbar:
        for i in range(len(xstart)-1):
            for j in range(len(ystart)-1):
                x,y=np.meshgrid(np.arange(xstart[i],xstart[i+1]),np.arange(ystart[j],ystart[j+1])) #create the coordinate grid of the first cell
                xvalid.append(x.flatten().astype(int)) #append the x-coordinates
                yvalid.append(y.flatten().astype(int)) #append the y-coordinates
                pbar.update(1)

    return xvalid,yvalid
    
#############################################################################################
def __assign_to_concentric(data,nregions,pa,inc,rmin,rmax,angles,x0,y0):
    """Given the parameters of the cones (number of coneas, inner radius, outer radius, x-center, y-center), the pixel resolution and the position angle and inclination of the galaxy, assign each pixel of the data within rmin and rmax into the correct cone.

    Args:
        data (ndarray): 3D array with the data
        nregions (int): number of cones
        pa (float): object position angle in degree
        inc (float): inclination of the object in degrees (0 deg means face-on)
        rmin (float): inner radius of the cone
        rmax (float): outer radius of the cone
        x0 (float): x-coordinate of the center
        y0 (float): y-coordinate of the center
        
    Returns:
        lists of the x and y coordinate of the pixel in each cone
        
    Raises:
        None
    """
    #SETUP THE EXTRACTION REGIONS PROPERTIES#
    if angles is not None: #if to compute the circles between some angles
        angles=np.radians(np.array(angles)) #convert the angles into radians
    pa=np.radians(pa-180) #convert the position angle into radians
    inc=np.radians(inc) #convert the inclination angle into radians
    xmin=round(x0-rmax) #store the min x-coordinate
    if xmin<0: #if xmin is less than 0
        xmin=0 #set to 0
    xmax=round(x0+rmax) #store the max x-coordinate
    if xmax>=data.shape[2]: #if xmax is greater than the x-shape of the data
        xmax=data.shape[2]-1 #set to the x-shape of the data
    ymin=round(y0-rmax) #store the min y-coordinate
    if ymin<0: #if ymin is less than 0
        ymin=0 #set to 0
    ymax=round(y0+rmax) #store the max y-coordinate
    if ymax>=data.shape[1]: #if xmax is greater than the y-shape of the data
        ymax=data.shape[1]-1 #set to the y-shape of the data   
    radii=np.linspace(rmin,rmax,nregions+1) #create the radii array of the concentric regions
    
    xvalid=[] #initialize the x-coordinates list of the pixel in the cones
    yvalid=[] #initialize the y-coordinates list of the pixel in the cones
    for i in range(nregions): #append an empty list to create a list of nregions list. We will place in the i-list the pixel belonging to the i-region
        xvalid.append([])
        yvalid.append([])
    yrange=np.arange(ymin,ymax+1) #define the y-range
    xrange=np.arange(xmin,xmax+1) #define the x-range
    with tqdm(desc='Assigned pixel') as pbar:
        for j in yrange: #run over the y-range
            for k in xrange: #run over the x-range
                x=k-x0 #x-position of the pixel w.r.t. the center
                y=j-y0 #y-position of the pixel w.r.t. the center
                if not np.all(np.isnan(data[:,j,k])): #check if the whole spectrum is nan
                    #We have to check first that the pixel is in between the selected angles
                    if angles is not None: #if to compute the circles between some angles
                        valid=False #initialize the valid boolean for the check. We run over each cone and stop when the correct cone is found
                        if (angles[0] >= 0 and angles[1] >= 0) or (angles[0] <= 0 and angles[1] >= 0): #if both angles are positive or angle[0] is negative and angle[1] is positive
                            if angles[0] <= np.arctan2(y,x) <= angles[1]: #the point angle must be in between angle[0] and angle[1]
                                valid=True
                        elif angles[0] >= 0 and angles[1] <= 0: #if angle[0] is positive and angle[1] is negative
                            if angles[0] <= np.arctan2(y,x) or np.arctan2(y,x) <= angles[1]: #the point angle must be higher than angle[0] but lower than angle[1]
                                valid=True
                        elif angles[0] <= 0 and angles[1] <= 0: #if both angles are negative
                            if angles[1] >= np.arctan2(y,x) >= angles[0]: #the point angle must be in between angle[1] and angle[0]
                                valid=True
                    else:
                        valid=True
                        #Now we have to assign the pixel to the correct concentric region
                    if valid: #this check is needed if angles is not None
                        d1=__radius(rmin,x,y,pa,inc) #calculate the elliptical radius of the pixel
                        d2=__radius(rmax,x,y,pa,inc) #calculate the elliptical radius of the pixel and compare the distance w.r.t. to the max radius
                        if d1*d2 <= 0: #if one of the distance is negative, means that the points resides between the innermost and outermost radius, hence, it is a valid pixel for the stacking. Indeed, if a pixel is inside rmin, then d1<0 but also d2<0 and so d1*d2>0. If the pixel is outside rmax, then d1>0 but also d2>0 and so d1*d2>0.
                            #r=np.sqrt(x**2+y**2) #calculate the distance of the pixel from the center
                            r=-__radius(0,x,y,pa,inc)
                            for i in range(nregions+1): #run over the regions
                                if r<radii[i]: #if the distance is less than the radii i
                                    xvalid[i-1].append(k) #append the x-coordinate
                                    yvalid[i-1].append(j) #append the y-coordinate
                                    pbar.update(1)
                                    break
                                
    return xvalid,yvalid
          
#############################################################################################
def __assign_to_cones(data,nregions,pa,inc,rmin,rmax,x0,y0):
    """Given the parameters of the cones (number of coneas, inner radius, outer radius, x-center, y-center), the pixel resolution and the position angle and inclination of the galaxy, assign each pixel of the data within rmin and rmax into the correct cone.

    Args:
        data (ndarray): 3D array with the data
        nregions (int): number of cones
        pa (float): object position angle in degree
        inc (float): inclination of the object in degrees (0 deg means face-on)
        rmin (float): inner radius of the cone
        rmax (float): outer radius of the cone
        x0 (float): x-coordinate of the center
        y0 (float): y-coordinate of the center
        
    Returns:
        lists of the x and y coordinate of the pixel in each cone
        
    Raises:
        None
    """
    #SETUP THE EXTRACTION REGIONS PROPERTIES#
    angles=np.arange(-180*((nregions-1)/nregions),180,360/nregions) #we define the angles from -180 to 180, since np.arctan2 uses this definition
    angles=np.radians(angles) #convert the angles into radians
    pa=np.radians(pa-180) #convert the position angle into radians
    inc=np.radians(inc) #convert the inclination angle into radians
    xmin=round(x0-rmax) #store the min x-coordinate
    if xmin<0: #if xmin is less than 0
        xmin=0 #set to 0
    xmax=round(x0+rmax) #store the max x-coordinate
    if xmax>=data.shape[2]: #if xmax is greater than the x-shape of the data
        xmax=data.shape[2]-1 #set to the x-shape of the data
    ymin=round(y0-rmax) #store the min y-coordinate
    if ymin<0: #if ymin is less than 0
        ymin=0 #set to 0
    ymax=round(y0+rmax) #store the max y-coordinate
    if ymax>=data.shape[1]: #if xmax is greater than the y-shape of the data
        ymax=data.shape[1]-1 #set to the y-shape of the data
            
    #to check if a point belongs to a given cone or not, we compare the tan of the angle defined by the line connecting the point and the rotation center and the x-axis.
    #If that angles is in between angle[i] and angle[i+1], the point belongs to the i-cone.
    #For each pixel in the square (xmin:xmax,ymin:ymax) we check if it resides whitin rmin and rmax.
    #If that is true, then we assign the pixel to the correct region by iterating to the angles until it ends up in the correct region
        
    xvalid=[] #initialize the x-coordinates list of the pixel in the cones
    yvalid=[] #initialize the y-coordinates list of the pixel in the cones
    for i in range(nregions): #append an empty list to create a list of nregions list. We will place in the i-list the pixel belonging to the i-region
        xvalid.append([])
        yvalid.append([])
    yrange=np.arange(ymin,ymax+1) #define the y-range
    xrange=np.arange(xmin,xmax+1) #define the x-range
    with tqdm(desc='Assigned pixel') as pbar:
        for j in yrange: #run over the y-range
            for k in xrange: #run over the x-range
                x=k-x0 #x-position of the pixel w.r.t. the center
                y=j-y0 #y-position of the pixel w.r.t. the center
                d1=__radius(rmin,x,y,pa,inc) #calculate the elliptical radius of the pixel and compare the distance w.r.t. to the min radius
                d2=__radius(rmax,x,y,pa,inc) #calculate the elliptical radius of the pixel and compare the distance w.r.t. to the max radius
                if d1*d2 <= 0: #if one of the distance is negative, means that the points resides between the innermost and outermost radius, hence, it is a valid pixel for the stacking. Indeed, if a pixel is inside rmin, then d1<0 but also d2<0 and so d1*d2>0. If the pixel is outside rmax, then d1>0 but also d2>0 and so d1*d2>0.
                    valid=False #initialize the valid boolean for the check. We run over each cone and stop when the correct cone is found
                    if not np.all(np.isnan(data[:,j,k])): #check if the whole spectrum is nan
                        #We now have to check to which cone it belongs
                        for i in range(len(angles)): #we have to run over the angles and check between which angle the pixel j,k is 
                            if i == len(angles)-1: #if you are at the last angle
                                idx=0 #the i+1 is the first angle
                            else:
                                idx=i+1
                            if (angles[i] > 0 and angles[idx] > 0) or (angles[i] < 0 and angles[idx] > 0): #if both angles are positive or angle[i] is negative and angle[i+1] is positive
                                if angles[i] <= np.arctan2(y,x) <= angles[idx]: #the point angle must be in between angle[i] and angle[i+1]
                                    valid=True
                            elif angles[i] > 0 and angles[idx] < 0: #if angle[i] is positive and angle[i+1] is negative
                                if angles[i] <= np.arctan2(y,x) or np.arctan2(y,x) <= angles[idx]: #the point angle must be higher than angle[i] but lower than angle[i+1]
                                    valid=True
                            elif angles[i] < 0 and angles[idx] < 0: #if both angles are negative
                                if angles[idx] >= np.arctan2(y,x) >= angles[i]: #the point angle must be in between angle[i+1] and angle[i]
                                    valid=True
                            if valid:
                                xvalid[i].append(k) #append the x-coordinate
                                yvalid[i].append(j) #append the y-coordinate
                                pbar.update(1)
                                break
    return xvalid,yvalid

#############################################################################################
def __auto_fit_fontsize(text,width,height,fig=None,ax=None):
    """Auto-fit the fontsize of a text object to a specified box-size.

    Args:
        text (matplotlib.text.Text): Text to resize
        width (float): Allowed width in data coordinates
        height (float): Allowed height in data coordinates
        fig (matplotlib.pyplot.figure): Figure to use
        ax (matplotlib.pyplot.axes): Axes to use
    """
    fig=fig or plt.gcf() #access the figure
    ax=ax or plt.gca() #access the axes

    renderer=fig.canvas.get_renderer() #define the render
    bbox_text=text.get_window_extent(renderer=renderer) #define the text box in figure coordinates
    bbox_text=Bbox(ax.transData.inverted().transform(bbox_text)) #transform bounding box to data coordinates

    #evaluate if text fits and recursively decrease fontsize until text fits
    fits_width=bbox_text.width < width if width else True
    fits_height=bbox_text.height < height if height else True
    if not all((fits_width, fits_height)): #if the width and/or the height do not fit
        text.set_fontsize(text.get_fontsize()-1) #reduce the fontsize
        __auto_fit_fontsize(text,width,height,fig,ax) #restart the function
            
############################################################################################# 
def __covariance_to_error_ellipse(covariance_matrix,idx1,idx2):
    """Calculate the radii and position angle of the error ellipse corresponding to a given 3x3 covariance matrix for a set of 2 parameters.

    Parameters:
        covariance_matrix (array-like): the 3x3 covariance matrix
        idx1 (float): index of the first parameter (0, 1 or 2)
        idx2 (float): index of the second parameter (0, 1 or 2). It differs from idx1.
    Returns:
        tuple: A tuple containing three values:
            - radius_maj (float): The major radius (standard deviation) of the error ellipse.
            - radius_min (float): The minor radius (standard deviation) of the error ellipse.
            - pa (float): The position angle (in degrees) of the major axis of the error ellipse. The position angle is measured counter-clockwise from the x-axis.
    """
    cov_xx=covariance_matrix[idx1,idx1] #extract the xx component of the covariance matrix
    cov_yy=covariance_matrix[idx2,idx2] #extract the yy component of the covariance matrix
    cov_xy=covariance_matrix[idx1,idx2] #extract the xy component of the covariance matrix

    eigenvalues,eigenvectors=np.linalg.eig(covariance_matrix) #calculate the eigenvalues and eigenvectors of the covariance matrix

    #sort the eigenvalues and eigenvectors in descending order
    sorted_indices=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[sorted_indices]
    eigenvectors=eigenvectors[:,sorted_indices]
    
    radius_maj=np.sqrt(eigenvalues[0]) if eigenvalues[0]>eigenvalues[1] else np.sqrt(eigenvalues[1]) #standard deviation along one axis
    radius_min=np.sqrt(eigenvalues[1]) if eigenvalues[0]>eigenvalues[1] else np.sqrt(eigenvalues[0])  #standard deviation along the other axis

    if cov_xy == 0: #if there is no covariance between x and y, the ellipse is aligned with the axes
        pa=0.0
    else:
        pa=np.degrees(np.arctan2(2*cov_xy,cov_xx-cov_yy)/2) #calculate the position angle (in degrees) of the major axis

    return radius_maj,radius_min,pa
    
#############################################################################################
def __load(data_to_load):
    """Load data provided in various type and return them as numpy.ndarray. If the data are fits, it returns also the header as astropy.fits.io.header object.

    Args:
        data_to_load (str/PrimaryHUD/HUDlist-like): name or path+name of the fits file, or PrimaryHUD or HUDList or array-like variable of the data
        
    Returns:
        data as numpy.ndarray
        
    Raises:
        ValueError: If input given in the wrong format
    """
    if type(data_to_load)==str: #if the data are given as a string, it is assumed it is the path to a fits file
        with fits.open(data_to_load) as Data: #open the fits file
            data=Data[0].data #get the data
            header=Data[0].header #get the header
        del Data[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
        return data,header
    elif type(data_to_load)==fits.HDUList: #if the data are given as an HDU list
        data=Data[0].data #get the data
        header=Data[0].header #get the header
        return data,header
    elif type(data_to_load)==fits.PrimaryHDU: #if the data are given as an primary HDU
        data=Data.data #get the data
        header=Data.header #get the header
        return data,header
    elif type(data_to_load)==list: #if the data are given as a list
        data=np.array(data_to_load) #convert the data in an array
        header=None #there is no header
        return data,header
    elif type(data_to_load)==np.ndarray or type(data_to_load)==np.array or type(data_to_load)==tuple:  #if the data are given as an array or tuple
        data=data_to_load #get the data
        header=None #there is no header
        return data,header
    else:
        raise ValueError(f'Wrong data type: {type(data_to_load)}. Accepted types are\nstring, PrimaryHUD, HUDList, or array-like.')
                                        
#############################################################################################
def __load_box(data,box):
    """Load the pixel coordinates of a spatial box and return them as xmin, xmax, ymin, ymax.

    Args:
        data (array-like): 3D or 2D data
        box [array-like]: spatial box as [xmin,xmax,ymin,ymax]
        
    Returns:
        box pixel coordinates as float in terms of xmin, xmax, ymin, ymax
        
    Raises:
        ValueError: If the length of the box coordinates is > 4.
    """
    if len(data.shape)==3: #if the data are 3D
        xshape=data.shape[2] #get the size of the x-axis
        yshape=data.shape[1] #get the size of the y-axis
    else: #if the data are 2D
        xshape=data.shape[1] #get the size of the x-axis
        yshape=data.shape[0] #get the size of the y-axis
    if box is None: #if no box is given
        xmin=ymin=0 #start from 0
        xmax=xshape #select until the last x-pixel
        ymax=yshape #select until the last y pixel
    elif len(box) != 4: #if the box has the wrong size
        raise ValueError('ERROR: Please provide the box in the format [xmin,xmax,ymin,ymax]. Aborting!')
    else: #store the spatial box from the input
        xmin=box[0]
        xmax=box[1]
        ymin=box[2]
        ymax=box[3]
        if xmin < 0: #if xmin is negative
            warnings.warn(f'Lower x limit is negative ({xmin}): set to 0.')
            xmin=0 #set it to 0
        if xmax > xshape: #if xmax is too high
            warnings.warn(f'Max x is too high ({xmax}): set to the size of x.')
            xmax=xshape #set it to size of data
        if ymin < 0: #if ymin is negative
            warnings.warn(f'Lower y limit is negative ({ymin}): set to 0.')
            ymin=0 #set it to 0
        if ymax > yshape: #if ymax is too high
            warnings.warn(f'Max y is too high ({ymax}): set to the size of y.')
            ymax=yshape #set it to size of data
            
    return xmin,xmax,ymin,ymax

#############################################################################################
def __normalize(arr,new_min,new_max):
    """Normalize the input array by scaling its values to fit within the range [new_min, new_max].

    Parameters:
        arr (array-like): The input array to be normalized.
        new_min (float): The desired minimum value of the array
        new_max (float): The desired maximum value of the array

    Returns:
        np.array: A new array with values scaled to the range [new_min, new_max].
    """
    arr=np.array(arr) #convert into a numpy array
    old_min=np.nanmin(arr) #get the old minimum
    old_max=np.nanmax(arr) #get the old maximum

    normalized_arr=[((x-old_min)/(old_max-old_min))*(new_max-new_min)+new_min for x in arr] #do the normalization

    return np.array(normalized_arr)
    
#############################################################################################
def __plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim):
    """Plot the beam on the lower right corner of the current axis.

    Args:
        pixelres (float): cube spatial resolution in pixunits
        bmaj (float): beam major axis in arcsec in pixunits
        bmin (float): beam minor axis in arcsec in pixunits
        bpa (float): beam position angle in degree
        xlim (array-like): x-axis limit of the plot
        ylim (array-like): y-axis limit of the plot
        
    Returns:
        The current axis with the beam in the lower right corner
        
    Raises:
        None
    """
    ax=plt.gca() #get the current axis
    pxbeam=np.array([bmaj,bmin])/pixelres #beam in pixel
    box=patch.Rectangle((xlim[1]-2*pxbeam[0],ylim[0]),2*pxbeam[0],2*pxbeam[1],fill=None) #create the box for the beam. The box start at the bottom and at twice the beam size from the right edge and is twice the beam large. So, the beam is enclosed in a box that extend a beam radius around the beam patch
    beam=patch.Ellipse((xlim[1]-pxbeam[0],ylim[0]+pxbeam[1]),pxbeam[0],pxbeam[1],bpa,hatch='/////',fill=None) #create the beam patch. The beam center is at a beamsize (in pixel) from the plot border
    ax.add_patch(box) #add the beam box
    ax.add_patch(beam) #add the beam

    return ax

#############################################################################################
def __plot_kpcline(pixelres,asectokpc,xlim,left_space,upper_space):
    """Plot the kpc reference line on the upper left corner of the current axis.

    Args:
        pixelres (float): cube spatial resolution in arcsec
        asectokpc (float): arcsec to kpc conversion 
        xlim (array-like): x-axis limit of the plot
        left_space (float): width of the left margin in axis units (from 0 to 1)
        upper_space (float): width of the top margin in axis units (from 0 to 1)
        
    Returns:
        The current axis with the kpc reference line in the upper left corner
        
    Raises:
        None
    """    
    ax=plt.gca() #get the current axis
    kpcline=10/(asectokpc*pixelres) #length of the 10 kpc line in pixel
    kpcline=kpcline/(xlim[1]-xlim[0]) #lenght of the 10 kpc line in axes size
    ax.hlines(upper_space-0.015,left_space,left_space+kpcline,color='black',linewidth=2,transform=ax.transAxes) #add the 10 kpc line
    ax.text(left_space+(kpcline/2),upper_space,'10 kpc',ha='center',transform=ax.transAxes)
                    
    return ax
    
#############################################################################################
def __plot_stack_result(v,spectrum,rms,expected_rms,smooth_rms,fluxunits,mask=None,aligned=False,nrows=1,ncols=2,color='blue',idx=1,smooth_kernel=None,sc_threshold=None,link_kernel=None,**converttoHI_kwargs):
    """Plot the stacked spectrum and the rms as a function of the number of stacked spectra.

    Args:
        v (array-like): spectral axis
        spectrum (array-like): stacked spectrum
        rms (array-like): rms value after each stacking iteration
        expected_rms (array-like): expected rms after each stacking iteration
        smooth_rms (float): rms value after the last smoothing iteration. Set it to None if no smoothing is applied
        fluxunits (str): flux units of the spectrum
        mask (array-like): optional mask. Set to None to disable
        aligned (bool): tells if the spectrum is aligned (True) or not w.r.t. the redshift
        pixelres (float): pixel resolution of the data
        nrows (int): number of rows in the plot
        ncols (int): number of cols in the plot
        color (str): color of the spectrum and rms lines
        idx (int): index of the plot (idx <= nrows*ncols)
        smooth_kernel (int/list of int): boxcar kernel size (or list of kernel sizes) to apply. The individual kernel sizes must be odd integer values of 3 or greater and denote the full width of the Boxcar filter used to smooth the spectrum. Set to None or 1 to disable.
        sc_threshold (float): flux threshold to be used by the source finder relative to the measured rms in each smoothing iteration. Values in the range of about 3 to 5 have proven to be useful in most situations, with lower values in that range requiring use of the reliability filter to reduce the number of false detections.
        link_kernel (int): minimum size of sources in channels. Sources that fall below this limit will be discarded by the linker.
        
    Kwargs:
        fluxunits (str): string with the flux units
        beamarea (float): area of the beam in pixunits^2
        pixelres (float): pixel resolution of the data
        pixunits (str): String with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - arcmin
            - arcsec
        specunits (str): String with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        spectralres (float): data spectral resolution in specunits
            
    Returns:
        Plot of the stacked spectrum, the measured and the expected rms as matplotlib.pyplot.figure.
        
    Raises:
        None
    """
    N=len(rms) #number of stacked spectra
    fig=plt.gcf() #get the figure
    
    ax=fig.add_subplot(nrows,ncols,idx) #create the subplot for the spectrum
    if idx==1: #if the plot is in the first row
        ax.set_title('Spectrum',fontsize=20,pad=10) #set the title
    ax=__plot_stack_spectrum(v,spectrum,mask,color,aligned,rms[-1],smooth_rms,fluxunits,smooth_kernel) #plot the stacked spectrum
            
    ax=fig.add_subplot(nrows,ncols,idx+1) #create the subplot for the rms
    if idx==1: #if the plot is in the first row
        ax.set_title('RMS',fontsize=20,pad=10) #set the title
    ## The new beamarea is the regionarea/beamarea
    regionarea=converttoHI_kwargs['pixelres']**2*N #the region area is the number of pixel in the area (which is the number of stacked spectra) multiplited by the pixel resolution squared
    ##
    beamarea=converttoHI_kwargs['beamarea']#regionarea/converttoHI_kwargs['beamarea']
    pixunits=converttoHI_kwargs['pixunits'] #get the pixel units
    spectralres=converttoHI_kwargs['spectralres'] #get the spectral resolution
    specunits=converttoHI_kwargs['specunits'] #get the spectral units
    rms=converttoHI(rms,fluxunits=fluxunits,beamarea=beamarea,pixunits=pixunits,spectralres=spectralres,specunits=specunits) #convert the rms into column density
    expected_rms=converttoHI(expected_rms,fluxunits=fluxunits,beamarea=beamarea,pixunits=pixunits,spectralres=spectralres,specunits=specunits) #convert the expected rms into column density
    ax.plot(np.arange(1,N+1),rms,lw=2,c=color,label='Measured rms') #plot the stacked rms
    ax.plot(np.arange(1,N+1),expected_rms,lw=2,c='gray',ls='--',label='Expected rms') #plot the expected rms
    ax.set_xscale('log') #set the x-axis to log scale
    ax.set_yscale('log') #set the y-axis to log scale
    ax.set_xlabel('Number of stacked spectra') #set the x-axis label
    ax.set_ylabel('RMS [cm$^{-2}$]') #set the y-axis label
    ax.set_xlim(1,N) #set the xlim
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    if None not in [sc_threshold,link_kernel]: #if the source finder is used
        ax.text(0.025,0.05,f"{sc_threshold}\u03C3 {link_kernel}-channel detection limit: {rms[-1]*sc_threshold*link_kernel:.1e} {'cm$^{-2}$'}",transform=ax.transAxes) #add the information of the detection limit
    ax.legend(loc='upper right')
    
    return fig

#############################################################################################   
def __plot_stack_spectrum(v,spectrum,mask,color,aligned,rms,smooth_rms,fluxunits,smooth_kernel=None):
    """Plot the stacked spectrum in the current matplotlib.pyplot.axis.

    Args:
        v (array-like): spectral axis
        spectrum (array-like): stacked spectrum
        mask (array-like): optional mask. Set to None to disable
        color (str): color of the spectrum
        aligned (bool): tells if the spectrum is aligned (True) or not w.r.t. the redshift
        rms (array-like): rms value to plot
        smooth_rms (float): rms value after the last smoothing iteration. Set it to None if no smoothing is applied
        fluxunits (str): flux units of the spectrum
        smooth_kernel (int/list of int): boxcar kernel size (or list of kernel sizes) to apply. The individual kernel sizes must be odd integer values of 3 or greater and denote the full width of the Boxcar filter used to smooth the spectrum. Set to None or 1 to disable.
        
    Returns:
        Plot of the stacked spectrum as matplotlib.pyplot.axis object.
        
    Raises:
        None
    """
    ax=plt.gca() #get the current axis
    exponent=int(np.nanmean(np.log10(np.abs(spectrum[np.where((~np.isnan(spectrum)) & (spectrum != 0))])))) #mean power of the flux values
    if mask is None: #if no mask is provided
        ax.plot(v,spectrum/10**exponent,c=color,lw=1,zorder=100) #plot the spectrum
    else:
        if ~np.all(np.isnan(mask)): #if a mask is provided means that there is a source in the spectrum
            mask[mask>0]=1 #set to 1 the positive values
        if smooth_kernel is not None: #if smoothing kernel have been used in the source finder
            alpha=np.linspace(0,1,len(smooth_kernel)+1) #set the transparency of the plot
            lw=np.arange(len(smooth_kernel))+1 #set the linewidth
            for i in range(len(smooth_kernel)): #for each kernel
                if i==len(smooth_kernel)-1: #if we are plotting the last kernel we add the legend labels
                    ax.plot(v,conv.convolve(spectrum/10**exponent,conv.Box1DKernel(smooth_kernel[i])),c=color,lw=lw[i],ls='dotted',alpha=alpha[i+1],label='Noise',zorder=100) #plot the smoothed stacked spectrum
                    if ~np.all(np.isnan(mask)): #if a source is in the spectrum
                        ax.plot(v,conv.convolve(spectrum/10**exponent,conv.Box1DKernel(smooth_kernel[i]))*mask,c=color,lw=lw[i],alpha=alpha[i+1],label='Source',zorder=100) #plot the source smoothed stacked spectrum in the channel range covered by the galaxy emission
                else:
                    ax.plot(v,conv.convolve(spectrum/10**exponent,conv.Box1DKernel(smooth_kernel[i])),c=color,lw=lw[i],ls='dotted',alpha=alpha[i+1],zorder=100) #plot the smoothed stacked spectrum
                    if ~np.all(np.isnan(mask)): #if a source is in the spectrum
                        ax.plot(v,conv.convolve(spectrum/10**exponent,conv.Box1DKernel(smooth_kernel[i]))*mask,c=color,lw=lw[i],alpha=alpha[i+1],zorder=100) #plot the masked smoothed stacked spectrum in the channel range covered by the galaxy emission
        else:
            ax.plot(v,spectrum/10**exponent,c=color,lw=2,ls='dotted',label='Noise',zorder=100) #plot the full spectrum
            if ~np.all(np.isnan(mask)): #if a source is in the spectrum
                ax.plot(v,mask*spectrum/10**exponent,c=color,lw=2,label='Source',zorder=100)  #plot the masked stacked spectrum in the channel range covered by the galaxy emission
        ax.legend(loc='upper right',prop={'size':11}).set_zorder(1000)
    ax.axhline(y=0,ls='--',c='black') #draw the 0-flux line
    if aligned: #if the spectrum is the aligned spectrum
        ax.axvline(x=0,ls='-.',c='black') #draw the 0 velocity line
    if smooth_rms is not None: #if a smoothing is applied
        ax.axhline(y=smooth_rms/10**exponent,ls='--',c='dimgray') #draw the smoothed rms line
        ax.axhline(y=-smooth_rms/10**exponent,ls='--',c='dimgray') #draw the smoothed rms line
    ax.axhline(y=rms/10**exponent,ls='--',c='silver') #draw the final rms line
    ax.axhline(y=-rms/10**exponent,ls='--',c='silver') #draw the final rms line
    ax.set_xlabel('Velocity [km/s]') #set the x-axis label
    ax.set_ylabel(fr'<Flux> [$\mathdefault{{10^{{{exponent}}}}}$ $\mathdefault{{{fluxunits}}}$]') #set the y-axis label
    ax.set_xlim(np.nanmin(v),np.nanmax(v))
    
    return ax
    
#############################################################################################
def __radius(radius,x,y,pa,inc):
    """Calculate the radial position of a point (x,y) with respect to a reference radius (radius).

    Args:
        radius (float): reference radius over which the radial distance of a point (x,y) is calculated
        x,y (float): x and y coordinate in pixel of a point in a 2D array
        pa (float): position angle in degree
        inc (float): inclination in degrees (0 deg means face-on)
    Returns:
        radial distance of the point (x,y) with respect to the reference radius as float
        
    Raises:
        None
    """
    xr=(-x*np.sin(pa)+y*np.cos(pa)) #x-radius is the same of a circle
    yr=(-x*np.cos(pa)-y*np.sin(pa))/np.cos(inc) #y-radius is the same of a circle but reduced by the inclination angle
    return(radius-np.sqrt(xr**2+yr**2)) #return the distance between the radius[i] and the elliptical radius of the point
        
#############################################################################################
def __read_string(string,object,**kwargs):
    """Read the string of a file name or of a path to the file and build the variable to be used by the other functions.

    Args:
        string (str): file name or of a path to the file
        object (str): which object (datacube, maskcube, ...) is the string referring to
    
    Kwargs:
        path (str): (optional) path to the file. Used if 'string' is not a path to the file
        
    Returns:
        string with the path and the file name
        
    Raises:
        None
    """
    if string == '' or string is None: #if a string is not given
        if object in kwargs: #if the object is in kwargs
            if kwargs[object] == '' or kwargs[object] is None:
                raise ValueError(f'ERROR: {object} is not set: aborting!')
            else:
                string=kwargs[object] #store the string from the input kwargs
        else:
            raise ValueError(f'ERROR: {object} is not set: aborting!')
    if string[0] != '.': #if the string start with a . means that it is a path to the files (so differs from path parameter)
        path=kwargs.get('path',None)
        if path == '' or path is None:
            raise ValueError(f'ERROR: no path to the {object} is set: aborting!')
        else:
            string=path+string
            
    return string
    
#############################################################################################
def __reliability(pos_sources,neg_sources,snrmin,threshold,rel_kernel,**kwargs):
    """Calculate the reliability of the positive sources in a catalogue, by comparing their density in each position of the parameter space with the density of negative sources in the same position. All the fluxes must be divided by the rms of the data prior calling this function.

    Args:
        pos_sources (array-like): array-like of array where each array contains the flux density (i.e., divided by the rms) per channel of a positive source
        neg_sources (array-like): array-like of array where each array contains the flux density (i.e., divided by the rms) per channel of a negative source
        snrmin (float): lower signal-to-noise limit for reliable sources. Detections that fall below this threshold will be classified as unreliable and discarded. The value denotes the integrated signal-to-noise ratio, SNR = Fsum/(RMS√N), of the source, where N is the number of channels of the source, Fsum is the summed flux density and RMS is the local RMS noise level (assumed to be constant). Note that the spectral resolution is assumed to be equal to the channel width.
        threshold (float): reliability threshold in the range of 0 to 1. Sources with a reliability below this threshold will be discarded.
        rel_kernel (float): scaling factor for the size of the Gaussian kernel used when estimating the density of positive and negative detections in the reliability parameter space.
        
    Kwargs:
        objname (str): name of the object (Default: '')
        outdir (str): output folder to save the plots
        outname (str): output file name
        plot_format (str): file format of the plots
        ctr_width (float): line width of the contours (Default: 2)
        verbose (bool): option to print messages and plot to terminal if True 
        
    Returns:
        Array with the values of the reliability for each positive source
        
    Raises:
        None
    """   
    #CHECK THE KWARGS#
    verbose=kwargs.get('verbose',False) #store the verbosity
    objname=kwargs.get('objname',None)
    outdir=kwargs.get('outdir',os.getcwd()+'/')
    outname=kwargs.get('outname','reliability_result')
    if objname is None or objname == '':
        outname=outdir+outname
    else:
        outname=outdir+objname+'_'+outname
    plot_format=kwargs.get('plot_format','.pdf')
    ctr_width=kwargs.get('ctr_width',2)
    
    #CALCULATE THE PARAMETERS FOR RELIABILITY CALCULATION
    #initialize the list of the parameters. Each ith-element of each list is the parameter value of the source i
    pos_peak=[] #initialize the list for the peak values
    pos_sum=[] #initialize the list of the sum values
    pos_mean=[] #initialize the list of the mean values
    for i in range(len(pos_sources)): #run over the positive sources
        pos_peak.append(np.log10(np.nanmax(pos_sources[i]))) #calculate log10(peak/rms)
        pos_sum.append(np.log10(np.nansum(pos_sources[i]))) #calculate log10(sum/rms)
        pos_mean.append(np.log10(np.nanmean(pos_sources[i]))) #calculate log10(mean/rms)
    pos_peak=np.array(pos_peak) #convert the list into an array
    pos_sum=np.array(pos_sum) #convert the list into an array
    pos_mean=np.array(pos_mean) #convert the list into an array
    
    neg_peak=[] #initialize the list for the peak values
    neg_sum=[] #initialize the list of the sum values
    neg_mean=[] #initialize the list of the mean values
    for i in range(len(neg_sources)): #run over the negative sources
        neg_peak.append(np.log10(-np.nanmin(neg_sources[i]))) #calculate log10(peak/rms)
        neg_sum.append(np.log10(-np.nansum(neg_sources[i]))) #calculate log10(sum/rms)
        neg_mean.append(np.log10(-np.nanmean(neg_sources[i]))) #calculate log10(mean/rms)
    neg_peak=np.array(neg_peak) #convert the list into an array
    neg_sum=np.array(neg_sum) #convert the list into an array
    neg_mean=np.array(neg_mean) #convert the list into an array

    #PREPARE THE DATA FOR KDE ESTIMATION
    tot_peak=np.concatenate((pos_peak,neg_peak)) #concatenate log10(peak/rms) of positive and negative into a single array
    tot_sum=np.concatenate((pos_sum,neg_sum)) #concatenate log10(sum/rms) of positive and negative into a single array
    tot_mean=np.concatenate((pos_mean,neg_mean)) #concatenate log10(mean/rms) of positive and negative into a single array
    rel_params=np.array([tot_peak,tot_sum,tot_mean]) #create the parameter space for all sources
    pos_rel_params=np.array([pos_peak,pos_sum,pos_mean]) #create the parameter space for positive sources
    neg_rel_params=np.array([neg_peak,neg_sum,neg_mean]) #create the parameter space for negative sources
    
    #PREPARE THE FIGURE TO PLOT THE RESULTS
    nrows=2 #number of rows in the figure
    ncols=3 #number of columns in the figure        
    fig=plt.figure(figsize=(6*ncols,6*nrows)) #create the figure
    ax=[] #initialize the plot list
    
    #CALCULATE THE DENSITY OF POSITIVE AND NEGATIVE SOURCES IN THE PARAMETER SPACE
    #We are going to use the multivariate kernel density estimation (https://en.wikipedia.org/wiki/Multivariate_normal_distribution). The probability density function in a point (x,y) given a sample of points populating the 2D space assuming they comes from a gaussian distribution (i.e., are noise measurements) is given by pdf=sum_j{exp(-0.5*(x_j-mu).T ⋅ covar_inv ⋅ (x_j-mu))}, where covar_inv is the inverse of the covariance matrix, mu is the point (x,y) at which we want to calculate the pdf and x_j is the value of another point in the space. There is a normalization factor that we can omit as we don't care that sum(pdf)=1 for all the points in the space. 
    
    scale_factor=1 #DO NOT CHANGE IT. From the source code of SoFiA2: 'this can be set to 1, as we don’t really care about the correct normalisation of the Gaussian kernel, so we might as well normalise the amplitude to 1 rather than the integral. The normalisation factor does matter for the Skellam parameter, though.'
    
    covar=np.cov(neg_rel_params,rowvar=True) #determine covariance matrix from negative detections. We use this matrix because we are assuming that the negative detections are only due to noise, hence, they represent the statistics of the data
    covar*=rel_kernel**2 #scale the covariance matrix
    try:
        covar_inv=np.linalg.inv(covar) #invert covariance matrix
    except np.linalg.LinAlgError:
        raise AssertionError("Covariance matrix is not invertible; cannot measure reliability.\nEnsure that there are enough negative detections.")
    rel=[] #initialize the reliability list. The i-element of the list is the reliability evaluation of the positive source i
    for i in range(len(pos_sources)): #for each positive source
        n=len(pos_sources[i]) #number of channels contributing to a source
        SNR=np.nansum(pos_sources[i])/np.sqrt(n) #SNR of the source
        if SNR>snrmin: #if the SNR is above the chosen minimum
            N=0 #initialized the total negative pdf
            for j in range(len(neg_sources)): #for each negative source
                vector=neg_rel_params.T[j]-pos_rel_params.T[i] #calculate 3D the distance between the point j and the point i. It is equivalent of (x-mu) for the multivariate kernel density estimation. mu is the point where the pdf is maximum, that is the point itself
                N+=np.exp(-0.5*np.dot(np.dot(vector.T,covar_inv),vector))*scale_factor #multivariate kernel density estimation
            P=0 #initialized the total positive pdf
            for j in range(len(pos_sources)): #for each negative source
                vector=pos_rel_params.T[j]-pos_rel_params.T[i] #calculate the 3D distance between the point j and the point i. It is equivalent of (x-mu) for the multivariate kernel density estimation. mu is the point where the pdf is maximum, that is the point itself
                P+=np.exp(-0.5*np.dot(np.dot(vector.T,covar_inv),vector))*scale_factor #multivariate kernel density estimation
            rel.append((P-N)/P) if P > N else rel.append(0) #determine reliability
        else:
            rel.append(0)
    
    #DO THE PLOTS#
    s=(0.25*plt.rcParams['lines.markersize'])**2 #size of the scatter points
    for i in range(len(rel_params)): #run over the dimensions of the parameter space
        #we must select the correct 2D space from the 3D parameter space at each iteration
        if i==0: #peak vs sum
            idx1=0 #index of the peak list in the parameter space
            idx2=1 #index of the sum list in the parameter space
        elif i==1: #peak vs mean
            idx1=0 #index of the peak list in the parameter space
            idx2=2 #index of the mean list in the parameter space
        else: #sum vs mean
            idx1=1 #index of the sum list in the parameter space
            idx2=2 #index of the mean list in the parameter space
             
        ax.append(fig.add_subplot(nrows,ncols,i+1)) #create the plot for the result of the iteration

        rmaj,rmin,pa=__covariance_to_error_ellipse(covar,idx1,idx2) #convert the covariance matrix into error ellipses
        for j in range(3):
            ls='solid' if j==0 else 'dashed' if j==1 else 'dotted' #style of the ellipse: solid for 1sigma, dashed for 2sigma, dotted for 3sigma
            e=patch.Ellipse([np.nanmean(neg_rel_params[idx1]),np.nanmean(neg_rel_params[idx2])],(j+1)*rmaj,(j+1)*rmin,pa,fill=False,ec='silver',ls=ls,lw=ctr_width-1) #define the ellipse 
            ax[i].add_patch(e) #draw the ellipse
            
        ax[i].scatter(neg_rel_params[idx1],neg_rel_params[idx2],s=s,c='hotpink',label='Negative') #plot the negative sources
        ax[i].scatter(pos_rel_params[idx1],pos_rel_params[idx2],s=s,c='royalblue',label='Positive') #plot the positive sources
        
        xmin,xmax=ax[i].get_xlim() #get the x-limits
        xextend=(xmax-xmin)*0.05 #increase the x-axis by 5%
        xmin-=xextend #apply the increase
        xmax+=xextend #apply the increase
        ax[i].set_xlim([xmin,xmax]) #set the new x-limits
        ymin,ymax=ax[i].get_ylim() #get the y-limits
        yextend=(ymax-ymin)*0.05 #increase the y-axis by 5%
        ymin-=yextend #apply the increase
        ymax+=yextend #apply the increase
        ax[i].set_ylim([ymin,ymax]) #set the new y-limits
        if i==0: #set the title and the label according to which 2D space we are using
            ax[i].set_title('Peak vs sum',pad=10)
            ax[i].set_xlabel('log(peak/rms)')
            ax[i].set_ylabel('log(sum/rms)')
        elif i==1:
            ax[i].set_title('Peak vs mean',pad=10)
            ax[i].set_xlabel('log(peak/rms)')
            ax[i].set_ylabel('log(mean/rms)')
        else: #in the sum vs mean space we want to plot also the min SNR line
            ax[i].plot([rel_params[idx1].min(),rel_params[idx1].max()],
                       2*np.log10(snrmin)-[rel_params[idx1].min(),rel_params[idx1].max()],
                       c='gray',ls='--',lw=1,zorder=0)
            ax[i].set_title('Sum vs mean',pad=10)
            ax[i].set_xlabel('log(sum/rms)')
            ax[i].set_ylabel('log(mean/rms)')
            ax[i].text(0.95,0.05,f'SNR min = {snrmin}',ha='right',transform=ax[i].transAxes)
    
    rel=np.array(rel).ravel() #convert the reliability list into an array. ravel() is needed to convert the list of arrays into a 1D array
    rel_x=pos_peak[rel>=threshold] #filter the reliable sources in the peak space
    rel_y=pos_sum[rel>=threshold] #filter the reliable sources in the sum space
    rel_z=pos_mean[rel>=threshold] #filter the reliable sources in the mean space

    for i in range(len(rel_params)): #update the scatter plots with the highlight of the reliable sources
        if i==0:
            ax[i].scatter(rel_x,rel_y,s=4*s,c='black',label='Reliable')
            ax[i].legend(loc='upper left')
        elif i==1:
            ax[i].scatter(rel_x,rel_z,s=4*s,c='black',label='Reliable')
            ax[i].legend(loc='upper left')
        else:
            ax[i].scatter(rel_y,rel_z,s=4*s,c='black',label='Reliable')
            ax[i].legend(loc='upper left')

    ax=fig.add_subplot(2,1,2,projection='3d') #plot the 3D parameter space
    #Below I have to use plot instead of scatter because scatter doesn't deal with the zorder
    ax.plot(neg_peak,neg_sum,neg_mean,ls='',markersize=np.sqrt(s/2),marker='o',c='hotpink',label='Negative')
    ax.plot(pos_peak,pos_sum,pos_mean,ls='',markersize=np.sqrt(s/2),marker='o',c='royalblue',label='Positive')
    ax.plot(rel_x,rel_y,rel_z,ls='',markersize=np.sqrt(2*s),marker='o',c='black',label='Reliable')
    ax.set_title('Peak vs sum vs mean',pad=10)
    ax.set_xlabel('log(peak/rms)')
    ax.set_ylabel('log(sum/rms)')
    ax.set_zlabel('log(mean/rms)')
    ax.legend(loc='upper right',prop={'size':11})
    
    fig.subplots_adjust(wspace=0.4,hspace=0.3)
    fig.savefig(outname+plot_format,dpi=300,bbox_inches='tight')
    
    #DO THE SKELLAM PLOT#
    skellam_list=[]  #initialize the list for the skellam values
    #the Skellam distribution is (P-N)/sqrt(P+N) calculated with the multivariate kernel density estimation on the negative sources
    for i in range(len(neg_sources)): #for each negative source
        N=0 #initialized the total negative pdf
        for j in range(len(neg_sources)): #for each negative source
            vector=neg_rel_params.T[j]-neg_rel_params.T[i] #calculate the 3D distance between the point j and the point i
            N+=np.exp(-0.5*np.dot(np.dot(vector.T,covar_inv),vector)) #multivariate kernel density estimation for negative detections  
        P=0 #initialized the total positive pdf
        for j in range(len(pos_sources)): #for each negative source
            vector=pos_rel_params.T[j]-neg_rel_params.T[i] #calculate the 3D distance between the point j and the point i
            P+=np.exp(-0.5*np.dot(np.dot(vector.T,covar_inv),vector)) #multivariate kernel density estimation for positive detections  
        skellam_list.append((P-N)/np.sqrt(P+N)) #calculate the normalized skellam parameter
    
    skellam=np.sort(np.array(skellam_list).ravel()) #sort the array for the plot. This step is necessary as we are going to normalize it
    skellam=((skellam-np.nanmedian(skellam))/np.nanstd(skellam))+np.nanmedian(skellam) #scale all Skellam values by standard deviation. This is necessary to ensure that the standard deviation of the Skellam parameter values is 1 such that their distribution can be readily compared to a standard Gaussian
    cum=np.arange(len(skellam))/(len(skellam)-1) #calculate the cumulative
    
    x=__normalize(np.arange(-100,101),-4,4) #create the x-axis for the Gaussian cumulative
    y=(0.5-0.5*erf(-x/np.sqrt(2.0))) #the Gaussian cumulative is given by the erf function
    
    fig,ax=plt.subplots(figsize=(8,8))
    ax.set_xlabel('Skellam parameter normalized to $\sigma=1$')
    ax.set_ylabel('Cumulative fraction')
    ax.plot(skellam,cum,c='b',lw=1,label=f'Data: $\mu={np.nanmedian(skellam):.2f}$\nkernel = {rel_kernel}')
    ax.plot(x,y,c='r',lw=1,label='Gaussian: $\mu=0$, $\sigma=1$')
    ax.axvline(x=np.nanmedian(skellam),c='b',ls='dashed',lw=1)
    ax.axvline(x=0,c='r',ls='dotted',lw=1)
    ax.set_xlim([-4,4])
    ax.set_ylim([0,1])
    ax.legend(loc='upper left')
    fig.savefig(outname+'_skellam'+plot_format,dpi=300,bbox_inches='tight')
    
    if verbose:
        plt.show()

    plt.close()
    
    return rel

############################################################################################# 
def __rms(data,mode,fluxrange):
    """Given an array of data, calculate its rms using different algorithm (standard deviation or meadian absolute deviation).

    Args:
        data (array): 1D array with the data
        mode (str): type of algorithm to use for rms calculation ('std' or 'mad').
        fluxrange (str): Flux range to be used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artifacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux.
        
    Kwargs:
        None
        
    Returns:
        Rms of the array.
        
    Raises:
        None
    """   
    mad_to_std=1.48 #conversion between mad and std (std=1.48*mad) 
    
    dummy=data.copy() #create a copy of the data to prevent changes in the input
    if fluxrange == 'negative':
        dummy[dummy>=0]=np.nan #blank the positive flux values for rms calculation
    elif fluxrange=='positive':
        dummy[dummy<=0]=np.nan #blank the negative flux values for rms calculation
        
    if mode=='std':
        rms=np.sqrt(np.nanmean(dummy**2)) #the rms is the same of the standard deviation, if and only if, the mean of the data is 0, i.e., the data are following a Gaussian distribution
    elif mode=='mad':
        rms=np.nanmedian(np.abs(dummy-np.nanmedian(dummy)))*mad_to_std
    
    return rms

############################################################################################# 
def __source_finder(data,threshold):
    """Find the elements of a 1D array whose absolute value is above a threshold).

    Args:
        data (array): 1D array with the data
        threshold (float): flux threshold to be used by the source finder relative to the measured rms in each smoothing iteration. Values in the range of about 3 to 5 have proven to be useful in most situations, with lower values in that range requiring use of the reliability filter to reduce the number of false detections.
        
    Kwargs:
        None
        
    Returns:
        Array of 0/1 values
        
    Raises:
        None
    """
    mask=np.zeros(data.shape) #initialize the mask
    mask[np.abs(data)>threshold]=1 #mark the detections
    
    return mask
    
############################################################################################# 
def __source_linker(data,mask,kernel,min_size):
    """Assign to kernel-size consecutive positive/negative elements of a 1D array the same value representing the index of the source.

    Args:
        data (array): 1D array with the data
        mask (array): 1D array with the mask
        kernel (int): minimum size of sources in channels. Sources that fall below this limit will be discarded by the linker.
        separation (int): maximum channel separation between two sources to be considered a single source
        
    Kwargs:
        None
        
    Returns:
        Array of values representing the sources indexes.
        
    Raises:
        ValueError: If the kernel is not an odd number
    """
    linked_mask=np.zeros(mask.shape) #initialize the linked mask
    pos_index=1 #initialize the positive source index
    neg_index=-1 #initialize the negative source index
    source=False #initialize the source check
    #REJECT THE DETECTIONS COVERING LESS THAN kernel CHANNELS#
    for i in range(len(mask)): #run over the mask
        if np.all(mask[i:i+kernel])>0: #if all the channels in the kernel are marked as a source
            if not source: #if it is a new source
                start=i #store the channel index
                source=True #tell you have a source
            else: #if you are already looking at a source
                pass #do nothing
        else:
            if source: #if you were looking at a source
                if np.nansum(data[start:i+kernel-1])>0: #if the source is positive (-1 accounts for 0-indexing and slicing). Imagine 'start' is channel 5 and the kernel=5. If I link the smallest source, it will cover channels 5, 6, 7, 8 and 9. Which means, channel 10 is a non-detection. The kernel gets to channel 10 when i is 6, because when i is 6 than i:i+kernel (that is 6:11) covers channels 6, 7, 8, 9 and 10 because slicing excludes the last element. So, start:i+kernel (that is 5:11) would be 5, 6, 7, 8, 9, 10. But I don't want 10, hence I use -1: start:i+kernel-1 = 5:10 = [5,6,7,8,9]
                    linked_mask[start:i+kernel-1]=pos_index #assign the positive label to the source (-1 accounts for 0-indexing)
                    pos_index+=1 #increase the positive label
                elif np.nansum(data[start:i+kernel-1])<0: #if the source is negative (-1 accounts for 0-indexing)
                    linked_mask[start:i+kernel-1]=neg_index #assign the negative label to the source (-1 accounts for 0-indexing)
                    neg_index-=1 #decrease the negative label
                else: #if the source has a total flux of 0
                    pass #do nothing
                source=False #tell you are no more looking for a source
    #LINK NEIGHBOUR DETECTIONS#
    i=1
    while i < np.nanmax(linked_mask):
        if (np.where(linked_mask==i+1)[0][0]-np.where(linked_mask==i)[0][-1]-1) <= min_size and np.all(linked_mask[np.where(linked_mask==i)[0][-1]+1:np.where(linked_mask==i+1)[0][0]]==0): #the first condition checks that the separation between two consecutive positive/negative sources, i.e., the difference between the last channel of the former and the first channel of the latter, is less than the maximum allowed separation. -1 is need because if the last channel is 4 and the first is 6, than the separation is 6-4-1=1 and not 6-4=2. The second condition checks that there are not sources between the two (it can happend that you have a positive then negative then positive and the two positives cannot be linked together.    
            linked_mask[np.where(linked_mask>i)[0]]-=1
        else:
            i+=1
    i=-1
    while i > np.nanmin(linked_mask):
        if (np.where(linked_mask==i-1)[0][0]-np.where(linked_mask==i)[0][-1]-1) <= min_size and np.all(linked_mask[np.where(linked_mask==i)[0][-1]+1:np.where(linked_mask==i-1)[0][0]]==0): #see above for description of conditions       
            linked_mask[np.where(linked_mask<i)[0]]+=1
        else:
            i-=1

    return linked_mask

############################################################################################# 
def __source_linker_new(data,mask,kernel,min_size):
    """Assign to kernel-size consecutive positive/negative elements of a 1D array the same value representing the index of the source.

    Args:
        data (array): 1D array with the data
        mask (array): 1D array with the mask
        kernel (int): minimum size of sources in channels. Sources that fall below this limit will be discarded by the linker.
        min_size (int): minimum number of channels a source can cover
        
    Kwargs:
        None
        
    Returns:
        Array of values representing the sources indexes.
        
    Raises:
        ValueError: If the kernel is not an odd number
    """
    pos_idx=1 #initialize the positive source ID
    neg_idx=-1 #initialize the negative source ID
    i=0 #initialize the channel counter
    linked_mask=np.zeros(mask.shape) #initialize the linked mask
    while i <= len(linked_mask): #until we reach the end of the mask
        start=i #initialize the starting channel of a source
        detection=False #initialize the control parameter to increment the source ID
        while True: #start a do-until loop. We run over the channels until a number of kernel consecutive channels are 0
            if np.all(mask[i:i+kernel]==0): #if there are no detected channel in the window
                break #stop the loop
            detection=True #else activate the control switch
            i+=1 #increment the channel and continue the do-until loop
        i+=1 #increment the channel for the next do-until loop
        if detection: #if a detection was found in the do-until loop
            source=mask[start:i].copy() #extract the source
            if len(source[np.where(source!=0)[0]])>=min_size: #if the number of detections (len(...)) in the source is greater than the minimum size
                if np.nansum(data[start:i][np.where(source!=0)[0]]) > 0: #if the total flux is positive
                    linked_mask[start:i][np.where(source!=0)[0]]=pos_idx #set all the non-0 channels scanned in the do-until loop to the source ID
                    pos_idx+=1 #increment the source ID
                elif np.nansum(data[start:i][np.where(source!=0)[0]]) < 0: #if the total flux is negative
                    linked_mask[start:i][np.where(source!=0)[0]]=neg_idx #set all the non-0 channels scanned in the do-until loop to the source ID
                    neg_idx-=1 #increment the source ID
                else: #rarely (but happens) the total flux is 0
                    linked_mask[start:i][np.where(source!=0)[0]]=0 #set all the non-0 channels scanned in the do-until loop to 0
            else:
                linked_mask[start:i][np.where(source!=0)[0]]=0 #set all the non-0 channels scanned in the do-until loop to 0
            detection=False #reset the control parameter

    return linked_mask    

#############################################################################################    
def __stack(data,x,y,weighting,stack_fluxrange='negative',stack_statistics='mad',flip=False,diagnostic=False,**diagnostic_kwargs):
    """Stack the spectra for each sky position (x,y).

    Args:
        data (ndarray): 3D array with the data
        x,y (int): x,y pixel coordinates of the spectra to stack
        weighting (str): type of weight to apply during the stacking between:
            - None (default): the stacked spectrum will be avereged with the number of stacked spectra
            - rms: the stacked spectrum will be avereged with the square of the rms of each stacked spectrum
        stack_fluxrange (str): flux range to be used in the noise measurement of the source finder. If set to 'negative' or 'positive', only pixels with negative or positive flux will be used, respectively. This can be useful to prevent real emission or artefacts from affecting the noise measurement. If set to anything else, all pixels will be used in the noise measurement irrespective of their flux.
        stack_statistics (str): statistic to be used in the noise measurement process of the source finder. Possible values are 'std' or 'mad' for standard deviation and median absolute deviation, respectively. Standard deviation is by far the fastest algorithm, but it is also the least robust one with respect to emission and artefacts in the data. Median absolute deviation is far more robust in the presence of strong, extended emission or artefacts.
        flip (bool): flip the cube along the spectral axis (set to True if the spectral resolution is negative)
        diagnostic (bool): store all the diagnostic files and plots. Warning: the diagnostic might occupy large portions of the disk (default: False)
        
    Kwargs:
        v (array-like): spectral axis
        fluxunits (str): units of the flux
        color (str): color of the plot
        aligned (bool): tells if the spectrum is aligned (True) or not w.r.t. the redshift
        outdir (str): output directiory to store the plots
        plot_format (str): file format of the plots

    Returns:
        Stacked spectrum, measured and expected rms of the stacked spectrum after each stacking as numpy.array each.
        
    Raises:
        None
    """
    N=1 #initialize the number of stacked spectra
    weights=0 #initialize the weights
    stack=np.zeros(data.shape[0]) #initialize the stacked spectrum
    stack_rms=[] #initialize the stacked rms list
    exp=[] #initialize the expected rms list
    
    if diagnostic: #if the diagnostic plots must be made
        v=diagnostic_kwargs.get('v') #store the spectral axis
        fluxunits=diagnostic_kwargs.get('fluxunits') #store the diagnostic plots format
        color=diagnostic_kwargs.get('color') #store the spectrum color
        aligned=diagnostic_kwargs.get('aligned') #store the diagnostic plots format
        outdir=diagnostic_kwargs.get('outdir') #store the diagnostic plots output folder
        plot_format=diagnostic_kwargs.get('plot_format') #store the diagnostic plots format
        if not os.path.exists(outdir): #if the output folder does not exist
            os.makedirs(outdir) #create the folder
            
    if flip: #if the spectra must be flipped
        data=np.flip(data,axis=0) #flip the  cube along the spectral axis
      
    if diagnostic: #if the diagnostic plots must be made
        plt.ioff() #disable the interactive plotting 
    
    ########## W.I.P. DON'T USE IT #############
    if weighting == 'pb':
        path_to_pbcube='../things/NGC_2403_NA_PB_THINGS.fits'
        with fits.open(path_to_pbcube) as pb_cube:
            pb=pb_cube[0].data
        del pb_cube[0].data #as the memory mapping in astropy is enable, when opening a file with memmap=True, because of how mmap works this means that when the HDU data is accessed (i.e. hdul[0].data) another handle to the FITS file is opened by mmap. This means that even after calling hdul.close() the mmap still holds an open handle to the data so that it can still be accessed by unwary programs that were built with the assumption that the .data attribute has all the data in-memory. In order to force the mmap to close either wait for the containing HDUList object to go out of scope, or manually call del hdul[0].data (this works so long as there are no other references held to the data array).
    ############################################   
    for i in tqdm(zip(y,x),desc='Spectra stacked',total=len(x)): #run over the pixels    
        if np.all(np.isnan(data[:,i[0],i[1]])): #if the spectrum to be stacked is empty
            pass
        else:
            if diagnostic: #if the diagnostic plots must be made
                dummy=data[:,i[0],i[1]].copy() #copy the spectrum used in the stacking
                pre_rms=__rms(dummy,stack_statistics,stack_fluxrange) #calculate the total rms
            
                ncols=1 #number of columns in the plot
                nrows=3 #number of rows in the plot
                fig=plt.figure(figsize=(12*ncols,6*nrows)) #create the figure
                fig.suptitle(f'Pre stack, to be stacked and after stack spectrum comparison (N={N})',fontsize=24) #add the title
                
                ax=fig.add_subplot(nrows,ncols,1) #create the subplot for the pre-stack stacked spectrum
                if N==1: #if this is the first iteration
                    ax=__plot_stack_spectrum(v,stack,None,color,aligned,0,None,fluxunits)
                else:
                    ax=__plot_stack_spectrum(v,stack/weights,None,color,aligned,stack_rms[-1],None,fluxunits)
                ax.set_xlabel('')
                ax.xaxis.set_ticklabels([])
                if N>1: #after the first iteration
                    ax.set_ylim(ylim)
                
                ax=fig.add_subplot(nrows,ncols,2) #create the subplot for the spectrum to be stacked
                ax=__plot_stack_spectrum(v,data[:,i[0],i[1]],None,color,aligned,pre_rms,None,fluxunits)
                ax.set_xlabel('')
                ax.xaxis.set_ticklabels([])
            
            if weighting is None:
                w=1 #the weight is 1
            elif weighting == 'rms': 
                w=1/__rms(data[:,i[0],i[1]],stack_statistics,stack_fluxrange)**2 #the weight is 1/rms**2 of the input spectrum
            ########## W.I.P. DON'T USE IT #############
            elif weighting == 'pb':
                    w=(pb[i[0],i[1]])**2
            ############################################
            stack=np.nansum((stack,data[:,i[0],i[1]]*w),axis=0) #stack the input spectrum
            weights+=w #stack the weight
            dummy=stack.copy()/weights #create a dummy copy of the stacked spectrum. Here we divide for the number of stacked weights so far, since we want the rms as if we have stacked all the spectra
                
            stack_rms.append(__rms(dummy,stack_statistics,stack_fluxrange)) #calculate the rms of the stacked spectrum        
            exp.append(stack_rms[0]/np.sqrt(N)) #expected value for the rms
            
            if diagnostic: #if the diagnostic plots must be made
                ax=fig.add_subplot(nrows,ncols,3) #create the subplot for the post-stack stacked spectrum
                ax=__plot_stack_spectrum(v,stack/weights,None,color,aligned,stack_rms[-1],None,fluxunits)
                if N==1: #after the first iteration
                    ylim=ax.get_ylim() #get the ylim
                else:
                    ax.set_ylim(ylim)
                fig.subplots_adjust(top=0.95,hspace=0.1)
                fig.savefig(outdir+f'result_after_{N}_stacked_spectra'+plot_format,dpi=300,bbox_inches='tight')
                plt.close()
            N+=1 #increase the number of stacked spectra
    
    if diagnostic: #if the diagnostic plots must be made
        plt.ion() #re-enable the interactive plotting 
        
    return stack/weights,stack_rms,exp
    
############## CODE FOR ROTATED 2D ELLIPTICAL GAUSSIAN ###############
# def elliptical_gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
#     theta = np.radians(90-theta)
#     a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
#     b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
#     c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#     g = amplitude * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
#     return g