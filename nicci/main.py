"""A collection of functions for the analysis of astronomical data cubes and images.

                NICCI
New analysIs Code for Cubes and Images
"""
#############################################################################################
#
# FILENAME: main
# VERSION: 0.11
# DATE: 06/12/2022
# CHANGELOG:
#   v 0.11: - (NEW) stacking
#           - (NEW) velfi
#           - (NEW) private functions
#               - __assign_to_cones
#               - __get_reliability
#               - __get_sources_catalogue
#               - __load
#               - __load_box
#               - __radius
#               - __read_string
#               - __plot_beam
#               - __plot_kpcline
#               - __plot_stack_result
#               - __plot_stack_spectrum
#               - __source_finder
#               - __source_linker
#               - __stack
#           - importparameters
#               - added options for rotcurve
#               - added options for stacking
#               - added options for velfi
#           - chanmap
#               - implemented __load and __read_string
#               - fixed typo in print
#               - implemented progress bar
#           - cubedo
#               - implemented __load and __read_string
#               - implemented progress bar on operation 'clip' and 'shuffle'
#               - can now work also with numpy.ndarray as input data
#               - changed 'shuffle' in order to wrap the emission
#               - fixed mom0 calculation when spectral resolution is negative
#               - fixed mask application on mom0 calculation
#               - optimized the channels and box import
#               - added option to save the output into a fits file or return a variable
#               - minor fixes
#           - cubestat
#               - implemented __load and __read_string
#           - fixmask
#               - removed optional input directory
#               - implemented __load and __read_string
#           - gaussfit
#               - implemented __load and __read_string
#               - can now work also with numpy.ndarray as input data
#               - implemented progress bar
#               - fixed data initialization when spectralres is negative
#               - fixed datacube and 2d mask import when not giving the input directory
#               - fixed fit calculation when observed spectrum contains nans
#               - changed mask from 2d to 3d
#               - improved mask application
#               - removed optional input directory
#           - getpv
#               - implemented __load and __read_string
#               - can now work also with numpy.ndarray as input data
#               - fixed units in parameters print 
#               - fixed default contour levels
#           - plotmom
#               - implemented __load and __read_string
#               - can now work also with numpy.ndarray as input data
#               - added pb cutoff plot in mom0 when pbcorr is True
#               - corrected FWHM/dispersion ambiguity in plot and in ancillary information
#           - removemod
#               - implemented __load and __read_string
#               - can now work also with numpy.ndarray as input data
#           - rotcurve
#               - completly reworked
#           - create_config
#               - fixed wrong indentations
#               - added options for rotcurve
#               - added options for stacking
#               - added options for velfi
#           - improved import of the output directory
#           - changed default contours colormap for flux maps to hot
#           - improved documentation
#
#   TO DO:  - rotcurve, non fa quello che fa rotcurve in gipsy
#           - aggiungere l'interpolazione in shuffle
#           - aggiungere galmod, fixhead di gipsy
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
from reproject import reproject_interp as spatial_regrid
from scipy.stats import chi2 as statchi2
from scipy.stats import gaussian_kde as kde
from tqdm.auto import tqdm
import configparser
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.patches as patch
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
fits.Conf.use_memmap=False #disable the memomory map when open a fits

#############################################################################################

# print("""||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# |||                                                                                |||
# ||| |||           ||| |||||||| |||         |||||||    |||||    |||    ||| |||||||| |||
# |||  |||         |||  |||      |||       |||||      |||   |||  |||||||||| |||      |||
# |||   |||   |   |||   ||||||   |||      |||        |||     ||| ||| || ||| ||||||   |||
# |||    ||| ||| |||    |||      |||       |||||      |||   |||  |||    ||| |||      |||
# |||     |||   |||     |||||||| |||||||||   |||||||    |||||    |||    ||| |||||||| |||
# |||                                                                                |||
# |||                                                                                |||
# |||                                                                                |||
# |||                             |||||||||    |||||                                 |||
# |||                                |||     |||   |||                               |||
# |||                                |||    |||     |||                              |||
# |||                                |||     |||   |||                               |||
# |||                                |||       |||||                                 |||
# |||                                                                                |||
# |||                                           |||   |||                            |||
# |||                                         |||   |||                              |||
# |||                         |||   ||| |||  |||   |||   |||                         |||
# |||                         ||||| ||| |||  |||   |||   |||                         |||
# |||                         ||| ||||| ||| |||   |||    |||                         |||
# |||                         |||  |||| |||  |||   |||   |||                         |||
# |||                         |||   ||| |||  |||   |||   |||                         |||
# |||                                         |||   |||                              |||
# |||                                           |||   |||                            |||
# |||                                                                                |||
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||""")

#############################################################################################
def importparameters(parameterfile):
    """Read an ini-structured text file and store each row in a dictionary entry that can be pass to the functions. Main function if you want to use the parameter file to control the arguments of the other functions.
    
    Args:
        parameterfile (string): name or path+name of the text file structured as ini configuration file (i.e., a list of rows with various [section] headers) containing the parameter values. Parameters not in the file, will be initialized to default values.

    Returns:
        Python dictionary with all the parameters needed to run the funtions.
        
    Raises:
        None
    """
    config=configparser.ConfigParser(allow_no_value=True,inline_comment_prefixes=('#', ';')) #initialize the import tool
    config.read(parameterfile) #read the parameters file
    parameters={} #initialize the parameters dictionary
    
    #GENERAL section
    parameters['GENERAL']='----------------- GENERAL -----------------'
    parameters['verbose']=config.getboolean('GENERAL','verbose') if config.has_option('GENERAL','verbose') else False #print the outputs option. If it is not given, set it to False
    
    #INPUT section
    parameters['INPUT']='----------------- INPUT -----------------'
    parameters['path']=config.get('INPUT','path')  if config.has_option('INPUT','path') else (os.getcwd()+'/') #path to the working directory (generally where the data are stored). If it is not given, set it to the currect directory
 
    #COSMOLOGY section (default values: https://ui.adsabs.harvard.edu/abs/2014ApJ...794..135B/abstract)
    parameters['COSMOLOGY']='----------------- COSMOLOGY -----------------'
    parameters['H0']=config.getfloat('COSMOLOGY','H0') if config.has_option('COSMOLOGY','H0') else 69.6 #Hubble parameter. If it is not given, set it to 69.6
    parameters['Omega_matter']=config.getfloat('COSMOLOGY','Omegam') if config.has_option('COSMOLOGY','Omegam') else 0.286 #Omega matter. If it is not given, set it to 0.286
    parameters['Omega_vacuum']=config.getfloat('COSMOLOGY','Omegav') if config.has_option('COSMOLOGY','Omegav') else 0.714 #Omega vacuum. If it is not given, set it to 0.714
    
    #GALAXY section
    parameters['GALAXY']='----------------- GALAXY -----------------'
    parameters['objname']=config.get('GALAXY','objname') if config.has_option('GALAXY','objname') else None #name of the object. If it is not given, set it to None
    parameters['redshift']=config.getfloat('GALAXY','redshift') if config.has_option('GALAXY','redshift') else None #redshift of the object. If it is not given, set it to None
    parameters['distance']=config.getfloat('GALAXY','distance') if config.has_option('GALAXY','distance') else None #distance of the object in Mpc. If it is not given, set it to None
    parameters['asectokpc']=config.getfloat('GALAXY','asectokpc') if config.has_option('GALAXY','asectokpc') else None #arcsec to kpc conversion. If it is not given, set it to None
    parameters['vsys']=config.getfloat('GALAXY','vsys') if config.has_option('GALAXY','vsys') else None #systemic velocity of the object in km/s. If it is not given, set it to None
    parameters['pa']=config.getfloat('GALAXY','pa') if config.has_option('GALAXY','pa') else None #position angle of the object in deg. If it is not given, set it to None
    parameters['inc']=config.getfloat('GALAXY','inc') if config.has_option('GALAXY','inc') else None #inclination angle of the object in deg. If it is not given, set it to None
    
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
    parameters['pixunits']=config.get('CUBEPAR','pixunits') if config.has_option('CUBEPAR','pixunits') else None #spatial units. If it is not given, set it to None
    parameters['specunits']=config.get('CUBEPAR','specunits') if config.has_option('CUBEPAR','specunits') else None #spectral units. If it is not given, set it to None
    parameters['fluxunits']=config.get('CUBEPAR','fluxunits') if config.has_option('CUBEPAR','fluxunits') else None #spatial units. If it is not given, set it to None
    parameters['spectralres']=config.getfloat('CUBEPAR','spectralres') if config.has_option('CUBEPAR','spectralres') else None #spectral resolution of km/s. If it is not given, set it to None
    parameters['pixelres']=config.getfloat('CUBEPAR','pixelres') if config.has_option('CUBEPAR','pixelres') else None #pixel resolution in arcsec. If it is not given, set it to None
    parameters['rms']=config.getfloat('CUBEPAR','rms') if config.has_option('CUBEPAR','rms') else None #root-mean-square value in Jy/beam. If it is not given, set it to None
    
    #BEAM section
    parameters['BEAM']='----------------- BEAM -----------------'
    parameters['bmaj']=config.getfloat('BEAM','bmaj') if config.has_option('BEAM','bmaj') else None #beam major axis in arcsec. If it is not given, set it to None
    parameters['bmin']=config.getfloat('BEAM','bmin') if config.has_option('BEAM','bmaj') else None #beam minor axis in arcsec. If it is not given, set it to None
    parameters['bpa']=config.getfloat('BEAM','bpa') if config.has_option('BEAM','bpa') else None #beam position angle in arcsec. If it is not given, set it to None
    
    #MODE section
    parameters['MODE']='----------------- MODE -----------------'
    parameters['mode']=config.get('MODE','mode') if config.has_option('MODE','mode') else None #main function operation option. If it is not given, set it to None

    #CORRECTION section
    parameters['CORRECTION']='----------------- CORRECTION -----------------'
    parameters['pbcorr']=config.getboolean('CORRECTION','pbcorr') if config.has_option('CORRECTION','pbcorr') else False #primary beam correction option. If it is not given, set it to False

    #FITS section
    parameters['FITS']='----------------- FITS -----------------'
    parameters['datacube']=config.get('FITS','datacube') if config.has_option('FITS','datacube') else None #name of the fits file of the data cube including .fits. If it is not given, set it to None
    parameters['beamcube']=config.get('FITS','beamcube') if config.has_option('FITS','beamcube') else None #name of the fits file of the beam cube including .fits. If it is not given, set it to None
    parameters['maskcube']=config.get('FITS','maskcube') if config.has_option('FITS','maskcube') else None #name of the fits file of the mask cube including .fits. If it is not given, set it to None
    parameters['mask2d']=config.get('FITS','mask2d') if config.has_option('FITS','mask2d') else None #name of the fits file of the 2D mask including .fits. If it is not given, set it to None
    parameters['channelmap']=config.get('FITS','channelmap') if config.has_option('FITS','channelmap') else None #name of the fits file of the channel map including .fits. If it is not given, set it to None
    parameters['modelcube']=config.get('FITS','modelcube') if config.has_option('FITS','modelcube') else None #name of the fits file of the model cube including .fits. If it is not given, set it to None
    parameters['mom0map']=config.get('FITS','mom0map') if config.has_option('FITS','mom0map') else None #name of the fits file of the moment 0 map including .fits. If it is not given, set it to None
    parameters['mom1map']=config.get('FITS','mom1map') if config.has_option('FITS','mom1map') else None #name of the fits file of the moment 1 map including .fits. If it is not given, set it to None
    parameters['mom2map']=config.get('FITS','mom2map') if config.has_option('FITS','mom2map') else None #name of the fits file of the moment 2 map including .fits. If it is not given, set it to None
    parameters['vfield']=config.get('FITS','vfield') if config.has_option('FITS','vfield') else None #name of the fits file of the velocity field including .fits. If it is not given, set it to None
    #CHANMAP
    parameters['CHANMAP']='----------------- CHANMAP -----------------'
    parameters['from_chan']=config.getint('CHANMAP','chanmin') if config.has_option('CHANMAP','chanmin') else 0 #starting channel to plot in the channel map. If it is not given, set it to 0
    parameters['to_chan']=config.getint('CHANMAP','chanmax') if config.has_option('CHANMAP','chanmax') else None #ending channel to plot in the channel map. If it is not given, set it to None
    parameters['chansep']=config.getint('CHANMAP','chansep') if config.has_option('CHANMAP','chansep') else 1 #channel separation to plot in the channel map (chamin,chanmin+chansep,chanmin+2*chansep,...,chanmax). If it is not given, set it to 1
    parameters['chanbox']=config.get('CHANMAP','box').split(',') if config.has_option('CHANMAP','box') else None #comma-separated pixel edges of the box to extract the channel mapS in the format [xmin,xmax,ymin,ymax]. If it is not given, set it to None
    if parameters['chanbox'] != [''] and parameters['chanbox'] is not None: #if the box is given
        parameters['chanbox']=[int(i) for i in parameters['chanbox']] #convert string to float
    parameters['chansig']=config.getfloat('CHANMAP','nsigma') if config.has_option('CHANMAP','nsigma') else 3 #rms threshold to plot the contours (lowest contours will be nsigma*rms). If it is not given, set it to 3
    parameters['chanmask']=config.getboolean('CHANMAP','usemask') if config.has_option('CHANMAP','usemask') else False #use a mask in the channel map [True,False]. If it is not given, set it to False
    parameters['chanmapoutdir']=config.get('CHANMAP','outputdir') if config.has_option('CHANMAP','outputdir') else None #output directory to save the channel map plot. If it is not given, set it to None
    parameters['chanmapoutname']=config.get('CHANMAP','outname') if config.has_option('CHANMAP','outname') else None #output name of the channel map plot (including file extension *.jpg,*.png,*.pdf,...). If it is not given, set it to None
     
    #CUBEDO section
    parameters['CUBEDO']='----------------- CUBEDO -----------------'
    parameters['cubedo']=config.get('CUBEDO','datacube') if config.has_option('CUBEDO','datacube') else None #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['operation']=config.get('CUBEDO','operation') if config.has_option('CUBEDO','operation') else None #operation to perform on the cube [blank,clip,crop,cut,extend,mirror,mom0,shuffle,toint]. If it is not given, set it to None
    parameters['chanmin']=config.getint('CUBEDO','chanmin') if config.has_option('CUBEDO','chanmin') else 0 #first channel for the operations 'blank,cut,mom0'. If it is not given, set it to 0
    parameters['chanmax']=config.getint('CUBEDO','chanmax') if config.has_option('CUBEDO','chanmax') else None #last channel for the operations 'blank,cut,mom0'. If it is not given, set it to None
    parameters['inbox']=config.get('CUBEDO','box').split(',') if config.has_option('CUBEDO','box') else None #comma-separated pixel edges of the box to extract for operation 'cut' in the format [xmin,xmax,ymin,ymax]. If it is not given, set it to None
    if parameters['inbox'] != [''] and parameters['inbox'] is not None: #if the box is given
        parameters['inbox']=[int(i) for i in parameters['inbox']] #convert string to float
    parameters['addchan']=config.getint('CUBEDO','addchan') if config.has_option('CUBEDO','addchan') else None #number of channels to add in operation 'extend'. Negative values add lower channels, positive add higher channels. If it is not given, set it to None
    parameters['value']=config.get('CUBEDO','value') if config.has_option('CUBEDO','value') else 'blank' #value to assign to blank pixel in operation 'blank' (blank is np.nan). If it is not given, set it to 'blank'
    if parameters['value'] != 'blank': #if the value is not blank
        parameters['value']=float(parameters['value']) #convert it to float
    parameters['withmask']=config.getboolean('CUBEDO','usemask') if config.has_option('CUBEDO','usemask') else False #use a 2D mask in the operation 'clip' [True,False]. If it is not given, set it to False
    parameters['cubedomask']=config.get('CUBEDO','mask') if config.has_option('CUBEDO','mask') else None #name of the fits file of the 2D mask including .fits. If empty, is the same of [FITS] mask2d. If it is not given, set it to None
    parameters['cliplevel']=config.getfloat('CUBEDO','cliplevel') if config.has_option('CUBEDO','cliplevel') else 0.5 #clip threshold as % of the peak (0.5 is 50%) for operation 'clip'. If it is not given, set it to 0.5
    parameters['xrot']=config.getfloat('CUBEDO','xrot') if config.has_option('CUBEDO','xrot') else None #x-coordinate of the rotational center for operation 'mirror'. If it is not given, set it to None
    parameters['yrot']=config.getfloat('CUBEDO','yrot') if config.has_option('CUBEDO','yrot') else None #y-coordinate of the rotational center for operation 'mirror'. If it is not given, set it to None
    parameters['zrot']=config.getint('CUBEDO','zrot') if config.has_option('CUBEDO','zrot') else None #z-coordinate of the rotational center for operation 'mirror'. If it is not given, set it to None
    parameters['cubedooutdir']=config.get('CUBEDO','outputdir') if config.has_option('CUBEDO','outputdir') else None #output directory to save the new cube. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['cubedooutname']=config.get('CUBEDO','outname') if config.has_option('CUBEDO','outname') else None #output name of the new cube including extension .fits. If empty, is the same of datacube. If it is not given, set it to None
        
    #CUBESTAT section 
    parameters['CUBESTAT']='----------------- CUBESTAT -----------------'
    parameters['nsigma']=config.getfloat('CUBESTAT','nsigma') if config.has_option('CUBESTAT','nsigma') else 3 #rms threshold in terms of nsigma*rms for detection limit. If it is not given, set it to 3
    
    #FITSARITH section
    parameters['FITSARITH']='----------------- FITSARITH -----------------'
    parameters['fits1']=config.get('FITSARITH','fits1') if config.has_option('FITSARITH','fits1') else None #name of reference fits file including .fits. If it is not given, set it to None
    parameters['fits2']=config.get('FITSARITH','fits2') if config.has_option('FITSARITH','fits2') else None #name of second fits file including .fits. If it is not given, set it to None
    parameters['fitsoperation']=config.get('FITSARITH','operation') if config.has_option('FITSARITH','operation') else None #operation to do between the two fits [sum,sub,mul,div]. If it is not given, set it to None
    parameters['fitsarithoutdir']=config.get('FITSARITH','outputdir') if config.has_option('FITSARITH','outputdir') else None #output directory to save the new cube. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['fitsarithoutname']=config.get('FITSARITH','outname') if config.has_option('FITSARITH','outname') else None #output name of the new fits file including extension .fits. If it is not given, set it to None

    #FIXMASK section
    parameters['FIXMASK']='----------------- FIXMASK -----------------'
    parameters['refcube']=config.get('FIXMASK','datacube') if config.has_option('FIXMASK','datacube') else None #name of the fits file of the reference data cube including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['masktofix']=config.get('FIXMASK','maskcube') if config.has_option('FIXMASK','maskcube') else None #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube. If it is not given, set it to None
    parameters['fixmaskoutdir']=config.get('FIXMASK','outputdir') if config.has_option('FIXMASK','outputdir') else None #output directory to save the new mask. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['fixmaskoutname']=config.get('FIXMASK','outname') if config.has_option('FIXMASK','outname') else None #output name of the new mask including extension .fits. If empty, is the same of maskcube. If it is not given, set it to None
    
    #GAUSSFIT section
    parameters['GAUSSFIT']='----------------- GAUSSFIT -----------------'
    parameters['cubetofit']=config.get('GAUSSFIT','datacube') if config.has_option('GAUSSFIT','datacube') else None #name of the fits file of the data cube to fit including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['gaussmask']=config.get('GAUSSFIT','gaussmask') if config.has_option('GAUSSFIT','gaussmask') else None #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube. The fit will be done inside the mask. If it is not given, set it to None
    parameters['linefwhm']=config.getfloat('GAUSSFIT','linefwhm') if config.has_option('GAUSSFIT','linefwhm') else 15 #first guess on the fwhm of the line profile in km/s. If it is not given, set it to 15
    parameters['amp_thresh']=config.getfloat('GAUSSFIT','amp_thresh') if config.has_option('GAUSSFIT','amp_thresh') else 0 #amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum. If it is not given, set it to 0
    parameters['p_reject']=config.getfloat('GAUSSFIT','p_reject') if config.has_option('GAUSSFIT','p_reject') else 1 #p-value threshold for fit rejection. If a best-fit as p>p_reject, it will be rejected . If it is not given, set it to 1
    parameters['clipping']=config.getboolean('GAUSSFIT','clipping') if config.has_option('GAUSSFIT','clipping') else False #clip the spectrum to a % of the profile peak [True,False]. If it is not given, set it to False
    parameters['clipthreshold']=config.getfloat('GAUSSFIT','threshold') if config.has_option('GAUSSFIT','threshold') else 0.5 #clip threshold as % of the peak (0.5 is 50%) if clipping is True. If it is not given, set it to 0.5
    parameters['errors']=config.getboolean('GAUSSFIT','errors') if config.has_option('GAUSSFIT','errors') else False #compute the errors on the best-fit [True,False]. If it is not given, set it to False
    parameters['write_field']=config.getboolean('GAUSSFIT','write_field') if config.has_option('GAUSSFIT','write_field') else False #compute the best-fit velocity field [True,False]. If it is not given, set it to False
    parameters['gaussoutdir']=config.get('GAUSSFIT','outputdir') if config.has_option('GAUSSFIT','outputdir') else None #output directory to save the model cube. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['gaussoutname']=config.get('GAUSSFIT','outname') if config.has_option('GAUSSFIT','outname') else None #output name of the model cube including extension .fits. If it is not given, set it to None

    #GETPV section
    parameters['GETPV']='----------------- GETPV -----------------'
    parameters['pvcube']=config.get('GETPV','datacube') if config.has_option('GETPV','datacube') else None #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['pvwidth']=config.getfloat('GETPV','width') if config.has_option('GETPV','width') else None #width of the slice in arcsec. If not given, will be the beam size. If it is not given, set it to None
    parameters['pvpoints']=config.get('GETPV','points').split(',') if config.has_option('GETPV','points') else None #RA-DEC comma-separated coordinate of the slice points in decimal degree. If two are given ([x,y]), it is assumed they are the center of the slice. Else, they need to be the starting and ending coordinates ([xmin,xmax,ymin,ymax]). If it is not given, set it to None
    if parameters['pvpoints'] != [''] and parameters['pvpoints'] is not None: #if the pv points are given
        parameters['pvpoints']=[float(i) for i in parameters['pvpoints']] #convert string to float
    parameters['pvangle']=config.getfloat('GETPV','angle') if config.has_option('GETPV','angle') else None #position angle of the slice in degree. If not given, will be the object position angle. If it is not given, set it to None
    parameters['pvchmin']=config.getint('GETPV','chanmin') if config.has_option('GETPV','chanmin') else 0 #starting channel of the slice. If it is not given, set it to 0
    parameters['pvchmax']=config.getint('GETPV','chanmax') if config.has_option('GETPV','chanmax') else None #ending channel of the slice. If it is not given, set it to None
    parameters['pvoutdir']=config.get('GETPV','outputdir') if config.has_option('GETPV','outputdir') else None #output directory to save the fits and the plot. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['fitsoutname']=config.get('GETPV','fitsoutname') if config.has_option('GETPV','fitsoutname') else None #output name of the pv fits file including extension .fits. If it is not given, set it to None
    parameters['plotoutname']=config.get('GETPV','plotoutname') if config.has_option('GETPV','plotoutname') else None #output name of the pv plot (including file extension *.jpg,*.png,*.pdf,...). If it is not given, set it to None
    parameters['subtitle']=config.get('GETPV','subtitle') if config.has_option('GETPV','subtitle') else None #optional subtitle for the pv plot. If it is not given, set it to None
    parameters['pvsig']=config.getfloat('GETPV','nsigma') if config.has_option('GETPV','nsigma') else 3 #rms threshold to plot the contours (lowest contours will be nsigma*rms). If it is not given, set it to 3
            
    #PLOTMOM section
    parameters['PLOTMOM']='----------------- PLOTMOM -----------------'
    parameters['subtitle']=config.get('PLOTMOM','subtitle') if config.has_option('PLOTMOM','subtitle') else None #optional subtitle for the plot. If it is not given, set it to None
    parameters['plotmomoutdir']=config.get('PLOTMOM','outputdir') if config.has_option('PLOTMOM','outputdir') else None #output directory to save the plot. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['plotmomoutname']=config.get('PLOTMOM','outname') if config.has_option('PLOTMOM','outname') else None #output name of the plot (including file extension *.jpg,*.png,*.pdf,...). If it is not given, set it to None
            
    #REMOVEMOD section
    parameters['REMOVEMOD']='----------------- REMOVEMOD -----------------'
    parameters['method']=config.get('REMOVEMOD','method') if config.has_option('REMOVEMOD','method') else 'subtraction' #method to remove the model [all,blanking,b+s,negblank,subtraction]. If it is not given, set it to 'subtraction'
    parameters['blankthreshold']=config.getfloat('REMOVEMOD','threshold') if config.has_option('REMOVEMOD','threshold') else 0 #flux threshold for the 'all,blanking,b+s' methods in units of cube flux. If it is not given, set it to 0
    parameters['removemodoutdir']=config.get('REMOVEMOD','outputdir') if config.has_option('REMOVEMOD','outputdir') else None #output directory to save the fits file. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['removemodoutname']=config.get('REMOVEMOD','outname') if config.has_option('REMOVEMOD','outname') else None #output name of the new fits file including extension .fits. If it is not given, set it to None

    #ROTCURVE section
    parameters['ROTCURVE']='----------------- ROTCURVE -----------------'
    parameters['rotcenter']=config.get('ROTCURVE','center').split(',') if config.has_option('ROTCURVE','center') else None #x-y comma-separated coordinates of the rotational center in pixel. If it is not given, set it to None
    if parameters['rotcenter'] is not None:
        parameters['rotcenter']=[float(i) for i in parameters['rotcenter']] #convert string to float
    parameters['save_csv']=config.getboolean('ROTCURVE','save_csv') if config.has_option('ROTCURVE','save_csv') else False #store the output in a csv file [True,False]. If it is not given, set it to False
    parameters['rotcurveoutdir']=config.get('ROTCURVE','outputdir') if config.has_option('ROTCURVE','outputdir') else None #output directory to save the plot. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['rotcurveoutname']=config.get('ROTCURVE','outname') if config.has_option('ROTCURVE','outname') else None #output name of the plot (including file extension *.jpg,*.png,*.pdf,...). If it is not given, set it to None

    #STACKING section
    parameters['STACKING']='----------------- STACKING -----------------'
    parameters['stackcenter']=config.get('STACKING','center').split(',') if config.has_option('STACKING','center') else None #x-y comma-separated coordinates of the galactic center in pixel. If it is not given, set it to None
    if parameters['stackcenter'] is not None:
        parameters['stackcenter']=[float(i) for i in parameters['stackcenter']] #convert string to float
    parameters['ncones']=config.getint('STACKING','ncones') if config.has_option('STACKING','ncones') else None #number of conic regions from which the spectra are extracted and stacked. If it is not given, set it to None
    parameters['from_to']=config.get('STACKING','radii').split(',') if config.has_option('STACKING','radii') else None #comma-separated min and max radius from the galactic center in pixunits from which the spectra are extracted and stacked. If it is not given, set it to None'
    if parameters['from_to'] is not None:
        parameters['from_to']=[float(i) for i in parameters['from_to']] #convert string to float
    parameters['rms_threshold']=config.getfloat('STACKING','threshold') if config.has_option('STACKING','threshold') else 3 #number of rms to reject flux values in the source finder. If it is not given, set it to None
    parameters['smooth_kernel']=config.get('STACKING','smooth_ker').split(',') if config.has_option('STACKING','smooth_ker') else None #kernel size (or comma-separated kernel sizes) in odd number of channels for spectral smoothing prior the source finding. Set to None or 1 to disable. If it is not given, set it to None
    if parameters['smooth_kernel'] is not None:
        parameters['smooth_kernel']=[int(i) for i in parameters['smooth_kernel']] #convert string to int
        if len(parameters['smooth_kernel'])==1: #if only 1 kernel is provided
            parameters['smooth_kernel']=parameters['smooth_kernel'][0] #convert to single float
    parameters['link_kernel']=config.getint('STACKING','link_ker') if config.has_option('STACKING','link_ker') else 3 #minimum odd number of channels covered by a spectral line. If it is not given, set it to 3
    parameters['snrmin']=config.getfloat('STACKING','min_snr') if config.has_option('STACKING','min_snr') else 3 #minimum SNR of a detected line to be reliable. If it is not given, set it to 3
    parameters['rel_threshold']=config.getfloat('STACKING','min_rel') if config.has_option('STACKING','min_rel') else 0.9 #minimum value (from 0 to 1) of the reliability to consider a source reliable. Set to 0 to disable the reliability calculation. If it is not given, set it to 0.9
    parameters['emission_width']=config.getfloat('STACKING','gal_range') if config.has_option('STACKING','gal_range') else 200000 #velocity range in specunits to exclude from rms calculation in the stacked spectra. Set to 0 to compute the rms over the whole spectrum. If it is not given, set it to None
    parameters['ref_spectrum']=config.get('STACKING','ref_spec') if config.has_option('STACKING','ref_spec') else None #path to the csv file of the reference spectrum. The first column is velocity in specunits, the second column is the flux in fluxunits. If it is not given, set it to None
    parameters['ref_rms']=config.get('STACKING','ref_rms') if config.has_option('STACKING','ref_rms') else None #path to the csv file of the reference rms as a function of the number of stacked spectra. The firs column is the rms in fluxunits. The number of rows is the number of stacked spectra. If it is not given, set it to None
    parameters['regrid']=config.getboolean('STACKING','regrid') if config.has_option('STACKING','regrid') else False #regrid the cube option. If it is not given, set it to False
    parameters['regrid_size']=config.getint('STACKING','regrid_size') if config.has_option('STACKING','regrid_size') else 3 #how many pixel to regrid. Set to 0 to regrid to a beam. If it is not given, set it to 3
    parameters['stackoutdir']=config.get('STACKING','outputdir') if config.has_option('STACKING','outputdir') else None #output directory to save the plot. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['stackoutname']=config.get('STACKING','outname') if config.has_option('STACKING','outname') else None #output name of the plot (including file extension *.jpg,*.png,*.pdf,...). If it is not given, set it to None

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
    parameters['extend_only']=config.getboolean('VELFI','extend_only') if config.has_option('VELFI','extend_only') else False #extend a given velocity field (True) or write a new one from scratch (False). If it is not given, set it to False
    parameters['correct']=config.getboolean('VELFI','correct') if config.has_option('VELFI','correct') else False #correct the input rotation velocities for the inclination angles (True) or not (False). If it is not given, set it to False
    parameters['velfioutdir']=config.get('VELFI','outputdir') if config.has_option('VELFI','outputdir') else None #output directory to save the fits file. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['velfioutname']=config.get('VELFI','outname') if config.has_option('VELFI','outname') else None #output name of the fits file including .fits. If it is not given, set it to None
    
    #PLOTSTYLE section
    parameters['PLOTSTYLE']='----------------- PLOTSTYLE -----------------'
    parameters['ctr_width']=config.getfloat('PLOTSTYLE','ctr width') if config.has_option('PLOTSTYLE','ctr width') else 2 #width of the contours levels. If it is not given, set it to 2
    parameters['plot_format']=config.getfloat('PLOTSTYLE','format') if config.has_option('PLOTSTYLE','format') else 'pdf' #file format for the plots. If it is not given, set it to pdf
            
    return parameters

#############################################################################################
def chanmap(datacube='',from_chan=0,to_chan=None,chansep=1,chanmask=False,chanmapoutdir='',chanmapoutname='',save=False,**kwargs):
    """Plot the channel maps of a data cube in a defined channel range every given number of channels.

    Args:
        datacube (string/ndarray): name or path+name of the fits data cube
        from_chan (int): first channel to be plotted in the channel map
        to_chan (int): last channel to be plotted in the channel map
        chansep (int): channel separation in the channel map. The channels plotted are from from_chan to to_chan each chansep
        chanmask (bool): option to use a detection mask (True) or not (False) to highlight the 'real' emission in the channel map. If it is True, a 3D mask must be provided using the 'maskcube' kwarg (see kwargs arguments below)
        chanmapoutdir (string): the output folder name
        chanmapoutname (string): output file name
        save (bool): option to save the plot (True) or not (False)

    Kwargs:
        path (string): path to the data cube if the datacube is a name and not a path+name
        maskcube (string/ndarray): name or path+name of the fits 3D mask cube. Used if chanmask=True
        pixunits (string): string with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - armin
            - arcsec
        specunits (string): string with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        pixelres (float): cube spatial resolution in pixunits (Default: None)
        spectralres (float): cube spectral resolution in specunits (Default: None)
        bmaj (float): beam major axis in arcsec in pixunits (Default: None)
        bmin (float): beam minor axis in arcsec in pixunits (Default: None)
        bpa (float): beam position angle in degree (Default: None)
        rms (float): rms of the data cube as a float (Default: None)
        chanbox (array-like): spatial box as [xmin,xmax,ymin,ymax] (Default: None)                     
        vsys (float): object systemic velocity in m/s (Default: 0)
        asectokpc (float): arcsec to kpc conversion to plot the spatial scale (Default: None)
        objname (string): name of the object (Default: '')
        contours (array-like): contour levels in units of rms. They will replace the default levels (Default: None)                                    
        chansig (float): lowest contour level in terms of chansig*rms (Default: 3)
        ctr_width (float): line width of the contours (Default: 2)
        plot_format (string): file format of the plots
        verbose (bool): option to print messages and plot to terminal if True (Default: None)

    Returns:
        None
    
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no path is provided
        ValueError: If no mask cube is provided when chanmask is True
        ValueError: If wrong spatial units are provided
        ValueError: If wrong spectral units are provided
        ValueError: If no spectral information is available through the input arguments or the cube header
        ValueError: If mask cube and data cube dimensions do not match
        ValueError: If size of spatial box is not 4
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    format='.'+kwargs['plot_format'] if 'plot_format' in kwargs else '.pdf'
    datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    if chanmask: #if a mask cube is used
        maskcube=kwargs['maskcube'] if 'maskcube' in kwargs else None
        if maskcube is None: #if no masckcube is provided
            chanmask=False #set to False the switch to use the mask
            if verbose:
                warnings.warn('You chose to use a mask but no mask cube is provided. No mask will be load.')
        else:
            maskcube=__read_string(maskcube,'maskcube',**kwargs)  #store the path to the data cube
    outdir=chanmapoutdir #store the output directory from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder  
    outname=chanmapoutname #store the output name from the input parameters
    if outname == '' or outname is None:  #if the outname is empty
        outname=datacube.replace('.fits','_chanmap'+format)  #the outname is the object name plus chanmap.pdf
    if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname+format
    #CHECK THE KWARGS#
    pixunits=kwargs['pixunits'] if 'pixunits' in kwargs else None
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    specunits=kwargs['specunits'] if 'specunits' in kwargs else None
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')   
    pixelres=kwargs['pixelres'] if 'pixelres' in kwargs else None
    spectralres=kwargs['spectralres'] if 'spectralres' in kwargs else None
    bmaj=kwargs['bmaj'] if 'bmaj' in kwargs else None
    bmin=kwargs['bmin'] if 'bmin' in kwargs else None
    bpa=kwargs['bpa'] if 'bpa' in kwargs else None
    rms=kwargs['rms'] if 'rms' in kwargs else None
    chanbox=kwargs['chanbox'] if 'chanbox' in kwargs else None
    if 'vsys' in kwargs: #if the systemic velocity is in the input kwargs
        if kwargs['vsys'] is not None: #if it is also not None
            vsys=kwargs['vsys'] #store the systemic velocity from the input kwargs
        else:
            vsys=0 #set it to 0
            warnings.warn('No systemic velocity is given: set it to 0 m/s!')
    else:
        vsys=0 #set it to 0
        warnings.warn('No systemic velocity is given: set it to 0 m/s!')
    asectokpc=kwargs['asectokpc'] if 'asectokpc' in kwargs else None
    objname=kwargs['objname'] if 'objname' in kwargs else None
    if objname is None:
        objname='' 
    contours=kwargs['contours'] if 'contours' in kwargs else None
    chansig=kwargs['chansig'] if 'chansig' in kwargs else 3
    ctr_width=kwargs['ctr_width'] if 'ctr_width' in kwargs else 2

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
            warnings.warn('I am missing some information: {}. Running cubestat to retrieve them!'.format(not_found))
        stats=cubestat(datacube,pixunits=pixunits,specunits=specunits,fluxunits=fluxunits,pixelres=pixelres,spectralres=spectralres,bmaj=bmaj,bmin=bmin,bpa=bpa,rms=rms,nsigma=nsigma,verbose=False) #calculate the statistics of the cube
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
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if still one or more cube parameters are not given
        if verbose:
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            warnings.warn('I am still missing some information: {}. I cannot display the information on the plot!'.format(not_found))   
    #------------   CONVERT THE VALUES IN STANDARD UNITS   ------------#
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
    if 'CRVAL3' in header: #if the header has the starting spectral value
        v0=header['CRVAL3'] #store the starting spectral value
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
            raise ValueError('ERROR: mask cube and data cube has different shapes:\n'
                                '({}, {}, {}) the mask cube and ({}, {}, {}) the data cube. Aborting!'.format(mask.shape[0],mask.shape[1],mask.shape[2],data.shape[0],data.shape[1],data.shape[2]))
        mask[mask>1]=1 #convert the mask into a 0/1 array
            
    #PREPARE THE CHANNELS VECTOR#  
    #WE CHECK THE CHANNELS#
    if from_chan is None or from_chan < 0: #if not given or less than 0
        warnings.warn('Starting channel wrongly set. You give {} but should be at least 0. Set to 0'.format(from_chan))
        from_chan=0 #set it to 0
    if to_chan is None: #if the upper channel is not set
        to_chan=data.shape[0]+1 #blank until the last channel
    elif to_chan > data.shape[0]: #if the higher channel is larger than the size of the data
        warnings.warn('Use choose a too high last channel ({}) but the cube has {} channels. Last channel set to {}.'.format(to_chan,data.shape[0],data.shape[0]))
        to_chan=data.shape[0]+1 #blank until the last channel
    elif to_chan < from_chan: #if the higher channel is less than the lower
        warnings.warn('Last channel ({}) lower than starting channel ({}). Last channel set to {}.'.format(to_chan,from_chan,data.shape[0]))
        to_chan=data.shape[0]+1 #blank until the last channel
    chans=np.arange(from_chan,to_chan,chansep) #channels to be used in channel map
    
    #CHECK THE SPATIAL BOX#
    xmin,xmax,ymin,ymax=__load_box(data,chanbox) #load the spatial box

    if verbose:
        print('The channel map will be plotted with the following parameters:\n'
        'Spectral resolution: {:.1f} km/s\n'
        'Spatial resolution: {:.1f} arcsec\n'
        'Starting velocity: {:.1f} km/s\n'
        'Systemic velocity: {:.1f} km/s\n'
        'Beam: {:.1f} x {:.1f} arcsec\n'
        'from pixel ({},{}) to pixel ({},{})\n'
        'from channel {} to channel {} every {} channels\n'
        '-------------------------------------------------'.format(spectralres,pixelres,v0-vsys,vsys,bmaj,bmin,xmin,ymin,xmax,ymax,from_chan,to_chan,chansep))
        
    #SETUP THE FIGURE#    
    nrows=int(np.floor(np.sqrt(len(chans)))) #number of rows in the channel map
    ncols=int(np.floor(np.sqrt(len(chans)))) #number of columns in the channel map
    fig=plt.figure(figsize=(4*ncols,4*nrows)) #create the figure
    fig.suptitle('{} channel maps'.format(objname),fontsize=24) #add the title
    k=0 #index to run over the channels
    
    #SETUP THE SUBPLOT PROPERTIES#    
    #-------------   LIMITS  -------------#
    leftmargin=0.075 #left margin for annotations position
    bottommargin=0.05 #bottom margin for annotations position
    topmargin=0.925 #top margin for annotations position
    if chanbox is None: #if no box is applied
        xlim=[np.min(np.where(~np.isnan(data))[2])+5,np.max(np.where(~np.isnan(data))[2])+5] #set the xlim to the first and last not-nan values
        ylim=[np.min(np.where(~np.isnan(data))[1])-30,np.max(np.where(~np.isnan(data))[1])-5] #set the ylim to the first and last not-nan values
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
    if contours is None: #if contours levels are not given:
        maxsig=round(np.log(np.sqrt(vmax)/rms)/np.log(chansig)) #calculate the max sigma in the data
        ctr=np.power(chansig,np.arange(1,maxsig,2)) #contours level in units of nsigma^i (i from 1 to max sigma in step of 2)
        if len(ctr) < 5: #if not enough contours are produced
            ctr=np.power(chansig,np.arange(1,maxsig,1)) #contours level in units of nsigma^i (i from 1 to max sigma in step of 1)
        if len(ctr) < 5: #if still not enough contours are produced
            ctr=np.power(chansig,np.arange(1,maxsig,0.5)) #contours level in units of nsigma^i (i from 1 to max sigma in step of 0.5)
    else:
        ctr=contours #use the levels given in input
    if verbose:
        ctrtoHI=converttoHI(ctr*rms,beamarea=beamarea,pixunits='arcsec',spectralres=spectralres,specunits='km/s')
        print('Contours level: {} cm⁻²'.format([ '%.1e' % elem for elem in ctrtoHI]))
        
    #DO THE PLOT#
    for k in tqdm(range((nrows*ncols)),desc='Channels plotted'):
        fig.add_subplot(nrows,ncols,k+1,projection=wcs) #create the subplot
        ax=plt.gca() #get the current axes
        chanmap=data[chans[k],ymin:ymax,xmin:xmax] #select the channel
        im=ax.imshow(chanmap,cmap='Greys',norm=norm,aspect='equal') #plot the channel map in units of detection limit
        if chanmask: #if a mask must be used
            ax.contour(chanmap/rms,levels=ctr,cmap='hot',linewidths=ctr_width/2,linestyles='solid') #add the contours
            ax.contour(chanmap/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width/2,linestyles='dashed') #add the negative contours
            ax.contour(chanmap*mask[chans[k],ymin:ymax,xmin:xmax]/rms,levels=ctr,cmap='hot',linewidths=ctr_width,linestyles='solid') #add the contours within the mask
        else:
            ax.contour(chanmap/rms,levels=ctr,cmap='hot',linewidths=ctr_width,linestyles='solid') #add the contours
            ax.contour(chanmap/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width,linestyles='dashed') #add the negative contours
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if k not in np.arange(0,nrows*ncols,ncols): #if not plotting the first column
            ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
        if k not in np.arange((nrows-1)*ncols,nrows*ncols): #if not plotting the last row
            ax.coords[0].set_ticklabel_visible(False) #hide the x-axis ticklabels and labels
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.text(leftmargin,bottommargin,'V$_{{rad}}$: {:.1f} km/s'.format(chans[k]*spectralres+v0-vsys),transform=ax.transAxes) #add the information of the channel velocity
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            if k in np.arange(0,nrows*ncols,nrows): #in the first column
                ax=__plot_kpcline(pixelres,asectokpc,xlim,leftmargin,topmargin) #draw the 10-kpc line
            
    fig.subplots_adjust(top=0.95,wspace=0.0,hspace=0.0) #fix the position of the subplots in the figure
    if save: #if the save switch is true
        fig.savefig(outname,dpi=300,bbox_inches='tight') #save the figure

    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    else:
        plt.close()
    #fig.clf() #clear the figure from the memory
    
#############################################################################################
def cubedo(cubedo='',operation=None,chanmin=0,chanmax=None,inbox=None,addchan=None,value='blank',withmask=False,cubedomask=None,cliplevel=0.5,xrot=None,yrot=None,zrot=None,write_fits=False,cubedooutdir='',cubedooutname='',**kwargs):
    """Perform the selected operation between blank, clip, crop, cut, extend, mirror, mom0, shuffle, toint on a fits data cube.

    Args:
        cubedo (string): name or path+name of the fits data cube
        operation (string): operation to perform. Accepted values:
            blank, blank all the data in the chanmin-chanmax range
            clip, blank all the data below the cliplevel value
            crop, crop the data cube to remove the blanked edges
            cut, extract the subcube from chanmin to chanmax
            extend, add channels to the spectral axis
            mirror, mirror the data around a rotation point (x,y,z)
            mom0, compute the moment 0 map from chanmin to chanmax
            shuffle, align the spectral profile with the galaxy rotation
            toint, convert the data cube into an integer cube
        chanmin (int): first channel for operations blank, cut, mom0
        chanmax (int): last channel for operations blank, cut, mom0
        inbox (array-like): spatial cut box as [xmin,xmax,ymin,ymax]
        addchan (int): number of channels to add in operation extend. If < 0 the channels are added at the beginning of the spectral axis, else at the end
        value (float/string): value to give to the blanked pixel in operation blank. If string 'blank' it will be np.nan
        withmask (bool): option to use a detection mask (True) or not (False) for clip and mom0. If it is True, a 2D mask for clip or 3D mask for mom0 must be provided using the cubedomask  argument or the 'mask2d' and 'maskcube' kwarg (see kwargs arguments below)
        cubedomask (string): name or path+name of the fits 2D or 3D mask. Used if withmask=True
        cliplevel (float): clip threshold as % of the peak (0.5 is 50%) for clip
        xrot (float): x-coordinate of the rotational center for operation mirror in pixel
        yrot (float): y-coordinate of the rotational center for operation mirror in pixel
        zrot (int): z-coordinate of the rotational center for operation mirror in channels
        write_fits (boolean): store the output in a fits file (True) or return a variable (False)
        cubedooutdir (string): output folder name
        cubedooutname (string): output file name
        
    Kwargs:
        datacube (string/ndarray): name or path+name of the fits data cube if cubedo is not given
        path (string): path to the data cube if the datacube is a name and not a path+name
        vfield (string): name or path+name of the fits velocity field to be used for operation shuffle
        specunits (string): string with the spectral units for operation mom0 and shuffle (Default: m/s). Accepted values:
            - km/s
            - m/s
            - Hz
        spectralres (float): cube spectral resolution in specunits for operation mom0 and shuffle (Default: None)
        v0 (float): starting velocity to create the spectral axis for operation mom0 and shuffle (Default: None)
        mask2d (string): name or path+name of the fits 2D mask for operation clip. Used if withmask=True
        maskcube (string/ndarray): name or path+name of the fits 3D mask cube for operation mom0. Used if withmask=True
        verbose (bool): option to print messages to terminal if True (Default: False)
 
    Returns:
        Resulting cube or moment map as fits file
        
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no path is provided
        ValueError: If no operation is set
        ValueError: If operation set do not match accepted values
        ValueError: If no velocity field is given when 'shuffle' is set
        ValueError: If no spectral resolution is provided when 'mom0' or 'shuffle' is set
        ValueError: If no mask is provided when usemask is True
        ValueError: If no spectral information is available through the input arguments or the cube header
        ValueError: If mask cube and data cube dimensions do not match
        ValueError: If size of spatial box is not 4
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    if type(cubedo)==str or cubedo is None: #if the datacube is a string
        datacube=__read_string(cubedo,'datacube',**kwargs) #store the path to the data cube
    else:
        datacube=cubedo
    if operation is None: #if no operation is set
        raise ValueError('ERROR: no operation set: aborting')
    if operation not in ['blank','clip','crop','cut','extend','mirror','mom0','shuffle','toint']: #if wrong operation is given
        raise ValueError("ERROR: wrong operation. Accepted values: ['blank','clip','crop','cut','extend','mirror','mom0','shuffle','toint']. Aborting")
    if operation in ['shuffle']: #if the datacube must be shuffled
        vfield=kwargs['vfield'] if 'vfield' in kwargs else None
        if vfield is None: #if no vfield is provided
            raise ValueError("Selected operation is 'shuffle', but no velocity field is provided. Aborting.")
        if type(vfield)==str: #if the velocity field is a string
            vfield=__read_string(vfield,'vfield',**kwargs)
    if operation in ['clip']:  #if the datacube must be clipped
        usemask=withmask #store the use mask switch from the input parameters
        if usemask: #if a 2D mask is used
            if type(cubedomask)==str or cubedomask is None: #if the 2D mask is a string
                mask2d=__read_string(cubedomask,'mask2d',**kwargs) #store the 2D mask from the input parameters
            else:
                mask2d=cubedomask
        threshold=cliplevel #store the clipping threshold from the input parameters
    if operation in ['mirror']: #if the cube must be mirrored
        x0=xrot #store the x-axis rotation central pixel
        y0=yrot #store the y-axis rotation central pixel
        z0=zrot #store the z-axis rotation central channel
    if operation in ['mom0']:  #if the datacube must be clipped
        specunits=kwargs['specunits'] if 'specunits' in kwargs else None #store the spectral units from the input paramaters
        spectralres=kwargs['spectralres'] if 'spectralres' in kwargs else None #store the spectral resolution from the input paramaters
        usemask=withmask #store the use mask switch from the input parameters
        if usemask: #if a 2D mask is used
            if type(cubedomask)==str or cubedomask is None: #if the 3D mask is a string
                maskcube==__read_string(cubedomask,'maskcube',**kwargs) #store the 3D mask from the input parameterse
            else:
                maskcube=cubedomask
    if operation in ['shuffle']: #if the datacube must be shuffled
        spectralres=kwargs['spectralres'] if 'spectralres' in kwargs else None #store the spectral resolution from the input paramaters
        v0=kwargs['v0'] if 'v0' in kwargs else None #store the starting velocity from the input paramaters
    if write_fits: #if the data must be written into a fits file
        outdir=cubedooutdir #store the output folder from the input parameters
        if outdir == '' or outdir is None:  #if the outdir is empty
            outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
        elif not os.path.exists(outdir): #if the output folder does not exist
            os.makedirs(outdir) #create the folder  
        outname=cubedooutname #store the outputname from the input parameters
        if outname == '' or outname is None: #if the outname is empty
            outname=datacube.replace('.fits','_'+operation+'.fits')
        if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
            outname=outdir+outname
    
    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube)
        
    #WE CHECK THE CHANNELS
    if operation in ['blank','cut','mom0']: #if the operation to be done is blank, cut or mom0
        if chanmin is None or chanmin < 0: #if not given or less than 0
            warnings.warn('Starting channel wrongly set. You give {} but should be at least 0. Set to 0'.format(chanmin))
            chanmin=0 #set it to 0
        if chanmax is None: #if the upper channel is not set
            chanmax=data.shape[0]+1 #select until the last channel
        elif chanmax > data.shape[0]: #if the higher channel is larger than the size of the data
            warnings.warn('Use choose a too high last channel ({}) but the cube has {} channels. Last channel set to {}.'.format(chanmax,data.shape[0],data.shape[0]))
            chanmax=data.shape[0]+1 #select until the last channel
        elif chanmax < chanmin: #if the higher channel is less than the lower
            warnings.warn('Last channel ({}) lower than starting channel ({}). Last channel set to {}.'.format(chanmax,chanmin,data.shape[0]))
            chanmax=data.shape[0]+1 #select until the last channel
    
    #WE CHECK THE SPATIAL BOX
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
            if mask.shape[0] != data.shape[1] or  mask.shape[1] != data.shape[2]: #if the mask cube has different size than the data cube
                raise ValueError('ERROR: mask and data cube has different spatial shapes: {} the mask and ({},{}) the data cube. Aborting!'.format(mask.shape,data.shape[1],data.shape[2]))
                x=np.where(mask>0)[0] #store the x coordinate of the non-masked pixels
                y=np.where(mask>0)[1] #store the y coordinate of the non-masked pixels
        else:
            x=np.where(~np.isnan(data))[1] #store the x coordinate of the non-masked pixels
            y=np.where(~np.isnan(data))[2] #store the y coordinate of the non-masked pixels
        for i in tqdm(zip(x,y),desc='Spectra clipped',total=len(x)): #run over the pixels
            spectrum=data[:,i[0],i[1]] #extract the spectrum
            peak=np.nanmax(spectrum) #define the peak of the gaussian as the maximum of the spectrum
            spectrum[spectrum<(peak*(1-threshold))]=0 #clip the spectrum at the threshold of the max 
            clip_cube[:,i[0],i[1]]=spectrum #store the result in the clipped cube
        data=clip_cube.copy() #copy the clipped cube into the data cube
            
    #------------   CROP     ------------#
    if operation == 'crop': #if the datacube must be cropped
        xlim=np.where(~np.isnan(data))[2] #select the extreme x coordinates of non-NaN values
        ylim=np.where(~np.isnan(data))[1] #select the extreme y coordinates of non-NaN values
        data=data[:,np.min(ylim):np.max(ylim),np.min(xlim):np.max(xlim)] #crop the data
        if header is not None: #if the header is None means the input data is not a fits file
            wcs=WCS(header) #store the wcs
            wcs=wcs[:,np.min(ylim):np.max(ylim),np.min(xlim):np.max(xlim)] #crop the wcs
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
        wcs=WCS(header) #store the wcs
        wcs=wcs[:,ymin:ymax,xmin:xmax] #crop the wcs
        newheader=wcs.to_header() #write the wcs into a header
        header['CRPIX1']=newheader['CRPIX1'] #update the header
        header['CRPIX2']=newheader['CRPIX2'] #update the header
    
    #------------   EXTEND     ------------# 
    if operation == 'extend': #if the datacube must be cutted
        if value == 'blank': #if the value is blank
            value=np.nan #set it to blank
        if header is not None: #if the header is None means the input data is not a fits file
            if 'CRPIX3' in header: #if the spectral keyword exists
                header['CRPIX3']=header['CRPIX3']+abs(addchan) #recalculate the spectral axis 
            else:
                raise ValueError('ERROR: no spectral keywords in the header. Cannot recalculate the spectral axis: aborting')
        if addchan < 0: #if the number of channels is less than 0
            for j in range(abs(addchan)):
                newplane=np.ones((1,data.shape[1],data.shape[2]))*value #create the new plane
                data=np.concatenate((newplane,data)) #concatenate the plane to the left
        else:
            for j in range(addchan):
                newplane=np.ones((1,data.shape[1],data.shape[2]))*value #create the new plane
                data=np.concatenate((data,newplane)) #concatenate the plane to the left
    
    #------------   MIRROR     ------------# 
    if operation == 'mirror': #if the datacube must be mirrored
        if x0 is None: #if no x-rotation center is given
            x0=data.shape[1]/2 #the center is the axis center
        if y0 is None: #if no x-rotation center is given
            y0=data.shape[2]/2 #the center is the axis center
        if z0 is None: #if no x-rotation center is given
            z0=data.shape[0]//2 #the center is the axis center
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
                header['BUNIT']='Jy/beam*{}'.format(specunits)
                if verbose:
                    warnings.warn('No flux unit was found: flux density unit set to Jy/beam*{}!'.format(specunits))
        else:
            if spectralres is None:
                raise ValueError('The spectral resolution is not provided. Aborting!')
        if usemask: #if a mask is used
            mask,_=__load(maskcube) #open the 3D mask
            if mask.shape[0] != data.shape[0] or mask.shape[1] != data.shape[1] or mask.shape[2] !=data.shape[2]: #if the mask cube has different size than the data cube
                raise ValueError('ERROR: mask cube and data cube has different shapes: {} the mask cube and {} the data cube. Aborting!'.format(mask.shape,cube[0].data.shape))
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
                if 'CRVAL3' in header: #if the header has the starting spectral value
                    v0=header['CRVAL3'] #store the starting spectral value
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
        if spectralres>0: #if the spectral resolution is positive
            v=np.arange(v0,v0+nchan*spectralres,spectralres) #define the spectral axis
        else:
            v=np.flip(np.arange(v0+(nchan-1)*spectralres,v0-spectralres,-spectralres)) #define the spectral axis
        x=np.where(~np.isnan(mom1))[0] #store the non-NaN x coordinates
        y=np.where(~np.isnan(mom1))[1] #store the non-NaN y coordinates
        for i in tqdm(zip(x,y),desc='Spectra shuffled',total=len(x)): #run over the pixels
            loc=np.argmin(abs(v-mom1[i[0],i[1]])) #define the spectral channel center of shuffle
            for z in range(nchan): #run over the spectral axis
                n=(loc-cen+z) % nchan #define the number of channels to be shuffled
            #    #!!!! SE VUOI CHE WRAPPING SIA OPZIONALE, QUI SOTTO GLI STEP SE wrap=False !!!!#
            #    #n=(loc-cen+z)
            #    #if (n < 0) or (n >= nchan): #if negative or grater than the number of channels
            #    #    pass #do nothing
            #    #else:
                shuffle_data[z,i[0],i[1]]=data[n,i[0],i[1]] #perform the shuffle wrapping the noise around   
            #shuffle_data[:,i[0],i[1]]=np.roll(data[:,i[0],i[1]],loc-cen,axis=0)      
        data=shuffle_data.copy() #store the shuffle data
        
    #------------   TOINT     ------------#     
    if operation == 'toint': #if the datacube must be converted into an integer cube
        data[np.isnan(data)]=0 #set to 0 the nans

    #PREPARE THE DATA FOR SAVING#
    if operation == 'toint': #if the datacube must be converted into an integer cube
        dtype=int #data type is int
    else:
        dtype='float32' #data type is float32
        
    if write_fits: #if the data must be write into a fits file
        hdu=fits.PrimaryHDU(data.astype(dtype),header=header) #create the primary HDU
        new_data=fits.HDUList([hdu]) #make the HDU list
        new_data.writeto(outname,overwrite=True) #write the data into a fits file
        
    else:
        return data.astype(dtype)
                        
#############################################################################################
def cubestat(datacube='',**kwargs):
    """Calculate the detection limit of a data cube and (optional) its rms, spectral resolution, beam major axis, beam minor axis, beam position angle and beam area. It also computes the errors on rms and sensitivity.

    Args:
        datacube (string/ndarray): string with name or path+name of the fits data cube

    Kwargs:
        path (string): path to the data cube if the datacube is a name and not a path+name
        pixunits (string): string with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - armin
            - arcsec
        specunits (string): string with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        fluxunits (string): string with the flux units (Default: None)
        pixelres (float): cube spatial resolution in pixunits (Default: None)
        spectralres (float): cube spectral resolution in specunits (Default: None)
        bmaj (float): beam major axis in arcsec in pixunits (Default: None)
        bmin (float): beam minor axis in arcsec in pixunits (Default: None)
        bpa (float): beam position angle in degree (Default: None)
        rms (float): rms of the data cube as a float in fluxunits (Default: None)
        nsigma (float): sigma of the detectin limit in terms of nsigma*rms (Default: 3)
        verbose (bool): option to print messages to terminal if True (Default: False)

    Returns:
        Python dictionary with the cube statistics
    
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no path is provided
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    #CHECK THE KWARGS#
    pixunits=kwargs['pixunits'] if 'pixunits' in kwargs else None
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    specunits=kwargs['specunits'] if 'specunits' in kwargs else None
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')   
    fluxunits=kwargs['fluxunits'] if 'fluxunits' in kwargs else None
    pixelres=kwargs['pixelres'] if 'pixelres' in kwargs else None
    spectralres=kwargs['spectralres'] if 'spectralres' in kwargs else None
    bmaj=kwargs['bmaj'] if 'bmaj' in kwargs else None
    bmin=kwargs['bmin'] if 'bmin' in kwargs else None
    bpa=kwargs['bpa'] if 'bpa' in kwargs else None
    rms=kwargs['rms'] if 'rms' in kwargs else None
    nsigma=kwargs['nsigma'] if 'nsigma' in kwargs else 3

    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #open the data cube
    #WE CHECK WHICH INFORMATION ARE ALREADY GIVEN#
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
    if bmaj is not None and bmin is not None: #if the beam axes are available
        beamarea=1.13*(bmin*bmaj) #calculate the beam area
    #------------   RMS     ------------#    
    if rms is None: #if the rms is not given
        rms=[] #initialize the rms list
        for i in np.arange(data.shape[0]): #run thorugh the spectral axis
            rms.append(np.sqrt(np.nanmean(data[i,:,:]**2))) #for each slice in the channels range calculate the rms and append the result to the rms list
        rms=sorted(rms) #order the rms in ascended order
        rms=rms[0:(round(data.shape[0]/10))] #select the lowest 10% of the rms values
        rms=np.median(rms) #calculate the median of the rms values
    
    #WE CALCULATE THE DETECTION LIMIT AND RMS ERROR#
    rms_error=rms/10 #the error on the rms is equal to the calibration error (typically of 10%)
    sensitivity=converttoHI(nsigma*rms,fluxunits=fluxunits,beamarea=beamarea,pixunits=pixunits,spectralres=spectralres,specunits=specunits) #the detection limit is the rms
    sens_error=converttoHI(nsigma*rms_error,fluxunits=fluxunits,beamarea=beamarea,pixunits=pixunits,spectralres=spectralres,specunits=specunits) #the detection limit error is the rms error
    
    #WE STORE THE RESULT IN A DICTIONARY#
    result={} #initialize the dictionary result
    result['CUBE UNITS']='--- CUBE UNITS ---'
    result['pixunits']=pixunits #store the pixel units
    result['specunits']=specunits #store the spectral units
    result['fluxunits']=fluxunits #store the flux units
    result['CUBE RESOLUTION']='--- CUBE RESOLUTION ---'
    result['pixelres']=pixelres #store the pixel resolution
    result['spectralres']=spectralres #store the spectral resolution
    result['BEAM']='--- BEAM ---'    
    result['bmaj']=bmaj #store the beam major axis
    result['bmin']=bmin #store the beam minor axis
    result['bpa']=bpa #store the beam position angle
    result['beamarea']=beamarea #store the beam area
    result['RMS']='--- RMS ---'
    result['rms']=rms #store the rms in the dictionary
    result['rms error']=rms_error #store the rms uncertainty in the dictionary
    result['sensitivity']=sensitivity #store the detection limit in the dictionary
    result['sensitivity error']=sens_error #store the detection limit uncertainty in the dictionary
    
    #WE PRINT THE INFORMATION TO THE TERMINAL IF VEROBSE#
    if verbose: #if print-to-terminal option is true
        name=datacube.split('/')[-1] #extract the cube name from the datacube variable
        coldenunits='cm\u207B\u00B2' #units of the column density
        print('Noise statistic of the cube {}:'.format(name))
        print('The median rms per channel is: {:.1e} {}'.format(rms,fluxunits))
        print('The {}\u03C3 1-channel detection limit is: {:.1e} {} i.e., {:.1e} {}'.format(int(nsigma),nsigma*rms,fluxunits,sensitivity,coldenunits))
    return result

#############################################################################################
def fitsarith(path='',fits1='',fits2='',fitsoperation=None,fitsarithoutdir='',fitsarithoutname='',**kwargs):
    """Perform a number of arithmetical operations (sum, sub, mul, div) between two fits file.

    Args:
        path (string): path to the fits file
        fits1 (string): name or path+name of one fits file
        fits2 (string): name or path+name of the other fits file
        fitsoperation (string): operation to perform between:
            sum, sum the two fits file
            sub, subtract the two fits file
            mul, multiply the two fits file
            div, divide the two fits file (the zeroes in the second fits will be blanked)
        fitsarithoutdir (string): output folder name
        fitsarithoutname (string): output file name

    Returns:
        Result of the operation as fits file
    
    Raises:
        ValueError: If one or both fits file are not set
        ValueError: If no operation is set
        ValueError: If operation set do not match accepted values
        ValueError: If no output name is given
        ValueError: If the shapes of the two fits are not matching
    """
    #CHECK THE INPUT#
    if path == '' or path is None: #if the path to the fits files is not given
        path=os.getcwd()+'/'
    indir=path #the input folder is the path
    if fits1 == '' or fits1 is None or fits2 == '' or fits2 is None:
        raise ValueError('ERROR: one or both fits file are not set: aborting')
    if fits1[0] != '.': #if the first fits name start with a . means that it is a path to the fits (so differs from path parameter)
        fits1=indir+fits1
    if fits2[0] != '.': #if the second fits name start with a . means that it is a path to the fits (so differs from path parameter)
        fits2=indir+fits2
    operation=fitsoperation #store the operation from the input parameters
    if operation is None: #if no operation is set
        raise ValueError('ERROR: no operation set: aborting')
    if operation not in ['sum','sub','mul','div']: #if wrong operation is given
        raise ValueError("ERROR: wrong operation. Accepted values: ['sum','sub','mul','div']. Aborting")
    outdir=fitsarithoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=indir #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder
    outname=fitsarithoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        raise ValueError("ERROR: no name for the output is given. Aborting")
    if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
 
    #---------------   START THE FUNCTION   ---------------#
    with fits.open(fits1) as Fits1: #open the first fits file
        data1=Fits1[0].data #store the first fits file data
        with fits.open(fits2) as Fits2: #open the second fits file
            data2=Fits2[0].data #store the second fits file data
        #CHECK THAT THE TWO DATASET HAVE SAME DIMENSIONS#
        if len(data1.shape) != len(data2.shape): #if the two fits file have different dimensions
            raise ValueError('!ERROR! The two fits files have different dimensions {} and {}. Aborting!!!'.format(data1.shape,data2.shape))
        for i in range(len(data1.shape)): #for each dimension of the fits
            if data1.shape[i] != data2.shape[i]: #if the two fits file have different shapes
                raise ValueError('!ERROR! The two fits files have different {}-dimension shapes ({} and {}). Aborting!!!'.format(i,data1.shape[i],data2.shape[i]))
        #WE DO THE SELECTED OPERATION#
        if operation == 'sum': #if the two fits file must be added
            data=data1+data2
        if operation == 'sub': #if the two fits file must be subtracted
            data=data1-data2
        if operation == 'mul': #if the two fits file must be multiplied
            data=data1*data2
        if operation == 'div': #if the two fits file must be divided
            data2[data2==0]=np.nan #blank the 0 values of second fits file to avoid divide-by-0 issues
            data=data1/data2
        Fits1[0].data=data
        Fits1.writeto(outname,overwrite=True) #write the new cube
        
#############################################################################################
def fixmask(refcube='',masktofix='',fixmaskoutdir='',fixmaskoutname='',**kwargs):
    """Fix a 3D detection mask by setting to 0 the voxel corresponding to negative detections (using a reference data cube). It is assumed that a value > 0 in the mask is marking a detection.
    
    Args:
        refcube (string/ndarray): name or path+name of the fits data cube used to fix the mask
        masktofix (string): name or path+name of the fits 3D mask to be fixed
        fixmaskoutdir (string): output folder name
        fixmaskoutname (string): output file name
        
    Kwargs:
        datacube (string/ndarray): name or path+name of the fits data cube if refcube is not given
        maskcube (string/ndarray): name or path+name of the fits 3D mask cube. Used if chanmask=True                          
        path (string): path to the data cube if the datacube is a name and not a path+name
        verbose (bool): option to print messages to terminal if True (Default: False)

    Returns:
        Fixed 3D mask cube as fits file
        
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no path is provided
        ValueError: If no mask cube is provided
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    datacube=__read_string(refcube,'datacube',**kwargs) #store the path to the data cube
    maskcube=__read_string(masktofix,'maskcube',**kwargs) #store the path to the mask cube
    outdir=fixmaskoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder
    outname=fixmaskoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty or the same as the input mask
        mode='update' #set the fits open mode to update
    else:
        mode='readonly'
        if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
            outname=outdir+outname        
        
    #---------------   START THE FUNCTION   ---------------#
    data,_=__load(datacube) #open the data cube
    with fits.open(maskcube,mode=mode) as Maskcube: #open the mask cube
        mask=Maskcube[0].data.copy() #store the mask data
        mask[data<0]=0 #fix the mask by setting to 0 the pixel where the emission is negative (hence, no source)
        Maskcube[0].data=mask #overwrite the mask data
        if mode == 'readonly': #if the open mode is read only
            Maskcube.writeto(outname,overwrite=True) #write the new mask
        
#############################################################################################
def gaussfit(cubetofit='',gaussmask='',linefwhm=15,amp_thresh=0,p_reject=1,
             clipping=False,threshold=0.5,errors=False,write_field=False,write_fits=False,gaussoutdir='',gaussoutname='',**kwargs):
    """Perform a gaussian fit on a spectral cube. Each spaxel will be fitted with a single gaussian.

    Args:
        cubetofit (string):name or path+name of the fits data cube to be fitted
        gaussmask (string): name or path+name of the fits 2D mask to be used in the fit
        linefwhm (float): first guess on the fwhm of the line profile in km/s
        amp_thresh (float): amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum
        p-reject (float): p-value threshold for fit rejection. If a best-fit has p > p-reject, it will be rejected
        clipping (bool): clip the spectrum to a % of the profile peak if True
        threshold (float): clip threshold as % of the peak (0.5 is 50%). Used if clipping is True
        errors (bool): compute the errors on the best-fit if True
        write_field (bool): compute the velocity field if True
        write_fits (boolean): store the output in a fits file (True) or return a variable (False)
        gaussoutdir (string): output folder name
        gaussoutname (string): output file name
    
    Kwargs:
        datacube (string/ndarray): name or path+name of the fits data cube if cubetofit is not given
        mask2d (string): name or path+name of the fits 2D mask if gaussmask is not given
        path (string): path to the data cube if the datacube is a name and not a path+name
        specunits (string): string with the spectral units for operation mom0 and shuffle (Default: m/s). Accepted values:
            - km/s
            - m/s
            - Hz
        spectralres (float): data spectral resolution in specunits. Will be taken from cube header if not provided.
        verbose (bool): option to print messages to terminal if True (Default: False)
 
    Returns:
        Best-fit model cube as fits file
        
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no path is provided
        ValueError: If mask cube and data cube dimensiosns do not match
        ValueError: If no spectral information is available through the input arguments or the cube header
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    if type(cubetofit)==str or cubetofit is None: #if the datacube is a string
        print(cubetofit)
        datacube=__read_string(cubetofit,'datacube',**kwargs) #store the path to the data cube
    else:
        datacube=cubetofit
    if type(gaussmask)==str or gaussmask is None: #if the maskcube is a string
        maskcube=__read_string(gaussmask,'maskcube',**kwargs) #store the path to the mask cube
    else:
        maskcube=gaussmask
    outdir=gaussoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder    
    outname=gaussoutname #store the output name from the input parameters
    if outname == '' or outname is None:  #if the outname is empty
        outname=datacube.replace('.fits','_gaussfit.fits')  #the outname is the object name plus chanmap.pdf
    elif outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    specunits=kwargs['specunits'] if 'specunits' in kwargs else 'm/s' #store the spectral units from the input paramaters
    spectralres=kwargs['spectralres'] if 'spectralres' in kwargs else None #store the spectral resolution from the input paramaters
    
    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #open the data cube
    if maskcube == '' or maskcube is None: #if no mask is given
        mask=np.ones(data.shape)
    else:
        mask,_=__load(maskcube) #open the mask
        if [mask.shape[0],mask.shape[1],mask.shape[2]] != [data.shape[0],data.shape[1],data.shape[2]]: #if the mask cube has different size than the data cube
            raise ValueError('ERROR: mask and data cube has different spatial shapes: {} the mask and ({},{}) the data cube. Aborting!'.format(mask.shape,data.shape[1],data.shape[2]))
        else:
            mask[mask>0]=1 #set the non-zero value in the mask to 1
            data=data*mask #apply the mask
    mask2d=np.nansum(mask,axis=0) #create the 2D mask
    model_cube=np.zeros(data.shape) #initialize the model cube as zeros
    if write_field: #if the velocity field whould be computed
        field=np.empty((data.shape[1],data.shape[2]))*np.nan #initialize the velocity field
        
    #WE CHECK THE REQUIRED INFORMATION#
    if spectralres is None: #if the spectral resolution is not given
        if header is not None: #if the header is None means the input data is not a fits file
            if 'CDELT3' in header: #but it is in the header
                spectralres=header['CDELT3'] #store the spectral resolution from the cube header
            else:
                raise ValueError('ERROR: no spectral resolution is provided or found. Aborting!')
        else:
            raise ValueError('ERROR: no spectral resolution is provided or found. Aborting!')
    if spectralres<0: #if the spectral resolution is negative
        np.flip(data,axis=0) #flip the cube
    
    if header is not None: #if the header is None means the input data is not a fits file    
        if 'CRVAL3' in header: #if the header has the starting spectral value
            v0=header['CRVAL3'] #store the starting spectral value
        else:
            raise ValueError('ERROR: no spectral value for starting channel was found. Aborting!')
    else:
        raise ValueError('ERROR: no spectral value for starting channel was found. Aborting!')
        
    if specunits == 'm/s': #if the spectral units are m/s
        spectralres/=1000 #convert the spectral resolution to km/s
        v0/=1000 #convert the starting velocity to km/s
        
    #WE PREPARE THE SPECTRAL AXIS#
    nchan=data.shape[0] #store the number of channels
    v=np.arange(v0,v0+nchan*spectralres,spectralres) #define the spectral axis
    if len(v) > nchan: #!! sometimes an additional channel is created. For the moment, this is a workaround
        v=v[:-1]
        
    #WE PREPARE FOR THE FIT#
    width=linefwhm/spectralres #define the first guess fwhm of the line in km/s
    x=np.where(mask2d > 0)[0] #store the x coordinate of the non-masked pixels
    y=np.where(mask2d > 0)[1] #store the y coordinate of the non-masked pixels
    
    #WE START THE FITTING ROUTINE#
    if verbose:
        print('Starting the Gaussian fit with the following parameters:\n'
        'Spectral resolution: {} km/s\n'
        'Starting velocity: {} km/s\n'
        'First-guess FWHM: {} km/s\n'
        'Amplitude threshold: {}\n'
        'p-value for rejection: {}'.format(np.abs(spectralres),v0,linefwhm,amp_thresh,p_reject))
    
    for i in tqdm(zip(x,y),desc='Spectra fitted',total=len(x)): #run over the pixels
        spectrum=data[:,i[0],i[1]].copy() #extract the spectrum
        peak=np.nanmax(spectrum) #define the peak of the gaussian as the maximum of the spectrum        
        if peak > amp_thresh: #if the peak is above the threshold
            vpeak=v0+np.nanargmax(spectrum)*spectralres #define the central velocity as the velocity of the peak
            if clipping: #if data must be clipped
                v_fit=v[np.where(spectrum>=(peak*(1-threshold)))] #extract the velocities of the clipped data
                spectrum=spectrum[np.where(spectrum>=(peak*(1-threshold)))] #clip the data
            else:
                v_fit=v[np.where(~np.isnan(spectrum))] #extract the velocities corresponding to non-nan in the spectrum
                spectrum=spectrum[np.where(~np.isnan(spectrum))] #extract the not-nan values
            model=models.Gaussian1D(amplitude=peak,mean=vpeak,stddev=width*2.355) #built the gaussian model
            model.amplitude.bounds=(0,2*peak) #bound the amplitude of the fit
            model.mean.bounds=(np.nanmin(v_fit),np.nanmax(v_fit)) #bound the velocity of the fit
            fitter=fit(calc_uncertainties=errors) #define the fitter
            if len(spectrum)>2: #fit only if you have at least 3 data points
                total_fit=fitter(model,v_fit,spectrum) #do the fit. Weights are 1/error. They are used only if calc_uncertainties=True
                v_fit=v_fit[np.where(spectrum!=0)] #extract the velocities corresponding to non-zero in the spectrum
                spectrum=spectrum[np.where(spectrum!=0)] #extract the non-zero values
                chi2=np.sum(((spectrum-total_fit(v_fit))**2)/(spectrum/10)**2) #calculate the chi2
                dof=len(spectrum-3) #calculate the degrees of freedom
                p_value=statchi2.cdf(chi2,dof) #calculate the p-value
                if p_value<=p_reject: #if the p-value is less than p
                    model_cube[:,i[0],i[1]]=total_fit(v) #store the result in the model cube
                    if write_field: #if the velocity field whould be computed
                        field[i[0],i[1]]=total_fit.mean.value #store the best-fit peak value
    
    dtype='float32' #data type is float32
    
    if write_field: #if the velocity field whould be computed
        wcs=WCS(header).dropaxis(2) #store the wcs
        h=wcs.to_header() #convert the wcs into a header
        h['BUNIT']=(specunits,'Best-fit peak velocities') #add the BUNIT keyword
        if specunits == 'm/s': #if the spectral units are m/s
            field=field*1000 #reconvert the velocoty into m/s
        hdu=fits.PrimaryHDU(field.astype(dtype),header=h) #create the primary HDU
        vfield=fits.HDUList([hdu]) #make the HDU list
        vfield.writeto(outname.replace('.fits','_vel.fits'),overwrite=True) #save the velocity field
          
    if write_fits: #if the data must be write into a fits file
        hdu=fits.PrimaryHDU(model_cube.astype(dtype),header=header) #create the primary HDU
        new_data=fits.HDUList([hdu]) #make the HDU list
        new_data.writeto(outname,overwrite=True) #write the data into a fits file
    else:
        return model_cube.astype(dtype)
            
#############################################################################################
def getpv(pvcube='',pvwidth=None,pvpoints=None,pvangle=None,pvchmin=0,pvchmax=None,pvoutdir='',
          write_fits=False,fitsoutname='',plot=False,saveplot=False,plotoutname='',**kwargs):
    """Extract the position-velocity slice of a given data cube along a path defined by the given points, angle and width. It also (optionally) plots the slice and (optionally) save it to a fits file.

    Args:
        pvcube (string/ndarray): name or path+name of the fits data cube
        pvwidth (float): width of the slice in arcsec. If not given, will be thebeam size
        pvpoints (array-like): ICRS RA-DEC comma-separated coordinate of the slice points in decimal degree. If two are given ([x,y]), it is assumed they are the center of the slice. Else, they need to be the starting and ending coordinates ([xmin,xmax,ymin,ymax])
        pvangle (float): position angle of the slice in degree when two points are given. If not given, will be the object position angle
        pvchmin (int): first channel of the slice
        pvchmax (int): last channel of the slice
        pvoutdir (string): string of the output folder name
        write_fits (bool): save the slice as fits file if True
        fitsoutname (string): string of the output fits file name
        plot (bool): plot the slice if True
        saveplot (string): save the plot if True
        plotoutname (string): string of the output plot file name
    
    Kwargs:
        datacube (string/ndarray): name or path+name of the fits data cube if pvcube is not given
        path (string): path to the data cube if the datacube is a name and not a path+name
        pixunits (string): string with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - armin
            - arcsec
        pixelres (float): cube spatial resolution in pixunits (Default: None)
        bmaj (float): beam major axis in arcsec
        pa (float): object position angle in degree
        pvsig (float): lowest contour level in terms of pvsig*rms
        figure (bool): create a plot figure if True
        position (int): position of the subplot in the figure as triplet of integers (111 = nrow 1, ncol 1, index 1)
        vsys (float): object systemic velocity in m/s
        rms (float): rms of the data cube in Jy/beam as a float. If not given (None), the function tries to calculate it
        asectokpc (float): arcsec to kpc conversion to plot the spatial scale
        objname (string): name of the object
        subtitle (string): subtitle of the pv plot
        lim (array-like): list or array of plot x and y limits as [xmin,xmax,ymin,ymax]. They will replace the default limits
        pv_ctr (array-like): contour levels in units of rms. They will replace the default levels                                    
        ctr_width (float): line width of the contours
        verbose (bool): option to print messages and plot to terminal if True   

    Returns:
        Pv slice of a cube as fits file or PrimaryHUD object
        
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no path is provided
        ValueError: If no starting and/or ending points are given
        ValueError: If number of given points is incorrect
        ValueError: If points are not given as list or tuple
        ValueError: If no position angle is provided
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    if type(pvcube)==str or pvcube is None: #if the datacube is a string
        datacube=__read_string(pvcube,'datacube',**kwargs) #store the path to the data cube
    else:
        datacube=pvcube
    if pvwidth is None: #if width is not given
        if verbose:
            warnings.warn('Pv width not set. Trying to set it to the beam major axis!')
        if 'bmaj' in kwargs: #if the beam major axis is in kwargs
            if kwargs['bmaj'] == '' or kwargs['bmaj'] is None:
                warnings.warn('Beam major axis not set. Pv width will be 1 pixel!')
                pvwidth=1
            else:
                pvwidth=kwargs['bmaj'] #store the beam major axis from the input kwargs
        else:
            warnings.warn('Beam major axis not set. Pv width will be 1 pixel!')
            pvwidth=1
    if pvpoints is None: #if no coordinates are given for the center
        raise ValueError('ERROR: No points provided for the pv slice: aborting!')
    if type(pvpoints) != list:
        raise ValueError('ERROR: Pv points should be given as a list of tuples [(x,y)] or [(xmin,ymin),(xmax,ymax)]: aborting!')
    if len(pvpoints)==2: #if two points are given
        from_center=True #extract the slice from its center
        pvpoints=SkyCoord(ra=pvpoints[0]*u.degree,dec=pvpoints[1]*u.degree,frame='icrs')
    elif (len(pvpoints)%2) ==0: #if even points are given
        from_center=False #extract the slice between them
        ra=[] #initialize the ra list
        dec=[] #initialize the dec list
        for i in range(0,len(pvpoints),2): #run over the path points
            ra.append(pvpoints[i])
            dec.append(pvpoints[i+1])
        pvpoints=ICRS(ra*u.degree,dec*u.degree) #convert the points into ICRS coordinates
    else:
        raise ValueError('ERROR: wrong number of coordinates given {}, required an even number. Aborting!'.format(len(pvpoints)))
    if from_center: #if the slice is defined from its center
        if pvangle is None: #if angle is not given
            if 'pa' in kwargs: #if the object position angle is in kwargs
                if kwargs['pa'] == '' or kwargs['pa'] is None:
                    raise ValueError('ERROR: no position angle is set: aborting!')
                else:
                    pvangle=kwargs['pa']
            else:
                raise ValueError('ERROR: no position angle is set: aborting!')
        pvangle=(180+pvangle)*u.degree #convert it due to position angle definition of pvextractor
    chmin=pvchmin #store the lower channel from the input parameters
    chmax=pvchmax #store the upper channel from the input parameters
    outdir=pvoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder    
    pvoutname=fitsoutname #store the output name from the input parameters
    if pvoutname == '' or pvoutname is None:  #if the outname is empty
        pvoutname=datacube.replace('.fits','_pvslice.fits') #the outname is the object name plus'_pvslice.fits
    plotoutname=plotoutname #store the output name from the input parameters
    if plotoutname == '' or plotoutname is None:  #if the outname is empty
        plotoutname=datacube.replace('.fits','_pvslice.pdf') #the outname is the object name plus'_pvslice.fits
    #CHECK THE KWARGS#
    pvsig=kwargs['pvsig'] if 'pvsig' in kwargs else 3
    pixunits=kwargs['pixunits'] if 'pixunits' in kwargs else None
    pixelres=kwargs['pixelres'] if 'pixelres' in kwargs else None
    figure=kwargs['figure'] if 'figure' in kwargs else False
    position=kwargs['position'] if 'position' in kwargs else 111
    rms=kwargs['rms'] if 'rms' in kwargs else None
    vsys=kwargs['vsys'] if 'vsys' in kwargs else None
    asectokpc=kwargs['asectokpc'] if 'asectokpc' in kwargs else None
    objname=kwargs['objname'] if 'objname' in kwargs else None
    if objname is None: #if in kwargs but not set
        objname='' #set it to empty
    subtitle=kwargs['subtitle'] if 'subtitle' in kwargs else None
    lim=kwargs['lim'] if 'lim' in kwargs else None
    pv_ctr=kwargs['pv_ctr'] if 'pv_ctr' in kwargs else None
    ctr_width=kwargs['ctr_width'] if 'ctr_width' in kwargs else 2

    #---------------   START THE FUNCTION   ---------------#
    data,header=__load(datacube) #open the data cube
    wcs=WCS(header) #store the wcs information
    
    #WE CHECK IF THE REQUIRED INFORMATION ARE PROVIDED#
    #------------   SPATIAL UNITS     ------------#
    if header is not None: #if the header is None means the input data is not a fits file
        if pixunits is None and 'CUNIT1' in header: #if no spatial units are given and the keyword is in the header
            pixunits=header['CUNIT1']#store the spatial units from the cube header
        elif pixunits is None and 'CUNIT2' in header: #if no spatial units are given and the keyword is in the header
            pixunits=header['CUNIT2']#store the spatial units from the cube header
    if pixunits is None:
        pixunits='deg' #set them to deg
        if verbose:
            warnings.warn('No spatial units were found: spatial units set to deg!')
    #------------   SPATIAL RESOLUTION     ------------#
    if header is not None: #if the header is None means the input data is not a fits file
        if pixelres is None and 'CDELT1' in header: #if the spatial resolution is not given and the keyword is in the header
            pixelres=header['CDELT1'] #store the spatial resolution from the cube header
        elif pixelres is None and 'CDELT2' in header: #if the spatial resolution is not given and the keyword is in the header
            pixelres=header['CDELT2'] #store the spatial resolution from the cube header
    if pixelres is None and verbose:
            warnings.warn('No spatial unit was found: unable to calculate the pixel resolution! Path length set to 0.5 deg')
    if pixelres is not None: #if the pixel resolution is available
        if pixunits == 'deg': #if the spatial unit is degree
            pixelres*=3600 #convert into arcsec
        elif pixunits == 'arcmin': #if the spatial unit is arcmin
            pixelres*=60 #convert into arcsec
        elif pixunits == 'arcsec': #if the spatial unit is arcsec
            pass #do nothing
    if pixelres < 0: #if the pixel resolution is negative
        pixelres=-pixelres #convert to positive

    #WE DEFINE THE PATH OF THE SLICE#
    if pixunits == 'deg': #if the spatial unit is degree
        pvwidth*=3600*u.arcsec #convert into arcsec
    elif pixunits == 'arcmin': #if the spatial unit is arcmin
        pvwidth*=60*u.arcsec #convert into arcsec
    elif pixunits == 'arcsec': #if the spatial unit is arcsec
        pvwidth*=u.arcsec #do nothing    
        
    if from_center: #if the slice type is from the center
        if pixelres is None: #if no pixel resolution is given
            length=0.5*u.degree #define the length of the slice as the size of the image
        else:
            length=np.sqrt((data.shape[0]*pixelres)**2+(data.shape[1]*pixelres)**2)*u.arcsec #define the length of the slice as the size of the image
        pvpath=PathFromCenter(center=pvpoints,length=length,angle=pvangle,width=pvwidth) #define the path for the pv slice
    else:
        pvpath=Path(pvpoints,width=pvwidth) #define the path for the pv slice  
        
    #WE CHECK THE CHANNELS
    if chmin is None or chmin < 0: #if not given or less than 0
        warnings.warn('Starting channel wrongly set. You give {} but should be at least 0. Set to 0'.format(chmin))
        chmin=0 #set it to 0
    if chmax is None: #if the upper channel is not set
        chmax=data.shape[0] #slice until the last channel
    elif chmax > data.shape[0]: #if the higher channel is larger than the size of the data
        warnings.warn('Use choose a too high last channel ({}) but the cube has {} channels. Last channel set to {}.'.format(chmax,data.shape[0],data.shape[0]))
        chmax=data.shape[0] #set to the max channel
    elif chmax < chmin: #if the higher channel is less than the lower
        warnings.warn('Last channel ({}) lower than starting channel ({}). Last channel set to {}.'.format(chmax,chmin,data.shape[0]))
        chmax=data.shape[0] #set to the max channel
    
    if verbose:
        if from_center:
            print('Extracting the pv slice with the following parameters:\n'
            'Spatial resolution: {:.1f} arcsec\n'
            'Path length: {:.1f}\n'
            'Path position angle: {:.1f} deg\n'
            'Path width: {:.1f}\n'
            'From channel {} to channel {}\n'
            '-------------------------------------------------'.format(pixelres,length,pvangle,pvwidth,chmin,chmax))
        else:
            print('Extracting the pv slice with the following parameters:\n'
            'Spatial resolution: {:.1f} arcsec\n'
            'Path width: {:.1f}\n'
            'From channel {} to channel {}\n'
            '-------------------------------------------------'.format(pixelres,pvwidth,chmin,chmax))
    #WE EXTRACT THE PVSLICE AND REFER THE SPATIAL AXIS TO THE SLICE CENTER#    
    pv=extract_pv_slice(data[chmin:chmax,:,:],pvpath,wcs=wcs[chmin:chmax,:,:]) #extract the pv slice
    pv.header['CRPIX1']=round(pv.header['NAXIS1']/2)+1 #fix the header in order to have the distance from the center as spatial dimension
    
    #WE SAVE THE SLICE IF NEEDED#
    if write_fits: #if the slice must be saved
        pv.writeto(outdir+pvoutname,overwrite=True) #write the pv as fits file
    
    #WE DO THE PLOT IF NEEDED#
    if plot: #if the slice must be plotted
        #PREPARE THE DATA#
        data=pv.data #store the pv data
        if rms is None: #if the rms is not given
            if verbose:
                warnings.warn('No rms is given: calculating the rms on the pv data')
            rms=np.sqrt(np.nanmean(data**2)) #calculate from the data
        #PREPARE THE FIGURE#
        if figure: #if it is a figure
            nrows=1 #number of rows in the atlas
            ncols=1 #number of columns in the atlas
            fig=plt.figure(figsize=(6*ncols,7*nrows)) #create the figure
            fig.suptitle('{} p-v plot'.format(objname),fontsize=24) #add the title
            ax=fig.add_subplot(nrows,ncols,1,projection=WCS(pv.header)) #create the subplot
        #PREPARE THE SUBPLOT#
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=WCS(pv.header)) #create the subplot
        #WE DEFINE THE LIMITS#
        if lim is None: #if no axes limits are given
            xlim=[np.min(np.where(~np.isnan(data))[1]),np.max(np.where(~np.isnan(data))[1])] #set the x limit by to the first and last non-nan
            ylim=[np.min(np.where(~np.isnan(data))[0]),np.max(np.where(~np.isnan(data))[0])] #set the y limit by to the first and last non-nan
        else: #get it from the input
            xlim=[lim[0],lim[1]]
            ylim=[lim[2],lim[3]]
        if pv.header['CDELT2'] < 0: #if the spectral separation is negative
            ylim=np.flip(ylim) #flip the y-axis
        leftmargin=0.025 #left margin for annotations position
        bottommargin=0.025 #bottom margin for annotations position
        topmargin=0.95 #top margin for annotations position
        #WE DEFINE THE NORMALIZATION#
        vmin=0.1*np.nanmin(data) #set the lower limit for the normalization
        vmax=0.7*np.nanmax(data) #set the upper limit for the normalization
        norm=cl.PowerNorm(gamma=0.3,vmin=vmin,vmax=vmax)
        #WE DEFINE THE CONTOURS#
        if pv_ctr is None: #if no contour levels are provided
            ctr=np.power(pvsig,np.arange(1,10,2)) #4 contours level between nsigma and nsigma^9
        else:
            ctr=pv_ctr #use those in input
        if verbose:
            print('Contours level: {} Jy/beam'.format(ctr*rms))
        #WE DO THE PLOT#
        im=ax.imshow(data,cmap='Greys',norm=norm,aspect='auto') #plot the pv slice
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        if pv.header['CUNIT2']=='m/s': #if the spectral units are m/s
            ax.coords[1].set_format_unit(u.km/u.s) #convert to km/s
        ax.coords[0].set_format_unit(u.arcmin) #convert x units to arcmin
        ax.set_xlabel('Offset from center [arcmin]') #set the x-axis label
        ax.set_ylabel('Velocity [km/s]') #set the y-axis label
        ax.axvline(x=pv.header['CRPIX1']-1,linestyle='-.',color='black') #draw the galactic center line.-1 is due to python stupid 0-couting
        #WE ADD THE CONTOURS#
        ax.contour(data/rms,levels=ctr,cmap='hot',linewidths=ctr_width,linestyles='solid') #add the positive contours
        ax.contour(data/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width,linestyles='dashed') #add the negative contours
        #WE ADD ANCILLARY INFORMATION#
        if vsys is not None: #if the systemic velocity is given
            if pv.header['CUNIT2']=='km/s': #if the spectral units are km/s
                vsys=vsys*1000 #convert the systemic velocity into m/s
            ax.axhline(y=((vsys-pv.header['CRVAL2'])/pv.header['CDELT2'])+pv.header['CRPIX2']-1,linestyle='--',color='black') #draw the systemic velocity line. -1 is due to python stupid 0-couting
        else:
            ax.axhline(y=((0-pv.header['CRVAL2'])/pv.header['CDELT2'])-pv.header['CRPIX2'],linestyle='--',color='black') #draw the systemic velocity line
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,leftmargin,topmargin) #draw the 10-kpc line
        #WE ADD THE SUBTITLE
        if subtitle != '' and subtitle is not None: #if the subtitle is not empty
            ax.set_title('{}'.format(subtitle),pad=60,fontsize=20) #add the subtitle

        #NOW WE SAVE AND PRINT THE PLOT#
        if figure: #if it is a figure
            fig.subplots_adjust(left=0.17, bottom=0.1, right=0.93, top=0.85, wspace=0.0, hspace=0.0) #fix the position of the subplots in the figure   
            if saveplot: #if the save switch is true
                fig.savefig(outdir+plotoutname,dpi=300,bbox_inches='tight') #save the figure
            if verbose: #if print-to-terminal option is true
                plt.show() #show the figure
            else:
                plt.close()
            #fig.clf() #clear the figure from the memory
    else: #if no plot has to be made
        return pv #return the pvslice
         
#############################################################################################
def plotmom(which='all',mom0map='',mom1map='',mom2map='',plotmomoutdir='',plotmomoutname='',save=False,**kwargs):
    """Plot the given moment maps. All (moment 0, moment 1 and moment 2) can be plotted simultaneously or only one.
    
    Args:
        which (string): what the function has to plot between:
            all, plot moment 0, moment 1 and moment 2 map in a single 1-row 3-columns figure
            mom0, plot moment 0 map
            mom1, plot moment 1 map
            mom2, plot moment 2 map
        mom0map (string/ndarray): name or path+name of the fits moment 0
        mom1map (string/ndarray): name or path+name of the fits moment 1
        mom2map (string/ndarray): name or path+name of the fits moment 2
        plotmomoutdir (string): output folder name
        plotmomoutname (string): output file name
        save (bool): save the plot if True
    
    Kwargs:
        path (string): path to the moment map if the momXmap is a name and not a path+name
        pbcorr (bool): apply the primary beam correction if True. Note that in that case you must supply a beam cube (Default: False)
        beamcube (string/ndarray): name or path+name of the fits beam cube if pbcorr is True
        use_cube (bool): use a data cube to get information like the rms, the spectral resolution and the beam if True (Default: False). Note that in that case you must supply a data cube
        datacube (string/ndarray): name or path+name of the fits data cube if use_cube is True
        pixunits (string): string with the spatial units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - deg
            - armin
            - arcsec
        specunits (string): string with the spectral units (Default: None). Accepted values:
            - None (it will try to retrieve them from the cube header)
            - km/s
            - m/s
            - Hz
        spectralres (float): cube spectral resolution in km/s (Default: None)
        bmaj (float): beam major axis in arcsec (Default: None)
        bmin (float): beam minor axis in arcsec (Default: None)
        bpa (float): beam position angle in degree (Default: None)
        nsigma (float): lowest contour level in terms of nsigma*rms (Default: 3)
        vsys (float): object systemic velocity in km/s (Default: None)
        pixelres (float): pixel resolution of the data in arcsec (Default: None)
        asectokpc (float): arcsec to kpc conversion to plot the spatial scale (Default: None)
        objname (string): name of the object (Default: '')
        subtitle (string): subtitle of the pv plot (Default: None)
        wcs (astropy.WCS): wcs as astropy.WCS object to use in the plot. It will replace the default one (Default: None)
        position (int): position of the subplot in the figure as triplet of integers (111 = nrow 1, ncol 1, index 1) (Default: 111)
        rms (float): rms of the data cube in Jy/beam as a float. If not given (None), the function tries to calculate it (Default: None)
        mom0_ctr (array-like): list or array of moment 0 contours level in units of 10^18. They will replace the default levels (Default: None)
        mom1_ctr (array-like): list or array of moment 1 contours level in units of km/s. They will replace the default levels (Default: None)
        mom2_ctr (array-like): list or array of moment 2 contours level in units of km/s. They will replace the default levels (Default: None)
        lim (array-like): list or array of plot x and y limits as [xmin,xmax,ymin,ymax]. They will replace the default limits (Default: None)
        mom0_cmap (string/ndarray): name of the colormap to be used for the moment 0 map. Accepted values are those of matplotlib.colors.Colormap (see: https://matplotlib.org/stable/tutorials/colors/colormaps.html) (Default: 'Greys')
        mom1_cmap (string/ndarray): name of the colormap to be used for the moment 1 map. Accepted values are those of matplotlib.colors.Colormap (see: https://matplotlib.org/stable/tutorials/colors/colormaps.html) (Default: 'jet')
        mom2_cmap (string/ndarray): name of the colormap to be used for the moment 2 map. Accepted values are those of matplotlib.colors.Colormap (see: https://matplotlib.org/stable/tutorials/colors/colormaps.html) (Default: 'YlGnBu')
        ctr_width (float): line width of the contours (Default: 2)
        matplotlib.colors.Colormap (see: https://matplotlib.org/stable/tutorials/colors/colormaps.html) (Default: 'Greys')
        mom0_ctrmap (string/ndarray): name of the colormap to be used for the moment 0 map contour levels. Accepted values are those of matplotlib.colors.Colormap (see: https://matplotlib.org/stable/tutorials/colors/colormaps.html) (Default: 'hot')
        mom2_ctrmap (string/ndarray): name of the colormap to be used for the moment 2 map contour levels. Accepted values are those of matplotlib.colors.Colormap (see: https://matplotlib.org/stable/tutorials/colors/colormaps.html) (Default: 'RdPu_r')
        verbose (bool): option to print messages and plot to terminal if True (Default: False)
    
    Returns:
        None
    
    Raises:
        ValueError: If 'which' argument does not match accepted values
        ValueError: If no moment 0 or moment 1 or moment 2 are given
        ValueError: If no path is provided
        ValueError: If no output folder is set
        ValueError: If no data cube is given when use_cube is True
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    if which not in ['all','mom0','mom1','mom2']: #if wrong data is given
        raise ValueError("ERROR: wrong data selection. Accepted values: ['all','mom0','mom1','mom2']. Aborting")
    if which in ['all','mom0']: #if all moment maps must be plotted or only the moment 0
        if type(mom0map)==str or mom0map is None: #if the moment 0 map is a string
            mom0map=__read_string(mom0map,'mom0map',**kwargs) #store the path to the moment 0 map
        #------------   PB CORRECTION     ------------#
        pbcorr=kwargs['pbcorr'] if 'pbcorr' in kwargs else False #store the apply pb correction option
        if pbcorr: #if the primary beam correction is applied
            beamcube=kwargs['beamcube'] if 'beamcube' in kwargs else None #store the data cube path from the input parameters
            if beamcube is None: #if the beam cube is not provided
                if verbose:
                    warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')
                pbcorr=False #set the pbcorr option to False
            elif type(beamcube)==str: #if the beam cube is a string
                beamcube=__read_string(beamcube,'beamcube',**kwargs) #store the path to the beam cube
    if which in ['all','mom1']: #if all moment maps must be plotted or only the moment 1 map
        if type(mom1map)==str or mom1map is None: #if the moment 1 map is a string
            mom1map=__read_string(mom1map,'mom1map',**kwargs) #store the path to the moment 1 map
        vsys=kwargs['vsys'] if 'vsys' in kwargs else None #store the systemic velocity from the input kwargs
    if which in ['all','mom2']: #if all moment maps must be plotted or only the moment 2 map
        if type(mom2map)==str or mom2map is None: #if the moment 2 map is a string
            mom2map=__read_string(mom2map,'mom2map',**kwargs) #store the path to the moment 2 map
    outdir=plotmomoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder  
    outname=plotmomoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        if which ==  'all': #if all maps must be plotted
            outname='{}_maps.pdf'.format(objname)
        if which in ['all','mom0']: #if only the moment 0 must be plotted
            outname=mom0map.replace('.fits','.pdf')  #the outname is the input .pdf
        if which in ['all','mom1']: #if only the moment 1 must be plotted
            outname=mom1map.replace('.fits','.pdf')  #the outname is the input .pdf
        if which in ['all','mom2']: #if the moment 2 must be plotted
            outname=mom2map.replace('.fits','.pdf')  #the outname is the input .pdf
    elif outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    #CHECK THE KWARGS#
    use_cube=kwargs['use_cube'] if 'use_cube' in kwargs else False
    if use_cube: #if the use cube option is True
        datacube=kwargs['datacube'] if 'datacube' in kwargs else None #store the data cube from the input kwargs
        if datacube is None: #if the data cube is not provided
            raise ValueError('ERROR: you set to use a data cube but no data cube is provided: aborting!')
        elif type(datacube)==str: #if the data cube is a string
            datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    pixunits=kwargs['pixunits'] if 'pixunits' in kwargs else None
    specunits=kwargs['specunits'] if 'specunits' in kwargs else None
    spectralres=kwargs['spectralres'] if 'spectralres' in kwargs else None
    bmaj=kwargs['bmaj'] if 'bmaj' in kwargs else None
    bmin=kwargs['bmin'] if 'bmin' in kwargs else None
    bpa=kwargs['bpa'] if 'bpa' in kwargs else None
    nsigma=kwargs['nsigma'] if 'nsigma' in kwargs else 3
    rms=kwargs['rms'] if 'rms' in kwargs else None
    pixelres=kwargs['pixelres'] if 'pixelres' in kwargs else None
    asectokpc=kwargs['asectokpc'] if 'asectokpc' in kwargs else None
    objname=kwargs['objname'] if 'objname' in kwargs else None
    if objname is None: #if in kwargs but not set
        objname='' #set it to empty
    subtitle=kwargs['subtitle'] if 'subtitle' in kwargs else None
    wcs=kwargs['wcs'] if 'wcs' in kwargs else None
    position=kwargs['position'] if 'position' in kwargs else 111
    mom0_ctr=kwargs['mom0_ctr'] if 'mom0_ctr' in kwargs else None
    mom1_ctr=kwargs['mom1_ctr'] if 'mom1_ctr' in kwargs else None
    mom2_ctr=kwargs['mom2_ctr'] if 'mom2_ctr' in kwargs else None
    lim=kwargs['lim'] if 'lim' in kwargs else None
    mom0_cmap=kwargs['mom0_cmap'] if 'mom0_cmap' in kwargs else 'Greys'
    mom1_cmap=kwargs['mom1_cmap'] if 'mom1_cmap' in kwargs else 'jet'
    mom2_cmap=kwargs['mom2_cmap'] if 'mom2_cmap' in kwargs else 'YlGnBu'
    ctr_width=kwargs['ctr_width'] if 'ctr_width' in kwargs else 2
    mom0_ctrmap=kwargs['mom0_ctrmap'] if 'mom0_ctrmap' in kwargs else 'hot'
    mom2_ctrmap=kwargs['mom2_ctrmap'] if 'mom2_ctrmap' in kwargs else 'RdPu_r'

    #---------------   START THE FUNCTION   ---------------#
    #WE CHECK IF CUBE STATISTICS MUST BE COMPUTED#
    if use_cube: #if use cube
        stats=cubestat(datacube,pixunits=pixunits,specunits=specunits,pixelres=pixelres,spectralres=spectralres,bmaj=bmaj,bmin=bmin,bpa=bpa,rms=rms,nsigma=nsigma,verbose=False) #calculate the statistics of the cube
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
        sens=stats['sensitivity']      
    #---------------   PREPARE THE DATA   ---------------#
    #WE PREPARE THE MOMENT 0 MAP#
    if which == 'all' or which == 'mom0': #if all moment maps or moment 0 map must be plotted
        #WE GET THE DATA AND DO A SIMPLE CONVERSION#
        mom0,mom0header=__load(mom0map) #open the moment 0 map
        #WE TRY TO GET THE PIXEL RESOLUTION#
        if mom0header is not None: #if the header is None means the input data is not a fits file
            if pixelres is None and 'CDELT1' in mom0header: #if the pixelres is not given and is in the header
                pixelres=mom0header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if mom0header is not None: #if the header is None means the input data is not a fits file
            if pixunits is None and 'CUNIT1' in mom0header: #if the spatial unit is in the header
                pixunits=mom0header['CUNIT1'] #store the spatial unit
        #WE DO THE PB CORRECTION IF NEEDED#    
        if pbcorr: #if the primary beam correction is applied
            with fits.open(beamcube) as pb_cube: #open the primary beam cube
                pb_slice=pb_cube[0].data[np.array(pb_cube[0].shape[0]/2).astype(int)] #extract the central plane
            mom0=mom0/pb_slice #apply the pb correction
            if mom0header is not None: #if the header is None means the input data is not a fits file
                momunit=mom0header['BUNIT'].replace('/beam'.casefold(),'')
        elif mom0header is not None: #if the header is None means the input data is not a fits file:
            momunit=mom0header['BUNIT']
        #WE TRY TO CONVERT INTO HI COLUMNS DENSITY# 
        if bmaj is not None and bmin is not None: #if the beam major and minor axis are given
            beamarea=1.13*(bmin*bmaj) #calculate the beam area
        else:
            beamarea=None #set it to None
        if beamarea is not None and spectralres is not None: #if we have the beamarea and spectral resolution
            if spectralres<0: #if the spectral resolution is negative
                spectralres=-spectralres #convert it to positive
            mom0=converttoHI(mom0,fluxunits=momunit,beamarea=beamarea,pixunits=pixunits,spectralres=spectralres,specunits=specunits) #convert the moment 0 map into an HI column density map
            mom0=mom0/10.**(18) #normalize the column density in terms of 10^18 cm^-2
            momunit='cm$^{-2}$' #define the column density unit
            if use_cube: #if use cube
                sens=sens/10.**(18) #normalize the column density in terms of 10^18 cm^-2
            elif rms is not None:
                sens=converttoHI(rms*nsigma,fluxunits='Jy/beam',beamarea=beamarea,pixunits=pixunits,spectralres=spectralres,specunits=specunits)/10.**(18) #calculate the sensitivity and normalize it
            else:
                if verbose:
                    warnings.warn('The sensitivity is refering to the minimum of the moment 0 map.')
                sens=np.nanmin(mom0[mom0>0]) #the sensitivity is the min of the moment 0 map
        if mom0header is not None: #if the header is None means the input data is not a fits file
            if mom0header['BUNIT'].casefold()=='jy/beam*m/s' or mom0header['BUNIT'].casefold()=='jy*m/s': #if the units are in m/s
                mom0=mom0/1000 #convert to km/s
        mom0[mom0==0]=np.nan #convert the moment 0 zeros into nan
        
    #WE PREPARE THE MOMENT 1 MAP#
    if which == 'all' or which == 'mom1': #if all moment maps or moment 1 map must be plotted
        #WE GET THE DATA AND DO A SIMPLE CONVERSION#
        mom1,mom1header=__load(mom1map) #open the moment 1 map
        if mom1header is not None and mom1header['BUNIT']=='m/s': #if the header is None means the input data is not a fits file
            mom1=mom1/1000 #convert to km/s
        #WE TRY TO GET THE PIXEL RESOLUTION 
        if mom1header is not None: #if the header is None means the input data is not a fits file
            if pixelres is None and 'CDELT1' in mom1header: #if the pixelres is not given and is in the header
                pixelres=mom1header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if mom1header is not None: #if the header is None means the input data is not a fits file
            if pixunits is None and 'CUNIT1' in mom1header: #if the spatial unit is in the header
                pixunits=mom1header['CUNIT1'] #store the spatial unit
        #WE REMOVE THE SYSTEMIC VELOCITY                                         
        if vsys is None: #if the systemic velocity is not given
            vsys=np.nanmedian(mom1) #calculate the systemic velocity
        else:
            vsys=vsys/1000 #convert it in km/s
        mom1=mom1-vsys #subract the result to the moment 1 map
        
    #WE PREPARE THE MOMENT 2 MAP             
    if which == 'all' or which == 'mom2': #if all moment maps or moment 2 map must be plotted
        #WE GET THE DATA AND DO A SIMPLE CONVERSION#
        mom2,mom2header=__load(mom2map) #open the moment 2 map
        if mom2header is not None and mom2header['BUNIT']=='m/s': #if the header is None means the input data is not a fits file
            mom2=mom2/1000 #convert to km/s
        #WE TRY TO GET THE PIXEL RESOLUTION 
        if mom2header is not None: #if the header is None means the input data is not a fits file
            if pixelres is None and 'CDELT1' in mom2header: #if the pixelres is not given and is in the header
                pixelres=mom2header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if mom2header is not None: #if the header is None means the input data is not a fits file
            if pixunits is None and 'CUNIT1' in mom2header: #if the spatial unit is in the header
                pixunits=mom2header['CUNIT1'] #store the spatial unit
        #WE CALCULATE THE FWHM           
        disp=np.nanmedian(mom2) #calculate the median velocity dispersion
    
    #WE CONVERT THE BEAM TO ARCSEC
    if pixunits == 'deg': #if the spatial units are deg
        bmaj*=3600 #convert the beam major axis in arcsec
        bmin*=3600 #convert the beam minor axis in arcsec
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        bmaj*=60 #convert the beam major axis in arcsec
        bmin*=60 #convert the beam minor axis in arcsec
    beamarea=1.13*(bmin*bmaj) #calculate the beam area 
            
    #WE CONVERT THE SPATIAL DIMENSION TO ARCSEC
    if pixelres is not None and pixunits == 'deg': #if the spatial unit is degree
        pixelres*=3600 #convert into arcsec
    elif pixelres is not None and pixunits == 'arcmin': #if the spatial unit is arcmin
        pixelres*=60 #convert into arcsec
    if pixelres<0: #if the spatial resolution is negative
        pixelres=-pixelres #convert it to positive

    #---------------   DO THE PLOT   ---------------#
    #WE PREPARE THE FIGURE#
    if which == 'all': #if all moment maps must be plotted
        nrows=1 #number of rows in the atlas
        ncols=3 #number of columns in the atlas        
        fig=plt.figure(figsize=(5*ncols,7*nrows)) #create the figure
        if which == 'all': #if all moment maps must be plotted
            fig.suptitle('{} HI moment maps'.format(objname),fontsize=24) #add the title
        elif which == 'mom0': #if the moment 0 map must be plotted
            fig.suptitle('{} HI moment 0 map'.format(objname),fontsize=24) #add the title
        elif which == 'mom1': #if the moment 1 map must be plotted
            fig.suptitle('{} HI moment 1 map'.format(objname),fontsize=24) #add the title
        elif which == 'mom2': #if the moment 2 map must be plotted
            fig.suptitle('{} HI moment 2 map'.format(objname),fontsize=24) #add the title
            
    #WE PREPARE THE MOMENT 0 SUBPLOT#
    if which == 'all' or which == 'mom0': #if all moment maps must be plotted or only the moment 0 map
        if wcs is None: #if no wcs is given
            wcs=WCS(mom0header)
        if which == 'all': #if all moment maps must be plotted
            ax=fig.add_subplot(nrows,ncols,1,projection=wcs) #create the subplot
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=wcs) #create the subplot
        #WE DEFINE THE LIMITS#
        leftmargin=0.025 #left margin for annotations position
        bottommargin=0.025 #bottom margin for annotations position
        topmargin=0.95 #top margin for annotations position
        if lim is None: #if no axes limits are given
            xmin=np.min(np.where(~np.isnan(mom0))[1]) #set xmin to the first non-nan pixel
            xmax=np.max(np.where(~np.isnan(mom0))[1]) #set xmax to the last non-nan pixel
            ymin=np.min(np.where(~np.isnan(mom0))[0]) #set ymin to the first non-nan pixel
            ymax=np.max(np.where(~np.isnan(mom0))[0]) #set ymax to the last non-nan pixel
            xlim=[xmin*(1-leftmargin),xmax*(1+leftmargin)] #extend a little the xlim
            ylim=[ymin*(0.8-bottommargin),ymax*(2-topmargin)] #extend the ylim to place the ancillary information. 0.8 and 2 are for the text size
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
        #WE DEFINE THE NORMALIZATION#
        if momunit == 'cm$^{-2}$': #if the moment 0 is column density
            norm=cl.PowerNorm(gamma=0.3,vmin=0,vmax=0.75*np.nanmax(mom0)) #define the normalization from negative detection limit to 20% of the maximum
        else:
            norm=cl.PowerNorm(gamma=0.3,vmin=0,vmax=0.75*np.nanmax(mom0)) #define the normalization from negative detection limit to 20% of the maximum
        #WE DEFINE THE CONTOURS#
        if mom0_ctr is None: #if the contours levels are not given
            if momunit == 'cm$^{-2}$': #if the moment 0 is column density
                ctr=np.array([0.1,1,10,100,500,1000,3000,7500]) #contours level in units of 10^18 (min=10^17, max=7.5*10^21
            else:
                ctr=np.linspace(-0.25*np.nanmax(mom0),0.65*np.nanmax(mom0),8) #7 contours levels between the min and max of the moment 0
        else:
            ctr=np.array(mom0_ctr) #use the user-defined levels
        #WE DO THE PLOT#
        im=ax.imshow(mom0,cmap=mom0_cmap,norm=norm,aspect='equal') #plot the moment 0 map
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        #WE ADD THE CONTOURS#
        if momunit == 'cm$^{-2}$': #if the moment 0 is column density
            ax.contour(mom0,levels=ctr,cmap=mom0_ctrmap,linewidths=ctr_width,linestyles='solid') #add the contours
        else:
            ax.contour(mom0,levels=ctr,cmap=mom0_ctrmap,linewidths=ctr_width,linestyles='solid',norm=norm) #add the contours
        if pbcorr: #if primary beam correction has been applied
            pb_slice[pb_slice==0]=np.nan #blank the zeroes
            ax.contour(pb_slice,levels=[np.nanmin(pb_slice)*1.02],colors='gray',linewidths=ctr_width,alpha=0.5) #add the sensitivity cutoff contours
        #WE ADD THE COLORBAR#
        if which == 'all': #if all moment maps must be plotted
            if momunit == 'cm$^{-2}$': #if the moment 0 is column density
                cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='HI column density [10$^{18}$ cm$^{-2}$]',fraction=0.0476) #add the colorbar on top of the plot
            else:
                cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='Flux [{}]'.format(momunit),shrink=1) #add the colorbar on top of the plot
        else:
            if momunit == 'cm$^{-2}$': #if the moment 0 is column density
                cb=plt.colorbar(im,ax=ax,location='top',pad=0.0,label='HI column density [10$^{18}$ cm$^{-2}$]',fraction=0.0476) #add the colorbar on top of the plot
            else:
                cb=plt.colorbar(im,ax=ax,location='top',pad=0.0,label='Flux [{}]'.format(momunit),shrink=1) #add the colorbar on top of the plot
        cb.ax.tick_params(direction='in',length=5) #change the ticks of the colorbar from external to internal and made them longer
        cb.set_ticks(ctr) #set the ticks of the colobar to the levels of the contours
        #WE ADD ANCILLARY INFORMATION#
        if momunit == 'cm$^{-2}$': #if the moment 0 is column density
            ax.text(leftmargin,bottommargin,'Detection limit: {:.1e} {}'.format(sens*10.**(18),momunit),transform=ax.transAxes) #add the information of the detection limit
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,leftmargin+0.05,topmargin-0.01) #draw the 10-kpc line
        #WE ADD THE SUBTITLE
        if which == 'mom0': #if only the moment 0 map must be plotted
            if subtitle != '' and subtitle is not None: #if the subtitle is not empty
                ax.set_title('{}'.format(subtitle),pad=60,fontsize=20) #add the subtitle
    
    #WE PREPARE THE MOMENT 1 SUBPLOT#
    if which == 'all' or which == 'mom1': #if all moment maps must be plotted or only the moment 0 map
        if wcs is None: #if no wcs is given
            wcs=WCS(mom1header)
        if which == 'all': #if all moment maps must be plotted
            ax=fig.add_subplot(nrows,ncols,2,projection=wcs) #create the subplot
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=wcs) #create the subplot
        #WE DEFINE THE LIMITS#
        leftmargin=0.025 #left margin for annotations position
        bottommargin=0.025 #bottom margin for annotations position
        topmargin=0.95 #top margin for annotations position
        if lim is None: #if no axes limits are given
            xmin=np.min(np.where(~np.isnan(mom1))[1]) #set xmin to the first non-nan pixel
            xmax=np.max(np.where(~np.isnan(mom1))[1]) #set xmax to the last non-nan pixel
            ymin=np.min(np.where(~np.isnan(mom1))[0]) #set ymin to the first non-nan pixel
            ymax=np.max(np.where(~np.isnan(mom1))[0]) #set ymax to the last non-nan pixel
            xlim=[xmin*(1-leftmargin),xmax*(1+leftmargin)] #extend a little the xlim
            ylim=[ymin*(0.8-bottommargin),ymax*(2-topmargin)] #extend the ylim to place the ancillary information. 0.8 and 2 are for the text size
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
        #WE DEFINE THE NORMALIZATION#
        norm=cl.CenteredNorm()
        #WE DEFINE THE CONTOURS#
        if mom1_ctr is None: #if the contours levels are not given
            ctr_res=np.floor(0.85*np.nanmax(np.abs(mom1))/15)*5 #we want the contours to be multiple of 5
            ctr_pos=np.arange(0,0.85*np.nanmax(np.abs(mom1)),ctr_res).astype(int) #positive radial velocity contours
            ctr_neg=np.flip(-ctr_pos) #negative radial velocity contours
        else:
            ctr=np.array(mom1_ctr) #use the user-defined levels
        #WE DO THE PLOT#
        im=ax.imshow(mom1,cmap=mom1_cmap,norm=norm,aspect='equal') #plot the moment 1 map with a colormap centered on 0
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if which == 'all': #if it is an atlas
            ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
        #WE ADD THE CONTOURS#
        ax.contour(mom1,levels=ctr_pos,colors='black',linewidths=ctr_width,linestyles='solid') #add the contours
        ax.contour(mom1,levels=ctr_neg,colors='gray',linewidths=ctr_width,linestyles='dashed') #add the contours
        ax.contour(mom1,levels=[0],colors='dimgray',linewidths=2*ctr_width,linestyles='solid') #add the contour for the 0-velocity
        #WE ADD THE COLORBAR#
        if which == 'all': #if it is an atlas
            cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='Radial velocity [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        else:
            cb=plt.colorbar(im,ax=ax,location='top',pad=0.0,label='Radial velocity [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        cb.ax.tick_params(direction='in',length=5) #change the ticks of the corobar from external to internal and made them longer
        cb.set_ticks(np.concatenate((ctr_neg,ctr_pos),axis=None)) #set the ticks of the colobar to the levels of the contours
        #WE ADD ANCILLARY INFORMATION#
        ax.text(leftmargin,bottommargin,'Systemic velocity: {:.1f} km/s'.format(vsys),transform=ax.transAxes) #add the information of the systemic velocity
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,leftmargin+0.05,topmargin-0.01) #draw the 10-kpc line
        #WE ADD THE SUBTITLE
        if which == 'all' or which == 'mom1': #if all moment maps must be plotted or only the moment 1 map
            if subtitle != '' and subtitle is not None: #if the subtitle is not empty
                ax.set_title('{}'.format(subtitle),pad=60,fontsize=20) #add the subtitle
                
    #WE PREPARE THE MOMENT 2 SUBPLOT#
    if which == 'all' or which == 'mom2': #if all moment maps must be plotted or only the moment 0 map
        if wcs is None: #if no wcs is given
            wcs=WCS(mom2header)
        if which == 'all': #if all moment maps must be plotted
            ax=fig.add_subplot(nrows,ncols,3,projection=wcs) #create the subplot
        else:
            fig=plt.gcf() #get the current figure
            ax=fig.add_subplot(position,projection=wcs) #create the subplot
        #WE DEFINE THE LIMITS#
        leftmargin=0.025 #left margin for annotations position
        bottommargin=0.025 #bottom margin for annotations position
        topmargin=0.95 #top margin for annotations position
        if lim is None: #if no axes limits are given
            xmin=np.min(np.where(~np.isnan(mom2))[1]) #set xmin to the first non-nan pixel
            xmax=np.max(np.where(~np.isnan(mom2))[1]) #set xmax to the last non-nan pixel
            ymin=np.min(np.where(~np.isnan(mom2))[0]) #set ymin to the first non-nan pixel
            ymax=np.max(np.where(~np.isnan(mom2))[0]) #set ymax to the last non-nan pixel
            xlim=[xmin*(1-leftmargin),xmax*(1+leftmargin)] #extend a little the xlim
            ylim=[ymin*(0.8-bottommargin),ymax*(2-topmargin)] #extend the ylim to place the ancillary information. 0.8 and 2 are for the text size
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
        #WE DEFINE THE CONTOURS#
        if mom2_ctr is None: #if the contours levels are not given
            ctr=np.around(np.linspace(disp,0.9*np.nanmax(mom2),5),1) #5 contours level from the median dispersion to the 90% of the max and convert to 1-decimal float
        else:
            ctr=np.array(mom2_ctr) #use the user-defined levels
        #WE DO THE PLOT#
        im=ax.imshow(mom2,cmap=mom2_cmap,aspect='equal') #plot the moment 2 map with a square-root colormap and in units of velocity dispersion
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if which == 'all': #if all moment maps must be plotted
            ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
        #WE ADD THE CONTOURS#
        ax.contour(mom2,levels=ctr,cmap=mom2_ctrmap,linewidths=ctr_width,linestyles='solid') #add the contours
        #WE ADD THE COLORBAR#
        if which == 'all': #if all moment maps must be plotted
            cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='Velocity dispersion [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        else:
            cb=plt.colorbar(im,ax=ax,location='top',pad=0.0,label='Velocity dispersion [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        cb.ax.tick_params(direction='in',length=5) #change the ticks of the colorbar from external to internal and made them longer
        cb.set_ticks(ctr) #set the ticks of the colobar to the levels of the contours
        #WE ADD ANCILLARY INFORMATION#
        ax.text(leftmargin,bottommargin,'Median dispersion: {:.1f} km/s'.format(disp),transform=ax.transAxes) #add the information of the velocity dispersion
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            ax=__plot_beam(pixelres,bmaj,bmin,bpa,xlim,ylim) #plot the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            ax=__plot_kpcline(pixelres,asectokpc,xlim,leftmargin+0.05,topmargin-0.01) #draw the 10-kpc line
        #WE ADD THE SUBTITLE
        if which == 'mom2': #if only the moment 2 map must be plotted
            if subtitle != '' and subtitle is not None: #if the subtitle is not empty
                ax.set_title('{}'.format(subtitle),pad=60,fontsize=20) #add the subtitle
                
    #---------------   FIX SUBPLOTS AND SAVE/SHOW   ---------------#
    if which == 'all': #if all moment maps must be plotted
        fig.subplots_adjust(left=0.07, bottom=0.1, right=0.97, top=0.85, wspace=0.0, hspace=0.3) #fix the position of the subplots in the figure   
        if save: #if the save switch is true
            fig.savefig(outname,dpi=300,bbox_inches='tight') #save the figure
            
        if verbose: #if print-to-terminal option is true
            plt.show() #show the figure
        else:
            plt.close()
        #fig.clf() #clear the figure from the memory
    
#############################################################################################
def removemod(datacube='',modelcube='',maskcube=None,method='subtraction',blankthreshold=0,write_fits=False,removemodoutdir='',
              removemodoutname='',**kwargs):
    """Remove a model from a data cube using five methods: data-model (subtraction), data blanking (blanking), data-model after data blanking (b+s), data-model and negative residual blanking (negblank) and a combination of blanking and negblank (all).

    Args:
        datacube (string/ndarray): name or path+name of the fits data cube
        modelcube (string/ndarray): name or path+name of the fits model cube
        maskcube (string/ndarray): name or path+name of the fits 3D mask to be used in the removal
        method (string): method to remove the model between:
            all, apply blanking and negblank methods
            blanking, blank the pixel in the data cube whose value in the model cube is > than blankthreshold
            b+s, same has above but it subtracts the model from the data AFTER the blanking
            negblank, subtract the model from the data and blank the negative residuals
            subtraction, subtract the model from the data
        blankthreshold (float): flux threshold for all, blanking and b+s methods
        write_fits (boolean): store the output in a fits file (True) or return a variable (False)
        removemodoutdir (string): output folder name
        removemodoutname (string): file name
        
    Kwargs:
        path (string): path to the data cube if the datacube is a name and not a path+name
        verbose (bool): option to print messages to terminal if True 

    Returns:
        Data cube with the model removed as fits file
        
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no model cube is provided
        ValueError: If no path is provided
        ValueError: If method set do not match accepted values
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
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
    outdir=removemodoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder  
    outname=removemodoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        outname=datacube.replace('.fits','_'+method+'.fits')
    elif outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname

    #---------------   START THE FUNCTION   ---------------#    
    data,header=__load(datacube) #open the data cube
    model,_=__load(modelcube) #open the model cube
    if maskcube is None: #if a mask cube is not given
        mask=np.ones(data.shape)
    else:
        mask,_=__load(maskcube) #open the mask cube
    if method in ['all','blanking','b+s']: #if all methods, or the blanking, or the blanking and subtraction must be performed
        data[np.where(model>threshold)]=np.nan #blank the data where the model is above the threshold
    if method in ['all','b+s','subtraction','negblank']: #if all methods, or the blanking and subtraction, or the subtraction or the negative blanking must be performed
        emission=np.where(mask>0) #store the emission coordinates
        data[emission]=data[emission]-model[emission] #subtract the model
    if method in ['all','negblank']: #if all methods, or the negative blanking must be performed
        mask[np.where(mask>0)]=1 #convert the mask into a 1/0 mask
        masked_data=data*mask #mask the data
        data[np.where(masked_data<0)]=np.nan #blank the negative data-model pixels

    dtype='float32' #data type is float32
        
    if write_fits: #if the data must be write into a fits file
        hdu=fits.PrimaryHDU(data.astype(dtype),header=header) #create the primary HDU
        new_data=fits.HDUList([hdu]) #make the HDU list
        new_data.writeto(outname,overwrite=True) #write the data into a fits file
        
    else:
        return data.astype(dtype)
        
#############################################################################################
def rotcurve(vfield='',pa=None,rotcenter=None,rotcurveoutdir='',rotcurveoutname='',save_csv=False,**kwargs):
    """Compute the rotation curve of a galaxy and plot it and (optionally) save it as a csv file.

    Args:
        vfield (string): name or path+name of the fits velocity field
        pa (float): object position angle in degree
        rotcenter (array-like): x-y comma-separated coordinates of the rotational center in pixel
        rotcurveoutdir (string):output folder name
        rotcurveoutname (string):output file name
        save_csv (bool):save the rotation curve as csv file if True
        
    Kwargs:
        path (string): path to the moment map if the momXmap is a name and not a path+name
        vsys (float): object systemic velocity in km/s
        pixunits (string): string with the spatial units. Accepted values:
            - deg
            - armin
            - arcsec
        specunits (string): string with the spectral units. Accepted values:
            - km/s
            - m/s
            - Hz   
        pixelres (float): pixel resolution of the data in arcsec
        asectokpc (float): arcsec to kpc conversion to plot the spatial scale
        objname (string): name of the object
        verbose (bool): option to print messages and plot to terminal if True  

    Returns:
        Plot and (optional) csv file of the galactic rotation curve
        
    Raises:
        ValueError: If no velocity field is provided
        ValueError: If no path is provided
        ValueError: If no galactic center is set
        ValueError: If galactic center is  not given as x-y comma-separated coordinates in pixel
        ValueError: If no output folder is provided
    """    
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    if vfield == '' or vfield is None: #if a moment 1 map is not given
        raise ValueError('ERROR: velocity field is not set: aborting!')
    if vfield[0] != '.': #if the moment 1 map name start with a . means that it is a path to the map (so differs from path parameter)
        if 'path' in kwargs: #if the path to the  moment 1 map is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the velocity field is set: aborting!')
            else:
                vfield=kwargs['path']+vfield
        else:
            raise ValueError('ERROR: no path to the velocity field is set: aborting!')
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
    outdir=rotcurveoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder  
    outname=rotcurveoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        outname=vfield.replace('.fits','_rotcurve.pdf')  #the outname is the object name plus rotvurve.pdf
    elif outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    #CHECK THE KWARGS#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    vsys=kwargs['vsys'] if 'vsys' in kwargs else None
    pixunits=kwargs['pixunits'] if 'pixunits' in kwargs else None
    specunits=kwargs['specunits'] if 'specunits' in kwargs else None
    pixelres=kwargs['pixelres'] if 'pixelres' in kwargs else None
    asectokpc=kwargs['asectokpc'] if 'asectokpc' in kwargs else None
    objname=kwargs['objname'] if 'objname' in kwargs else None
    if objname is None: #if in kwargs but not set
        objname='' #set it to empty

    #IMPORT THE DATA AND SETUP THE SPATIAL/SPECTRAL PROPERTIES#
    with fits.open(vfield) as V: #open the moment 1 map
        data=V[0].data #store the data
        header=V[0].header #store the header
    
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
    elif pa in np.radians(np.arange(45,360,90)): #if the position angle is 45, 135, 225 or 315 deg
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
    
    rotcurve=data[y.astype(int),x.astype(int)]-vsys #extract the rotation curve

    if pixelres is None: #if no pixel resolution is given
        radius=np.sqrt((x-x0)**2+(y-y0)**2) #radius from center in pixel
        units='pixel' #set the units to pixel
    else:
        radius=np.sqrt((x-x0)**2+(y-y0)**2)*pixelres  #radius from center in pixunits
        units='{}'.format(pixunits) #set the units to pixunits
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
        df=pd.DataFrame(columns=['RADIUS [{}]'.format(pixunits),'VROT (APP) [{}]'.format(specunits),'VROT (REC) [{}]'.format(specunits),'VROT (TOT) [{}]'.format(specunits)]) #create the dataframe
        df['RADIUS [{}]'.format(pixunits)]=np.abs(newradius) #store the radius. abs needed since depending on which side is larger, the radius can be positive or negative
        df['VROT (APP) [{}]'.format(specunits)]=newapp #store the approaching rotation curve
        df['VROT (REC) [{}]'.format(specunits)]=newrec #store the receding rotation curve. flip needed since the  receding curve goes from larger to smaller values
        df['VROT (TOT) [{}]'.format(specunits)]=total #store the global rotation curve        
        df.to_csv(outname.replace('.pdf','.csv'),index=False) #convert into a csv

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
    fig.suptitle('{} rotation curve'.format(objname),fontsize=24) #add the title
    ax=fig.add_subplot(nrows,ncols,1) #create the subplot
    ax.plot(recradius,rec,color='red',label='receding side',linewidth=1.5,marker='o') #plot the receding rotation curve
    ax.plot(np.abs(appradius),np.abs(app),color='blue',label='Approaching side',linewidth=1.5,marker='o') #plot the approaching rotation curve
    ax.plot(np.abs(newradius),total,color='green',label='Global',linewidth=1.5,linestyle='--') #plot the approaching rotation curve
    ax.set_xlabel('Radius from center [{}]'.format(units)) #set the x-axis label
    ax.set_ylabel('Velocity [km/s]') #set the y-axis label
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right') #set the legend
    
    fig.subplots_adjust(left=0.17, bottom=0.1, right=0.93, top=0.85, wspace=0.0, hspace=0.0) #fix the position of the subplots in the figure   
    fig.savefig(outname,dpi=300,bbox_inches='tight') #save the figure
    
    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    else:
        plt.close()
    #fig.clf() #clear the figure from the memory
                        
#############################################################################################
def stacking(datacube='',mask2d='',vfield='',pa=None,inc=None,stackcenter=None,ncones=None,from_to=None,diagnostic=False,rms_threshold=3,smooth_kernel=None,link_kernel=3,snrmin=3,rel_threshold=0.9,stackoutdir='',stackoutname='',**kwargs):
    """Stack the spectra extracted from a given number of conic regions around a center starting from a minimum radius up to a maximum radius. Afterwards, it runs a source-finding algorithm on each stacked spectrum to check for detected lines. It optionally store diagnostic plots of the stacking and the source-finding routines and also optionally stores relevant diagnostic fits file of the source-finding routine.

    Args:
        datacube (string/ndarray):name or path+name of the fits data cube
        mask2d (string): name or path+name of the fits 2D emission mask
        vfield (string): name or path+name of the fits velocity field to be used for the spectral alignment
        pa (float): object position angle in degree.
        inc (float): inclination of the object in degrees (0 deg means face-on)
        stackcenter (array-like): x-y comma-separated coordinates of the rotational center in pixel 
        ncones (int): number of conic regions from which the spectra are extracted and stacked
        from_to (array-like): comma-separated min and max radius from the galactic center in pixunits from which the spectra are extracted and stacked
        diagnostic (boolean): store all the diagnostic files and plots. Warning: the diagnostic might occupy large portions of the disk (default: False)
        rms_threshold (float): number of rms to reject flux values in the source finder
        smooth_kernel (int/list of int): kernel size (or list of kernel size) in odd number of channels for spectral smoothing prior the source finding. Set to None or 1 to disable.
        link_kernel (int): minimum odd number of channels covered by a spectral line
        rel_threshold (float): minimum value (from 0 to 1) of the reliability to consider a source reliable. Set to 0 to disable the reliability calculation
        snrmin (float): minimum SNR of a detected line to be reliable
        stackoutdir (string): output folder name
        stackoutname (string): output file name
        
    Kwargs:
        path (string): path to the data cube and the velocity field if one or both is a name and not a path+name
        pixunits (string): string with the spatial units. Will be taken from cube header if not provided. Accepted values:
            - deg
            - armin
            - arcsec
        specunits (string): string with the spectral units. Will be taken from cube header if not provided.  Accepted values:
            - km/s
            - m/s
            - Hz   
        fluxunits (string): string with the flux units. Will be taken from cube header if not provided.
        pixelres (float): pixel resolution of the data. Will be taken from cube header if not provided
        spectralres (float): data spectral resolution in specunits. Will be taken from cube header if not provided
        rms (float): rms of the data cube as a float in fluxunits (Default: None)
        bmaj (float): beam major axis in arcsec in pixunits (Default: None)
        objname (string): name of the object (Default: '')
        emission_width (float): velocity range in specunits to exclude from rms calculation in the stacked spectra. Set to 0 to compute the rms over the whole spectrum (Default: 200 km/s)
        ref_spectrum (csv file): path to the csv file of the reference spectrum. The first column is velocity in specunits, the second column is the flux in fluxunits. If not given, it will be automatically computed (Default: None)
        ref_rms (csv file): path to the csv file of the reference rms as a function of the number of stacked spectra. The firs column is the rms in fluxunits. The number of rows is the number of stacked spectra. If not given, it will be automatically computed (Default: None)
        regrid (boolean): regrid the cube option (Default: False)
        regrid_size (int): how many pixel to regrid. Set to 0 to regrid to a beam (Deafult: 0)
        plot_format (string): format type of the plots (pdf, jpg, png, ...) (Default: pdf)
        ctr_width (float): line width of the contours (Default: 2)
        verbose (bool): option to print messages and plot to terminal if True  

    Returns:
        Plot of the stacking result
        
    Raises:
        ValueError: If mandatory inputs are missing
        ValueError: If no galactic center is set
        ValueError: If galactic center is  not given as x-y comma-separated coordinates in pixel
        ValueError: If no output folder is provided
    """    
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    format=kwargs['plot_format'] if 'plot_format' in kwargs else 'pdf'
    datacube=__read_string(datacube,'datacube',**kwargs) #store the path to the data cube
    mask2d=__read_string(mask2d,'mask2d',**kwargs) #store the path to the 2D mask
    vfield=__read_string(vfield,'vfield',**kwargs) #store the path to the velocity field
    center=stackcenter #store the rotation center from the input parameters
    inputs=np.array([pa,inc,center,ncones,from_to],dtype='object') #store the mandatory inputs
    inputsnames=['pa','inc','center','ncones','from_to'] #store the mandatory input names
    if None in inputs: #check if one or more mandatory inputs are missing
        raise ValueError('ERROR: one or more mandatory inputs are missing ({}): aborting!'.format([inputsnames[i] for i in range(len(inputs)) if inputs[i]==None]).replace("'",""))
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
    outdir=stackoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder  
    outname=stackoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        outname='stacking_results'  #the outname is the data cube name plus _stack.pdf
    elif outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    #CHECK THE KWARGS#
    pixunits=kwargs['pixunits'] if 'pixunits' in kwargs else None
    specunits=kwargs['specunits'] if 'specunits' in kwargs else None
    fluxunits=kwargs['fluxunits'] if 'fluxunits' in kwargs else None
    pixelres=kwargs['pixelres'] if 'pixelres' in kwargs else None
    spectralres=kwargs['spectralres'] if 'spectralres' in kwargs else None
    rms=kwargs['rms'] if 'rms' in kwargs else None
    bmaj=kwargs['bmaj'] if 'bmaj' in kwargs else None
    objname=kwargs['objname'] if 'objname' in kwargs else ''
    reference_spectrum=kwargs['ref_spectrum'] if 'ref_spectrum' in kwargs else None
    reference_rms=kwargs['ref_rms'] if 'ref_rms' in kwargs else None
    regrid=kwargs['regrid'] if 'regrid' in kwargs else False
    regrid_size=kwargs['regrid_size'] if 'regrid_size' in kwargs else 0
    ctr_width=kwargs['ctr_width'] if 'ctr_width' in kwargs else 2
    
    if diagnostic: #if the diagnostic option is true
        diagnostic_outdir=outdir+'diagnostic/'
        if not os.path.exists(diagnostic_outdir): #if the output folder does not exist
            os.makedirs(diagnostic_outdir) #create the folder
            
    #IMPORT THE DATA AND SETUP THE SPATIAL/SPECTRAL PROPERTIES#
    data,header=__load(datacube) #load the data cube
    mask,_=__load(mask2d) #load the mask
    
    # NOW WE CHECK FOR THE RELEVANT INFORMATION #
    #------------   CUBE PROPERTIES    ------------#
    prop=[pixunits,specunits,fluxunits,pixelres,spectralres,rms,bmaj] #list of cube properties values
    prop_name=['pixunits','specunits','fluxunits','pixelres','spectralres','rms','bmaj'] #list of cube properties names
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if one or more cube parameters are not given
        if verbose:
            not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
            warnings.warn('I am missing some information: {}. Running cubestat to retrieve them!'.format(not_found))
        stats=cubestat(datacube,pixunits=pixunits,specunits=specunits,fluxunits=fluxunits,pixelres=pixelres,spectralres=spectralres,bmaj=None,bmin=1,bpa=1,rms=rms,nsigma=1,verbose=True) #calculate the statistics of the cube
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
    prop=[pixunits,specunits,fluxunits,pixelres,spectralres,rms,bmaj] #update the list of cube properties values
    if len([prop[i] for i in range(len(prop)) if prop[i] is None])>0: #if still one or more cube parameters are not given
        not_found=[prop_name[i] for i in range(len(prop)) if prop[i] is None]
        raise ValueError('ERROR: I am still missing some information: {}. Please check the parameter!'.format(not_found))   
        
    #SETUP THE VELOCITY AXIS PROPERTIES#
    if 'CRVAL3' in header and 'CRPIX3' in header: #if the header has the starting spectral value
        v0=header['CRVAL3']+(header['CRPIX3']-1)*spectralres #store the starting spectral value
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
    #MASK THE EMISSION IN THE CUBE#
    if verbose:
        print('Masking the emission\n----------------------------------------------')
    if mask.shape != data.shape[1:3]: #if the mask shape is not the same of the data cube
        if verbose:
            warnings.warn('mask spatial shape {} and data spatial shape {} mismatch. Cannot apply the detection mask; make sure the supplied data cube is already masked from the emission.'.format(mask.shape,data.shape[1:3]))
    else:
        mask=mask.astype('float32') #convert the mask into a float array to assign nan values
        mask[mask!=0]=np.nan #set to nan the non-0 voxel
        mask[mask==0]=1 #set to 1 the 0 voxel
        mask=np.array([mask]*data.shape[0]) #transform the 2D mask into a cube
        data*=mask #mask the emission the data

    if diagnostic:
        if verbose:
            print('Writing the masked cube\n----------------------------------------------')
        hdu=fits.PrimaryHDU(data.astype('float32'),header=header) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        if objname != '': #if the object name is given
            hdul.writeto(diagnostic_outdir+'{}_masked_cube.fits'.format(objname),overwrite=True) #write the data into a fits
        else:
            hdul.writeto(diagnostic_outdir+'masked_cube.fits',overwrite=True) #write the data into a fits 

    #REGRID THE CUBE IF NECESSARY#
    if regrid: #if the cube must be regridded
        wcs=WCS(header) #store the wcs
        if regrid_size==0: #if the regrid size is 0
            regrid_size=int(bmaj//np.abs(pixelres)) #define how many pixel is a beam
        regrid_header=wcs[:,::regrid_size,::regrid_size].to_header() #create the regridded header
        header['NAXIS1']=data.shape[2]//regrid_size #redefine the x-axis
        header['NAXIS2']=data.shape[1]//regrid_size #redefine the y-axis
        for key in regrid_header.keys(): #update the header with the new wcs
            if key in header.keys():
                header[key]=regrid_header[key]
        shape_out=tuple([header['NAXIS{0}'.format(i + 1)] for i in range(header['NAXIS'])][::-1]) #define the shape of the regridded cube
        if verbose:
            print('Spatially regridding the cube by a factor of {}\n----------------------------------------------'.format(regrid_size))
        data=spatial_regrid((data,wcs),WCS(regrid_header),shape_out=shape_out,order='bilinear')[0] #regrid the data
        pixelres*=regrid_size #recalculate the pixel resolution
        x0/=regrid_size #recalculate the stacking region center
        y0/=regrid_size #recalculate the stacking region center
        
        if diagnostic:
            if verbose:
                print('Writing the regridded cube\n----------------------------------------------')
            hdu=fits.PrimaryHDU(data.astype('float32'),header=header) #create the primary HDU
            hdul=fits.HDUList([hdu]) #make the HDU list
            if objname != '': #if the object name is given
                hdul.writeto(diagnostic_outdir+'{}_{}x_regrid_cube.fits'.format(objname,regrid_size),overwrite=True) #write the data into a fits
            else:
                hdul.writeto(diagnostic_outdir+'{}x_regrid_cube.fits'.format(regrid_size),overwrite=True) #write the data into a fits 
        
    #CREATE THE REFERENCE NOISE SPECTRUM#
    #It assumes that for a non-aligned cube, the stacking produces a spectrum of only noise, no matter how many spectra it stacks
    if reference_spectrum is None or reference_rms is None: #if no reference spectrum or rms is provided
        if verbose:
            warnings.warn('No reference spectrum and/or reference rms is provided. Starting their computation:')
            print('Computing the reference stacked spectrum\n----------------------------------------------')
        x=np.where(~np.isnan(data[0]))[1] #store the x-coordinates of the non-nan values
        y=np.where(~np.isnan(data[0]))[0] #store the y-coordinates of the non-nan values
        reference_spectrum,reference_rms,exp=__stack(data,x,y,flip=flip) #store the reference spectrum
        
        #SAVE THE REFERENCE NOISE SPECTRUM INTO A CSV#
        #!!!consider to make this section optional and controlled through save_csv=True!!!
        if verbose:
            print('Saving the reference stacked spectrum and the reference rms\n----------------------------------------------')
        df=pd.DataFrame() #create the dataframe
        df['V [km/s]']=v #store the spectral axis
        df['<FLUX> [{}]'.format(fluxunits)]=reference_spectrum #store the reference stacked spectrum
        if objname != '': #if the object name is given
            df.to_csv(outdir+'{}_reference_stacked_spectrum.csv'.format(objname),index=False) #convert into a csv
        else:
            df.to_csv(outdir+'reference_stacked_spectrum.csv',index=False) #convert into a csv
            
        df=pd.DataFrame() #create the dataframe
        df['RMS [{}]'.format(fluxunits)]=reference_rms #store the reference stacked rms
        if objname != '': #if the object name is given
            df.to_csv(outdir+'{}_reference_stacked_rms.csv'.format(objname),index=False) #convert into a csv
        else:
            df.to_csv(outdir+'reference_stacked_rms.csv',index=False) #convert into a csv
            
        #PLOT THE REFERENCE NOISE SPECTRUM#
        ncols=2
        nrows=1
        fig=plt.figure(figsize=(6*ncols,6*nrows)) #create the figure for the spectra
        if objname != '': #if the object name is given
            fig.suptitle('{} reference stacked spectrum'.format(objname),fontsize=24) #add the title
        else:
            fig.suptitle('Reference stacked spectrum',fontsize=24) #add the title
            
        fig=__plot_stack_result(v,reference_spectrum,reference_rms,exp,fluxunits)
        fig.subplots_adjust(top=0.85,wspace=0.1)
        if objname != '': #if the object name is given
            fig.savefig(outdir+'{}_reference_spectrum'.format(objname)+'.'+format,dpi=300,bbox_inches='tight')
        else:
            fig.savefig(outdir+'reference_spectrum'+'.'+format,dpi=300,bbox_inches='tight')
        
        if verbose:
            plt.show()
        else:
            plt.close()
        #fig.clf() #clear the figure from the memory
    else:
        reference_spectrum=pd.read_csv(reference_spectrum).iloc[:,1].values #load the reference spectrum from the csv file   
        reference_rms=pd.read_csv(reference_rms).iloc[:,0].values #load the reference rms from the csv file   
        
    #REGRID THE VELOCITY FIELD IF NECESSARY#
    vfield,vfield_header=__load(__read_string(vfield,'vfield')) #read and load the velocity field
    if vfield.shape != data.shape[1:3]: #if the velocity field has a different shape then the data
        if verbose:
            warnings.warn('The spatial sizes of the data and the velocity field are different. Starting the regrid of the velocity field:')
            print('Regridding the velocity field to the data spatial resolution\n----------------------------------------------')
        regrid_size=vfield.shape[0]//data.shape[1] #calculate how much to regrid
        if regrid_size<1: #if the regrid size is less than 1 means it must be upscaled
            regrid_header=WCS(header).dropaxis(2).to_header() #the regridded header is the one of the cube without the spectral axis
            regrid_header['NAXIS1']=data.shape[1] #redefine the x-axis
            regrid_header['NAXIS2']=data.shape[2] #redefine the y-axis
            regrid_header['BUNIT']=vfield_header['BUNIT'] #store the velocity units
        else:
            regrid_header=WCS(vfield_header)[::regrid_size,::regrid_size].to_header() #create the regridded header
            regrid_header['NAXIS1']=vfield.shape[0]//regrid_size #redefine the x-axis
            regrid_header['NAXIS2']=vfield.shape[1]//regrid_size #redefine the y-axis
            regrid_header['BUNIT']=vfield_header['BUNIT'] #store the velocity units
        regrid_header['NAXIS']=2 #number of axes
        shape_out=tuple([regrid_header['NAXIS{0}'.format(i + 1)] for i in range(regrid_header['NAXIS'])][::-1]) #define the shape of the regridded field
        vfield=spatial_regrid((vfield,WCS(vfield_header)),WCS(regrid_header),shape_out=shape_out,order='bilinear')[0] #regrid the data
            
        if diagnostic:
            if verbose:
                print('Writing the regridded velocity field\n----------------------------------------------')
            hdu=fits.PrimaryHDU(vfield.astype('float32'),header=regrid_header) #create the primary HDU
            hdul=fits.HDUList([hdu]) #make the HDU list
            if objname != '': #if the object name is given
                hdul.writeto(diagnostic_outdir+'{}_regrid_vfield.fits'.format(objname),overwrite=True) #write the data into a fits
            else:
                hdul.writeto(diagnostic_outdir+'regrid_vfield.fits',overwrite=True) #write the data into a fits 
    
    #SHUFFLE THE DATA TO ALIGN THE CUBE#
    if verbose:
        print('Aligning the spectra\n----------------------------------------------')
    if specunits == 'm/s': #if the spectral units are m/s
        data=cubedo(cubedo=data,operation='shuffle',vfield=vfield,v0=v0*1000,spectralres=spectralres*1000,verbose=True)
    else:
        data=cubedo(cubedo=data,operation='shuffle',vfield=vfield,v0=v0,spectralres=spectralres,verbose=True)
    
    if diagnostic:
        if verbose:
            print('Writing the shuffled cube\n----------------------------------------------')
        header['CRPIX3']=(nchan//2)+1 #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred. +1 is needed for account the stupid python 0-counting
        header['CRVAL3']=0. #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred
        hdu=fits.PrimaryHDU(data.astype('float32'),header=header) #create the primary HDU
        hdul=fits.HDUList([hdu]) #make the HDU list
        if objname != '': #if the object name is given
            hdul.writeto(diagnostic_outdir+'{}_shuffled_cube.fits'.format(objname),overwrite=True) #write the data into a fits
        else:
            hdul.writeto(diagnostic_outdir+'shuffled_cube.fits',overwrite=True) #write the data into a fits 
            
    #REDEFINE THE SPECTRAL AXIS#
    v0=(nchan//2)*spectralres #redefine the starting velocity
    if spectralres>0: #if the spectral resolution is positive
        v=np.arange(v0,v0+nchan*spectralres,spectralres) #redefine the spectral axis
    else:
        v=-np.arange(v0,v0-nchan*spectralres,-spectralres) #redefine the spectral axis
    if len(v) > nchan: #!! sometimes an additional channel is created. For the moment, this is a workaround
        v=v[:-1]
    
    #SETUP THE EMISSION RANGE#
    emission_width=kwargs['emission_width'] if 'emission_width' in kwargs else 200000 #store the velocity range to exclude for rms calculation
    if emission_width!=0: #if the emission width is given
        if specunits == 'm/s': #if the spectral units are m/s
            emission_width/=1000 
            emission_width/=2 #half the emission width
            chmin=np.argmin(np.abs(v-emission_width)) #channel corresponding to -emission_width/2 km/s
            chmax=np.argmin(np.abs(v+emission_width)) #channel corresponding to +emission_width/2 km/s  
    
    #ASSIGN THE PIXEL TO THE CORRECT REGION#
    if verbose:
        print('Assigning pixel to each stacking region\n----------------------------------------------')
    rmin/=np.abs(pixelres) #convert the max radius into pixel
    rmax/=np.abs(pixelres) #convert the max radius into pixel
    xvalid,yvalid=__assign_to_cones(data,ncones,pa,inc,rmin,rmax,np.abs(pixelres),x0,y0) #assign the pixel to the cones     
    
    #PREPARE THE FIGURES FOR THE RESULT#
    wcs=WCS(header).dropaxis(2) #store the wcs
    xmin=round(x0-rmax) #store the min x-coordinate
    xmax=round(x0+rmax) #store the max x-coordinate
    ymin=round(y0-rmax) #store the min y-coordinate
    ymax=round(y0+rmax) #store the max y-coordinate  
    xlim=(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin)) #set the xlim
    ylim=(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin)) #set the ylim
    
    fig1=plt.figure(figsize=(6,6)) #create the figure for the stacked positions
    if objname!='': #if the object name is given
        fig1.suptitle('{} stacked positions'.format(objname),fontsize=24) #add the title
    else:
        fig1.suptitle('Stacked positions',fontsize=24) #add the title
    ax1=fig1.add_subplot(111,projection=wcs) #create the subplot for the stacked pixels
    ax1.imshow(np.nansum(data,axis=0),origin='lower',aspect='equal',cmap='gray_r',vmin=0) #plot the image
    ax1.tick_params(direction='in') #change the ticks of the axes from external to internal
    ax1.set_xlim(xlim) #set the xlim
    ax1.set_ylim(ylim) #set the ylim
    ax1.set_xlabel('RA') #set the x-axis label
    ax1.set_ylabel('DEC') #set the y-axis label  
    
    nrows=ncones #number of rows in the plot
    ncols=2 #number of columns in the plot    
    fig2=plt.figure(figsize=(6*ncols,6*nrows)) #create the figure for the stacked spectra
    if objname!='': #if the object name is given
        fig2.suptitle('{} stacking result'.format(objname),fontsize=24) #add the title
    else:
        fig2.suptitle('Stacking result',fontsize=24) #add the title
        
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color'][:ncones] #store the color cycle for assign the correct color to the plots
    
    #DO THE STACKING#
    if verbose and diagnostic:
        warnings.warn("you activated the diagnostic mode. Stacking may take a while and stores large quantity of plots in your disk! If you are using the Jupyter notebook with '%matplotlib inline' there will be a serious memory leak. To fix it, add 'import matplotlib' and matplotlib.use('agg') before running the stacking function.")
    for i in range(ncones):
        if diagnostic:
            plots_outdir=diagnostic_outdir+'region {}/'.format(i+1)
        else:
            plots_outdir=None
        s1=ax1.scatter(xvalid[i],yvalid[i],marker='.') #add the stacked pixels to the plot
        if verbose:
            print('--------------------------------------------------------------------------------------------\n                               STACKING STARTED ON REGION {}              \n--------------------------------------------------------------------------------------------\n'.format(i+1))
        spectrum,rms,_=__stack(data,xvalid[i],yvalid[i],ignore=[chmin,chmax],flip=flip,diagnostic=diagnostic,v=v,fluxunits=fluxunits,color=colors[i],outdir=plots_outdir,format='.jpg') #do the stacking
        ylim=[np.nanmax(spectrum)*1.1,np.nanmin(spectrum)*1.1] #store the ylim of the plot

        #SOURCE FINDING#
        if verbose:
            print('Running the source finder with the following parameters:\nFlux threshold: {}*rms\nSmoothing kernel(s): {} channel(s)\nLinker kernel(s): {} channel(s)\nReliability threshold: {}\n----------------------------------------------'.format(rms_threshold,smooth_kernel,link_kernel,rel_threshold))
        density_spectrum=spectrum/rms[-1] #define the flux density spectrum
        threshold=rms_threshold #store the source finding threshold
        if smooth_kernel is None or smooth_kernel==1:
            mask=__source_finder(density_spectrum,threshold) #run the source finding
        else:
            mask=density_spectrum.copy()*0 #initialize the spectral mask
            for ker in smooth_kernel: #for each smoothing kernel
                if ker==1:
                    mask=__source_finder(density_spectrum,threshold) #run the source finding
                else:
                    dummy=density_spectrum.copy() #create a a copy of the original spectrum
                    dummy[mask>0]=threshold*0.9 #replace the already detected channels with a value equal to the rms. Now it is 'threshold' because the spectrum is then multiplied by the rms in the next step
                    kernel=conv.Box1DKernel(ker) #define the smoothing boxcar kernel
                    smooth_spectrum=conv.convolve(dummy*rms[-1],kernel) #smooth the spectrum
                    dummy=smooth_spectrum.copy() #create a copy of the smoothed spectrum
                    dummy[chmin:chmax+1]=np.nan #blank the galactic emission
                    smooth_rms=np.sqrt(np.nanmean(dummy**2)) #calculate the rms
                    smooth_spectrum/=smooth_rms #divide by the new rms
                    mask=np.add(mask,__source_finder(smooth_spectrum,threshold)) #run the source finding
        
        #LINKER#
        if verbose:
            print('Linking the detected lines\n----------------------------------------------')
        sources_id=__source_linker(density_spectrum,mask,link_kernel) #run the linker

        #SOURCES CATALOGUE#
        if verbose:
            print('Creating the sources catalogue\n----------------------------------------------\n----------------------------------------------')
        pos,neg=__get_sources_catalogue(density_spectrum,sources_id) #create the source catalogue
        if verbose:
            print('Total sources: {}\nPositive sources: {}\nNegative sources: {}\n----------------------------------------------\n----------------------------------------------'.format(len(pos)+len(neg),len(pos),len(neg)))
            
        #RELIABILITY#
        if rel_threshold>0: #if the reliability must be computed
            if verbose:
                print('Rejecting unreliable sources\n----------------------------------------------')
            plt.ioff()
            total_rel=__get_reliability(pos,neg,snrmin,rel_threshold,outdir=outdir,objname=objname,outname='reliability_result_region_{}'.format(i+1),plot_format=format,ctr_width=ctr_width)
            plt.ion()
            if verbose:
                print('Detected reliable lines for region {}: {}'.format(i+1,len(total_rel[total_rel>=rel_threshold])))
            non_rel_idx=np.where(total_rel<0.9)[0]+1 #find the index of the non reliable sources. +1 is needed because python counts from 0. total_rel is an array of length equal to the number of positive sources. The first element is the first positive source, which has an index of 1.
            for idx in non_rel_idx: #blank the non reliable sources
                sources_id[sources_id==idx]=np.nan #sources_id contains positive values (indexes of the positive sources) and negative values. I know the indexes of the non-reliable positive values, hence I can blank them
            sources_id[sources_id<0]=np.nan #blank the negative sources
            sources_id[~np.isnan(sources_id)]=1 #set to 1 the positive sources
        else:
            sources_id=None #set to None to not plot the reliable/unreliable sources in the next step
                
        #PLOT THE RESULT#
        fig2=__plot_stack_result(v,spectrum,rms,reference_rms[:len(rms)],fluxunits,mask=sources_id,chmin=chmin,chmax=chmax,nrows=nrows,ncols=ncols,color=colors[i],aligned=True,ylim=ylim,idx=2*i+1)
        
        print('--------------------------------------------------------------------------------------------\n                               STACKING FINISHED ON REGION {}              \n--------------------------------------------------------------------------------------------\n'.format(i+1))
    
    fig2.subplots_adjust(top=0.95,hspace=0.25,wspace=0.05)
    if objname != '': #if the object name is given
        fig1.savefig(outdir+'{}_stacked_positions'.format(objname)+'.'+format,dpi=300,bbox_inches='tight')
        fig2.savefig(outdir+'{}_stacking_results'.format(objname)+'.'+format,dpi=300,bbox_inches='tight')
    else:
        fig1.savefig(outdir+'stacked_positions'+'.'+format,dpi=300,bbox_inches='tight')
        fig2.savefig(outdir+'stacking_results'+'.'+format,dpi=300,bbox_inches='tight')

    if verbose:
        plt.show()
    else:
        plt.close('all')
                      
#############################################################################################
def velfi(vfield='',radii=None,vrot=None,pa=None,inc=None,vrad=None,vsys=None,vcenter=None,extend_only=False,correct=False,velfioutdir='',velfioutname='',**kwargs):
    """Compute a syntethic velocity field or (optional) extend a given velocity field.
    
    Args:
        vfield (string): name or path+name of the measured velocity field
        radii (float/array-like): radii in pixunits at which the rotation velocity is measured
        vrot (float/array-like): rotation velocities (subtracted from the systemic velocities) in specunits for the radii
        pa (float/array-like): position angle of the object in degrees (counter-clockwise from the North to the most receding velocity)
        inc (float/array-like): inclination of the object in degrees (0 deg means face-on)
        vrad (float/array-like): expansion velocity for the radii
        vsys (float/array-like): systemic velocity for the radii
        center (array-like): comma-separated value for the x and y coordinates of the rotation center in pixel
        extend_only (boolean): extend a given velocity field (True) or write a new one from scratch (False)
        correct (boolean): correct the input rotation velocities for the inclination angles (True) or not (False)
        velfioutdir (string): output folder name
        velfioutname (string): output file name
        
    Kwargs:
        path (string): path to the fits velocity field if the vfield is a name and not a path+name
        pixelres (float): pixel resolution of the data in pixunits
        pixunits (string): string with the spatial units. Accepted values:
            - deg
            - armin
            - arcsec
        specunits (string): string with the spectral units. Accepted values:
            - km/s
            - m/s
            - Hz   
        verbose (bool): option to print messages and plot to terminal if True (Default: False)
             
    Returns:
        Sythetic velocity field as fits file
    
    Raises:
        ValueError: If mandatory inputs are missing
        ValueError: If no path to the velocity field is provided
        ValueError: If wrong spatial units are provided
        ValueError: If wrong spectral units are provided
    """
    #CHECK THE INPUT#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    if vfield == '' or vfield is None: #if a moment 1 map is not given
        raise ValueError('ERROR: velocity field is not set: aborting!')
    center=vcenter #store the rotational center
    inputs=np.array([radii,vrot,pa,inc,vrad,vsys,center],dtype='object') #store the mandatory inputs
    inputsnames=['radii','vrot','pa','inc','vrad','vsys','center'] #store the mandatory input names
    if None in inputs: #check if one or more mandatory inputs are missing
        raise ValueError('ERROR: one or more mandatory inputs are missing ({}): aborting!'.format([inputsnames[i] for i in range(len(inputs)) if inputs[i]==None]).replace("'",""))
    if vfield[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the velocity field is set: aborting!')
            else:
                vfield=kwargs['path']+vfield
        else:
            raise ValueError('ERROR: no path to the velocity field is set: aborting!')
    if len(center)!=2: #check that the rotation center has the correct format
        raise ValueError('ERROR: Please provide the rotation center in the format [x0,y0]. Aborting!')
    else: #store the rotation center from the input
        x0=center[0]-1 #convert the x-center into 0-indexing
        y0=center[1]-1 #convert the y-center into 0-indexing
    outdir=velfioutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] if 'path' in kwargs else os.getcwd()+'/' #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder     
    outname=velfioutname #store the output name from the input parameters
    if outname == '' or outname is None:  #if the outname is empty
        outname=vfield.replace('.fits','_velfi.fits') #the name is the velocity field plus _velfi
    if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    #CHECK THE KWARGS#
    pixunits=kwargs['pixunits'] if 'pixunits' in kwargs else None
    specunits=kwargs['specunits'] if 'specunits' in kwargs else None
    pixelres=kwargs['pixelres'] if 'pixelres' in kwargs else None
    if pixunits is not None: #if the spatial units are given
        if pixunits not in ['deg','arcmin','arcsec']: #if wrong spatial units are given
            raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    if specunits is not None: #if the spectral units are given
        if specunits not in ['km/s','m/s','Hz']: #if wrong spectral units are given
            raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')
        
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
    def radius(i,x,y): #define the elliptical radius
        xr=(-x*np.sin(pa[i])+y*np.cos(pa[i])) #x-radius is the same of a circle
        yr=(-x*np.cos(pa[i])-y*np.sin(pa[i]))/np.cos(inc[i]) #y-radius is the same of a circle but reduced by the inclination angle
        return(radii[i]-np.sqrt(xr**2+yr**2)) #return the distance between the radius[i] and the elliptical radius of the point
    
    with fits.open(vfield) as V: #open the velocity field fits file
        field=V[0].data #store the data
        header=V[0].header #store the header
        
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
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if np.isnan(result[i,j]): #if the output in (i,j) is empty
                x=(j-x0)*pixelres #convert the x-position of the pixel w.r.t. the center in arcsec
                y=(i-y0)*pixelres #convert the y-position of the pixel w.r.t. the center in arcsec
                r=np.sqrt(x**2+y**2) #calculate the distance from the center in arcsec
                if r <= radii[nradii-1]: #if the distance is less or equal the outer radius
                    d1=radius(0,x,y) #calculate the radius of (x,y) compared to the innermost radius
                    d2=radius(nradii-1,x,y) #calculate the radius of (x,y) compared to the outermost radius
                    if d1*d2 <= 0: #if one of the distance is negative, means that the points resides between the innermost and outermost radius, hence, it is okay to calculate the rotation velocity at that point
                        for k in np.arange(1,nradii+1,1): #find the radius[i] corresponding to that point
                            d2=radius(k,x,y) #keep calculate the distance between the radius of (x,y) and the k-radius
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
    
    #WE DO THE PLOT#
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
    fig.suptitle('Velfi output',fontsize=24) #add the title
    
    ax=fig.add_subplot(nrows,ncols,1,projection=wcs) #create the subplot
    ax.imshow(field,origin='lower',vmin=vmin,vmax=vmax,aspect='equal',cmap='jet') #plot the original field
    ax.tick_params(direction='in') #change the ticks of the axes from external to internal
    ax.set_xlim(xlim) #set the xlim
    ax.set_ylim(ylim) #set the ylim
    ax.set_xlabel('RA') #set the x-axis label
    ax.set_ylabel('DEC') #set the y-axis label
    ax.set_title('Original field',pad=20,fontsize=20)
    
    ax=fig.add_subplot(nrows,ncols,2,projection=wcs) #create the subplot
    ax.imshow(result,origin='lower',vmin=vmin,vmax=vmax,aspect='equal',cmap='jet') #plot the original field
    ax.tick_params(direction='in') #change the ticks of the axes from external to internal
    ax.set_xlim(xlim) #set the xlim
    ax.set_ylim(ylim) #set the ylim
    ax.set_xlabel('RA') #set the x-axis label
    ax.set_ylabel('DEC') #set the y-axis label
    ax.set_title('Extended Field',pad=20,fontsize=20)
    ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
    
    fig.subplots_adjust(left=0.17, bottom=0.1, right=0.93, top=0.99, wspace=0.0, hspace=0.0) #fix the position of the subplots in the figure 
    fig.savefig(outname.replace('fits','pdf'),dpi=300,bbox_inches='tight') #save the figure
    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    else:
        plt.close()
                        
    hdu=fits.PrimaryHDU(result,header=header) #create the primary HDU
    output=fits.HDUList([hdu]) #make the HDU list
    output.writeto(outname,overwrite=True) #save the syntethic velocity field
        
################################ --- ANCILLARY FUNCTIONS --- ################################
#############################################################################################
def converttoHI(data,fluxunits='Jy/beam',beamarea=None,pixunits='deg',spectralres=None,specunits='m/s'):
    """Convert the flux in an array into HI column density.

    Args:
        data (string/array): name or path+name of the fits data, or array of the data
        fluxunits (string): string with the flux units
        beamarea (float): area of the beam in pixunits^2
        pixunits (string): string with the spatial units. Accepted values:
            - deg
            - armin
            - arcsec
        spectralres (float): data spectral resolution in specunits
        specunits (string): string with the spectral units. Accepted values:
            - km/s
            - m/s
            - Hz

    Returns:
        Flux array converted into HI column density
        
    Raises:
        ValueError: If wrong spatial units are provided
        ValueError: If wrong spectral units are provided
        ValueError: If no spectral resolution and beam area are provided when given the data as array
        ValueError: If no beam information is provided
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
    else: #if the data is given as a numpy array
        if spectralres is None or beamarea is None: #if no additional information are given, abort
            raise ValueError('ERROR: Please provide the spectral resolution and the beam area. Aborting!')
        darray=data #store the data   
     
    #WE CHECK THE REQUIRED INFORMATION FOR FLUX CALCULATION#
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
    """Given a cosmological model and the redshift of an object, it calculates the relevant cosmological  quantitities (distances, age of Unvierse, light travel time, ...) for that object.

    Args:
        z (float): redshift of the object
        H0 (float): Hubble Parameter in km/s*Mpc
        omega_matter (float): Omega for the matter
        omega_vacuum (float): Omega for the vacuum
        verbose (bool): option to print messages to terminal if True 

    Returns:
        Python dictionary with the cosmological quantities of a given object
        
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
        print('For H₀={:.1f}, \u03A9ₘ={:.1f}, \u03A9ᵥ={:.1f}, z={}:'.format(H0,omega_matter,omega_vacuum,z))
        print('It is now {:.1f} Gyr since the Big Bang.'.format(age_Gyr))
        print('The age at redshift z was {:.1f} Gyr.'.format(zage_Gyr))
        print('The light travel time was {:.3f} Gyr.'.format(DTT_Gyr))
        print('The comoving radial distance, which goes into Hubbles law, is {:.1f} Mpc or {:.3f} Gly'.format(DCMR_Mpc,DCMR_Gyr))
        print('The comoving volume within redshift z is {:.3f} Gpc^3.'.format(V_Gpc))
        print('The angular size distance Dₐ is {:.1f} Mpc or {:.3f} Gly.'.format(DA_Mpc,DA_Gyr))
        print('This gives a scale of {:.3f} kpc/arcsec.'.format(kpc_DA))
        print('The luminosity distance Dₗ is {:.1f} Mpc or {:.3f} Gly.'.format(DL_Mpc,DL_Gyr))
        print('The distance modulus, m-M, is {:.1f}'.format(5*np.log10(DL_Mpc*10.**6)-5))

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
    """Compute the HI mass of a given array.

    Args:
        data (string/array): name or path+name of the fits data, or array of the data
        fluxunits (string): string with the flux units
        beamarea (float): area of the beam in pixunits^2
        pixelres (float): pixel resolution of the data in pixunits
        pixunits (string): string with the spatial units. Accepted values:
            - deg
            - armin
            - arcsec
        pbcorr (bool): apply the primary beam correction if True. Note that in that case you must supply a beam cube
        distance (float): distance of the object in Mpc
        box (array-like): region to compute the flux as [xmin,xmax,ymin,ymax]

    Kwargs:
        beamcube (string/ndarray): name or path+name of the fits beam cube if pbcorr is True
        path (string): path to the beam cube if the beamcube is a name and not a path+name
        verbose (bool): option to print messages to terminal if True   
        
    Returns:
        HI mass over the given array with the uncertainty as float: mass,error = getHImass()
        
    Raises:
        ValueError: If wrong spatial units are provided
        ValueError: If no spatial resolution and beam area is given when data type is array
        ValueError: If data are not given as path-to-fits file or as numpy.ndarray
        ValueError: If no beam information are found
        ValueError: If no pixel resolution is found
        ValueError: If no data units are found
        ValueError: If data units are not km/s or m/s
        ValueError: If distance is set
        ValueError: If size of spatial box is not 4
    """
    #CHECK THE INPUT#
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    if pixunits not in ['deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    if type(data) is str: #if the data as a path-to-file
        with fits.open(data) as Data: #open the fits file
            header=Data[0].header #store the header
            darray=Data[0].data #store the data
    elif type(data) is np.ndarray: #if the data is given as a numpy array
        if beamarea is None or pixelres is None: #if no additional information are given, abort
            raise ValueError('ERROR: Please provide the spatial resolution and the beam area. Aborting!')
        darray=data #store the data
    else:
        raise ValueError('ERROR: Please provide the data as path-to-fits file or as numpy.ndarray. Aborting!')
    darray[darray<0]=0 #remove the negatives if any (like noise artifacts sneaked in the moment map    
    
    #WE CHECK THE REQUIRED INFORMATION FOR FLUX CALCULATION#
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
        raise ValueError("ERROR: Wrong flux units provided: {}. They must contain 'm/s' or 'km/s'. Aborting!".format(fluxunits))
    #------------   PB CORRECTION     ------------# 
    if pbcorr: #if the primary beam correction is applied
        #------------   IS IN THE INPUT?     ------------# 
        if 'beamcube' in kwargs: #if the beamcube is in kwargs
            if kwargs['beamcube'] == '' or kwargs['beamcube'] is None: #if no beam cube is provided
                if verbose:
                    warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')
            else:
                beamcube=kwargs['beamcube'] #store the data cube path from the input parameters.
                #------------   IS IN THE INPUT, BUT THE PATH?     ------------# 
                if beamcube[0] != '.': #if the primary beam cube name start with a . means that it is a path to the cube (so differs from path parameter)
                    if 'path' in kwargs: #if the path to the beam cube is in kwargs
                        if kwargs['path'] == '' or kwargs['path'] is None:
                            if verbose:
                                warnings.warn('You have not provided a path to the beam cube. Cannot apply primary beam correction!')
                        else:
                            indir=kwargs['path']
                            beamcube=indir+beamcube
                            #WE CAN NOW DO THE PB CORRECTION#
                            with fits.open(beamcube) as pb_cube: #open the primary beam cube
                                pb_slice=pb_cube[0].data[np.array(pb_cube[0].shape[0]/2).astype(int)] #extract the central plane
                                darray=darray/pb_slice #apply the pb correction
                    elif verbose:
                        warnings.warn('You have not provided a path to the beam cube. Cannot apply primary beam correction!')
                else:
                    #WE CAN NOW DO THE PB CORRECTION#
                    with fits.open(beamcube) as pb_cube: #open the primary beam cube
                        pb_slice=pb_cube[0].data[np.array(pb_cube[0].shape[0]/2).astype(int)] #extract the central plane
                        darray=darray/pb_slice #apply the pb correction
        elif verbose:
            warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')

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
    xmin,xmax,ymin,ymax=__load_box(data,box) #load the spatial box
    
    #NOW WE CALCULATE THE MASS# 
    HI_mass=(2.35*10**5)*(distance**2)*pixelsize*np.nansum(darray[ymin:ymax,xmin:xmax])/beamarea #compute the HI masss
    error=HI_mass/10 #the error on the mass is equal to the calibration error (typically of 10%)
    if verbose: #if the print-to-terminal option is True
        print('Total HI mass: {:.1e} solar masses'.format(HI_mass))
    return HI_mass,error #return the mass and its error

#############################################################################################
def flux(data,fluxunits='Jy/beam',beamarea=None,pixelres=None,pixunits='deg',spectralres=None,specunits='m/s',box=None,verbose=False):
    """Calculate the flux of an array.

    Args:
        data (string/array): name or path+name of the fits data, or array of the data
        fluxunits (string): string with the flux units
        beamarea (float): area of the beam in pixunits^2
        pixelres (float): pixel resolution of the data in pixunits
        pixunits (string): string with the spatial units. Accepted values:
            - deg
            - armin
            - arcsec
        spectralres (float): data spectral resolution in specunits
        specunits (string): string with the spectral units. Accepted values:
            - km/s
            - m/s
            - Hz
        box (array-like): region to compute the flux as [xmin,xmax,ymin,ymax]
        verbose (bool): option to print messages to terminal if True
        
    Returns:
        Flux over the given array
        
    Raises:
        ValueError: If wrong spatial units are provided
        ValueError: If wrong spectral units are provided
        ValueError: If no spatial resolution, spectral resolution, and beam area is given when data type is array
        ValueError: If data are not given as path-to-fits file or as numpy.ndarray
        ValueError: If no beam information are found
        ValueError: If no pixel resolution is found
        ValueError: If no data units are found
        ValueError: If no spectral resolution is found
        ValueError: If size of spatial box is not 4
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
    elif type(data) is np.ndarray: #if the data is given as a numpy array
        if spectralres is None or beamarea is None or pixelres is None: #if no additional information are given, abort
            raise ValueError('ERROR: Please provide the spatial and spectral resolution, and the beam area. Aborting!')
        darray=data #store the data
    else:
        raise ValueError('ERROR: Please provide the data as path-to-fits file or as numpy.ndarray. Aborting!')
    #------------   REMOVE THE NEGATIVE VALUES     ------------# 
    darray[darray<0]=0 #set to 0 the negative flux values    
        
    #WE CHECK THE REQUIRED INFORMATION FOR FLUX CALCULATION#
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
    if units is None: #if no units for the data are provided
        if type(data) is str: #if the data as a path-to-file
            if 'BUNIT'.casefold() in header: #check the data units keyword
                units=header['BUNIT'.casefold()] #store the data units
            else:
                raise ValueError('ERROR: No data units are found: aborting!')
        else:
            raise ValueError('ERROR: No data units are found: aborting!')
    if 'm/s'.casefold() in units or 'Hz'.casefold() in units: #if the units contains the spectral information
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
    xmin,xmax,ymin,ymax=__load_box(data,box) #load the spatial box
    
    #NOW WE CALCULATE THE FLUX# 
    flux=np.nansum(darray[ymin:ymax,xmin:xmax])*spectralres*pixelsize/beamarea #calculate the flux
    
    if verbose: #if the print-to-terminal option is True
        print('The flux is {:.1e} {}'.format(flux,units))
    return flux #return the flux
            
#############################################################################################
def create_config(name='default_parameters'):
    """Create a default configuration file.

    Args:
        name (string): name of the configuration filea
        
    Returns:
        Default configuration file saved as 'name'.par
        
    Raises:
        None
    """
    with open(name+'.par',mode='w') as configfile: #equivalent of saying configfile=open()
        configfile.write(";configuration file for nhicci. Use ';' or '#' to comment the parameters you don't need/know\n"
            '\n'
            '[GENERAL]\n'
            'verbose		=	   #if True, all plots and ancillary information will be printed to terminal [True,False] (default: False)\n'
            '\n'
            '[INPUT]\n'
            'path		=	   #path to the working directory (generally where the data are stored). Defatult: current directory\n'
            '\n'
            '[COSMOLOGY]\n'
            'HO		    =	   #Hubble parameter (default: 69.6)\n'
            'Omegam	    =	   #Omega matter (default: 0.286)\n'
            'Omegav	    =	   #Omega vacuum (default: 0.714)\n'
            '\n'
            '[GALAXY]\n'
            'objname		=	   #name of the object (default: None)\n'
            'distance	=	   #distance of the object in Mpc (default: None)\n'
            'redshift	=	   #redshift of the object (default: None)\n'
            'asectokpc	=	   #arcsec to kpc conversion (default: None)\n'
            'vsys		=	   #systemic velocity of the object in km/s (default: None)\n'
            'pa			=	   #position angle of the object in deg (default: None)\n'
            'inc			=	   #inclination angle of the object in deg (default: None)\n'
            '\n'
            '[CUBEPAR]\n'
            'pixunits	=	   #spatial units (default: None)\n'
            'specunits	=	   #spectral units (default: None)\n'
            'fluxunits	=	   #flux units (default: None)\n'
            'pixelres	=	   #pixel resolution in pixunits (default: None)\n'
            'spectralres	=	   #spectral resolution in specunits (default: None)\n'
            'rms         =	   #root-mean-square value in fluxunits (default: None)\n'
            '\n'
            '[BEAM]\n'
            'bmaj		=	   #beam major axis in arcsec (default: None)\n'
            'bmin		=	   #beam minor axis in arcsec (default: None)\n'
            'bpa         =	   #beam position angle in arcsec (default: None)\n'
            '\n'
            '[MODE]\n'
            'mode		=	   #operation mode of the pipeline (default: None)\n'
            '\n'
            '[CORRECTION]\n'
            'pbcorr		=	   #apply the primary beam correction [True,False] (default: False)\n'
            '\n'
            '[FITS]\n'
            'datacube	=	   #name of the fits file of the data cube including .fits (default: None)\n'
            'beamcube	=	   #name of the fits file of the beam cube including .fits (default: None)\n'
            'maskcube	=	   #name of the fits file of the mask cube including .fits (default: None)\n'
            'mask2d		=	   #name of the fits file of the 2D mask including .fits (default: None)\n'
            'channelmap	=	   #name of the fits file of the channel map including .fits (default: None)\n'
            'modelcube	=	   #name of the fits file of the model cube including .fits (default: None)\n'
            'mom0map		=	   #name of the fits file of the moment 0 map including .fits (default: None)\n'
            'mom1map		=	   #name of the fits file of the moment 1 map including .fits (default: None)\n'
            'mom2map		=	   #name of the fits file of the moment 2 map including .fits (default: None)\n'
            'vfield      =      #name of the fits file of the velocity field including .fits (default:None)\n'
            '\n'
            '[CHANMAP]\n'
            'chanmin		=	   #starting channel to plot in the channel map (default: 0)\n'
            'chanmax		=	   #ending channel to plot in the channel map (default: None)\n'
            'chansep		=	   #channel separation to plot in the channel map (chamin,chanmin+chansep,chanmin+2*chansep,...,chanmax) (default: 1)\n'
            'box         =      #comma-separated pixel edges of the box to extract the channel map in the format [xmin,xmax,ymin,ymax] (default: None)\n'
            'nsigma		=	   #rms threshold to plot the contours (lowest contours will be nsigma*rms) (default: 3)\n'
            'usemask		=	   #use a mask in the channel map [True,False] (default: False)\n'
            'outputdir	=	   #output directory to save the channel map plot. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the channel map plot (including file extension *.jpg,*.png,*.pdf,...) (default: None)\n'
            '\n'
            '[CUBEDO]\n'
            'datacube 	=	   #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'operation	=	   #operation to perform on the cube [blank,clip,crop,cut,extend,mirror,mom0,shuffle,toint] (default: None)\n'
            "chanmin		=	   #first channel for the operations 'blank,cut,mom0' (default: 0)\n"
            "chanmax		=	   #last channel for the operations 'blank,cut,mom0' (default: None)\n"
            "box         =      #comma-separated pixel edges of the box to extract for operation 'cut' in the format [xmin,xmax,ymin,ymax] (default: None)\n"
            "addchan		=	   #number of channels to add in operation 'extend'. Negative values add lower channels, positive add higher channels (default: None)\n"
            "value		=	   #value to assign to blank pixel in operation 'blank' (blank is np.nan) (default: blank)\n"
            "usemask		=	   #use a 2D mask in the operation 'clip' [True,False] (default: False)\n"
            'mask		=	   #name of the fits file of the 2D mask including .fits. If empty, is the same of [FITS] mask2d (default: None)\n'
            "cliplevel	=	   #clip threshold as % of the peak (0.5 is 50%) for operation 'clip' (default: 0.5)\n"
            "xrot		=	   #x-coordinate of the rotational center for operation 'mirror' (default: None)\n"
            "yrot		=	   #y-coordinate of the rotational center for operation 'mirror' (default: None)\n"
            "zrot		=	   #z-coordinate of the rotational center for operation 'mirror' default: None)\n"
            'outputdir	=	   #output directory to save the new cube. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the new cube including extension .fits. If empty, is the same of datacube (default: None)\n'
            '\n'
            '[CUBESTAT]\n'
            'nsigma		=	   #rms threshold in terms of nsigma*rms for detection limit (default: 3)\n'
            '\n'
            '[FITSARITH]\n'
            'fits1		=	   #name of reference fits file including .fits (default: None)\n'
            'fits2		=	   #name of second fits file including .fits (default: None)\n'
            'operation	=	   #operation to do between the two fits [sum,sub,mul,div] (default:None)\n'
            'outputdir	=	   #output directory to save the new cube. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the new fits file including extension .fits. If empty, is the same of fits1 (default: None)\n'
            '\n'
            '[FIXMASK]\n'
            'datacube 	=	   #name of the fits file of the reference data cube including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'maskcube	=	   #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube (default: None)\n'
            'outputdir	=	   #output directory to save the new mask. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the new mask including extension .fits. If empty, is the same of maskcube (default: None)\n'
            '\n'
            '[GAUSSFIT]\n'
            'datacube	=	   #name of the fits file of the data cube to fit including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'gaussmask	=	   #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube. The fit will be done inside the mask (default: None)\n'
            'linefwhm	=	   #first guess on the fwhm of the line profile in km/s (default: 15)\n'
            'amp_thresh	=	   #amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum (default: 0)\n'
            'p_reject	=	   #p-value threshold for fit rejection. If a best-fit as p>p_reject, it will be rejected (default: 1) \n'
            'clipping	=	   #clip the spectrum to a % of the profile peak [True,False] (default: False)\n'
            'threshold	=	   #clip threshold as % of the peak (0.5 is 50%) if clipping is True (default: 0.5)\n'
            'errors		=	   #compute the errors on the best-fit [True,False] (default: False)\n'
            'write_field	=	   #compute the best-fit velocity field [True,False] (default: False)\n'
            'outputdir	=	   #output directory to save the model cube. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the model cube including extension .fits (default: None)\n'
            '\n'
            '[GETPV]\n'
            'datacube	=	   #name of the fits file of the data cube including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'width 		=	   #width of the slice in arcsec. If not given, will be the beam size (default: None)\n'
            'points 		=	   #RA-DEC comma-separated coordinate of the slice points in decimal degree. If two are given ([x,y]), it is assumed they are the center of the slice. Else, they need to be the starting and ending coordinates ([xmin,xmax,ymin,ymax]) (default: None)\n'
            'angle 		=	   #position angle of the slice in degree. If not given, will be the object position angle (default: None)\n'
            'chanmin		=	   #starting channel of the slice (default: 0)\n'
            'chanmax		=	   #ending channel of the slice (default: None)\n'
            'outputdir	=	   #output directory to save the fits and the plot. If empty, is the same as [INPUT] path (default: None)\n'
            'fitsoutname	=	   #output name of the pv fits file including extension .fits (default: None)\n'
            'plotoutname	=	   #output name of the pv plot (including file extension *.jpg,*.png,*.pdf,...) (default: None)\n'
            'subtitle	=	   #optional subtitle for the pv plot (default: None)\n'
            'nsigma		=	   #rms threshold to plot the contours (lowest contours will be nsigma*rms) (default: 3)\n'
            '\n'
            '[PLOTMOM]\n'
            'subtitle	=	   #optional subtitle for the plot (default: None)\n'
            'outputdir	=	   #output directory to save the plot. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the plot (including file extension *.jpg,*.png,*.pdf,...) (default: None)\n'
            '\n'
            '[REMOVEMOD]\n'
            'method		=	   #method to remove the model [all,blanking,b+s,negblank,subtraction] (default: subtraction)\n'
            "threshold	=	   #flux threshold for the 'all,blanking,b+s' methods in units of cube flux (default: 0)\n"
            'outputdir	=	   #output directory to save the fits file. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the new fits file including extension .fits (default: None)\n'
            '\n'
            '[ROTCURVE]\n'
            'center		=	   #x-y comma-separated coordinates of the rotational center in pixel (default: None)\n'
            'save_csv    =      #store the output in a csv file [True,False] (default: False)\n'
            'outputdir	=	   #output directory to save the plot. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the plot (including file extension *.jpg,*.png,*.pdf,...) (default: None)\n'
            '\n'
            '[STACKING]\n'
            'center      =      #x-y comma-separated coordinates of the rotational center in pixel (default: None)\n'
            'ncones      =      #number of conic regions from which the spectra are extracted and stacked (default: None)\n'
            'radii       =      #comma-separated min and max radius from the galactic center in pixunits from which the spectra are extracted and stacked (default: None)\n'
            'diagnostic  =      #store all the diagnostic files and plots. Warning: the diagnostic might occupy large portions of the disk (default: False)\n'
            'threshold   =      #number of rms to reject flux values in the source finder (default: 3)\n'
            'smooth_ker  =      #kernel size (or comma-separated kernel sizes) in odd number of channels for spectral smoothing prior the source finding. Set to None or 1 to disable (default: None)\n'
            'link_ker    =      #minimum odd number of channels covered by a spectral line (default: 3)\n'
            'min_rel     =      #minimum value (from 0 to 1) of the reliability to consider a source reliable. Set to 0 to disable the reliability calculation (default: 0.9)\n'
            'min_snr     =      #minimum SNR of a detected line to be reliable (default: 3)\n'
            'gal_range   =      #velocity range in specunits to exclude from rms calculation in the stacked spectra. Set to 0 to compute the rms over the whole spectrum (Default: 200 km/s)\n'
            'ref_spec    =      #path to the csv file of the reference spectrum. The first column is velocity in specunits, the second column is the flux in fluxunits. If not given, it will be automatically computed (Default: None)\n'
            'ref_rms     =      #path to the csv file of the reference rms as a function of the number of stacked spectra. The firs column is the rms in fluxunits. The number of rows is the number of stacked spectra. If not given, it will be automatically computed (Default: None)\n'
            'regrid      =      #regrid the cube option (Default: False)\n'
            'regrid_size =      #how many pixel to regrid. Set to 0 to regrid to a beam (Deafult: 0)\n'
            'outputdir   =      #output directory to save the fits file. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the fits file (including file extension .fits) (default: None)\n'
            '\n'
            '[VELFI]\n'
            'radii       =      #radii in pixunits at which the rotation velocity is measured (default: None)\n'
            'vrot        =      #rotation velocities (subtracted from the systemic velocities) in specunits for the radii (default: None)\n'
            'vrad        =      #expansion velocity for the radii (default: None)\n'
            'center      =      #comma-separated value for the x and y coordinates of the rotation center in pixel (default: None)\n'
            'extend_only =      #extend a given velocity field (True) or write a new one from scratch (False) (default: False)\n'
            'correct     =      #correct the input rotation velocities for the inclination angles (True) or not (False) (default: False)\n'
            'outputdir   =      #output directory to save the fits file. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the fits file (including file extension .fits) (default: None)\n'
            '\n'
            '[PLOTSTYLE]\n'
            'ctr width	=	   #width of the contours levels (default: 2)\n'
            'format	    =	   #file format for the plots (default: pdf)')

#############################################################################################
############################### Code template for atlas plot ################################   
# 
# nrows= #number of rows in the atlas
# ncols= #number of columns in the atlas
# outname='<Insert outname here>' #name of the outfile
# 
# fig=plt.figure(figsize=(6*ncols,6*nrows))
# fig.suptitle('<Insert text here>',fontsize=24) #add the title
# 
# k=1
# fig.add_subplot(nrows,ncols,k,projection=)
# <Insert plotting function here>
# k+=1
# fig.add_subplot(nrows,ncols,k,projection=)
# <Insert plotting function here>
# k+=1
# ### Repeat above for how many plot you want ###
# ### To hide axis labels copy code below ###
# ax=plt.gca()
# ax.coords[1].set_ticklabel_visible(False) #disable y-axis labels
# ax.coords[0].set_ticklabel_visible(False) #disable x-axis labels
# 
# fig.subplots_adjust(left=0.07, bottom=0.1, right=0.97, top=0.90, wspace=0.0, hspace=0.0)      
# fig.savefig(outname,dpi=300,bbox_inches='tight') #save the figure
# plt.show()
#
################################## --- W.I.P. FUNCTIONS --- #################################
#############################################################################################
"""#function: showparams
#input: parameters = python dictionary
#description: print a python dictionary in a line by line (keyword, value)
#return: none

def showparams(parameters):
    for key, value in parameters.items():
        print(key,' : ',value)
        
#############################################################################################
#function: runsofia
#input: parameters = keyworded arguments with the necessary variables passed thourgh the parameter dictionary
        sofiaalias = sofia2 alias name          [] (string)
        sofiafile  = sofia2 parameter file name [] (string)
        sofiamode  = sofia2 running mode        [] (string)
#description: run sofia2 with the given parameter file in a bash shell
#output: none

def runsofia(sofiaalias,sofiafile):
    
    sofiaalias=parameters['sofiaalias'] #store the sofia2 alias name from the input parameters
    sofiafile=parameters['sofiafile'].split(',') #store the sofia2 parameter file from the input parameters and split if there are more files
    sofiamode=parameters['sofiamode'] #store the sofia2 running mode from the input parameters

    if mode == 's+c':
        subprocess.run(["{} {}".format(sofiaalias,sofiafile[0])],shell=True)
    elif mode == 'moments':
        subprocess.run(["{} {}".format(sofiaalias,sofiafile[1])],shell=True)
"""        
#############################################################################################
def __assign_to_cones(data,ncones,pa,inc,rmin,rmax,pixelres,x0,y0):
    """Given the parameters of the cones (number of coneas, inner radius, outer radius, x-center, y-center), the pixel resolution and the position angle and inclination of the galaxy, assign each pixel of the data within rmin and rmax into the correct cone.

    Args:
        data (ndarray): 3D array with the data
        ncones (int): number of cones
        pa (float): object position angle in degree
        inc (float): inclination of the object in degrees (0 deg means face-on)
        rmin (float): inner radius of the cone
        rmax (float): outer radius of the cone
        pixelres (float): pixel resolution of the data in arcsec
        x0 (float): x-coordinate of the center
        y0 (float): y-coordinate of the center
        
    Returns:
        lists of the x and y coordinate of the pixel in each cone
        
    Raises:
        None
    """
    #SETUP THE EXTRACTION REGIONS PROPERTIES#
    angles=np.arange(-180*((ncones-1)/ncones),180,360/ncones) #we define the angles from -180 to 180, since np.arctan2 uses this definition
    angles=np.radians(angles) #convert the angles into radians
    pa=np.radians(pa-180) #convert the position angle into radians
    inc=np.radians(inc) #convert the inclination angle into radians
    xmin=round(x0-rmax) #store the min x-coordinate
    xmax=round(x0+rmax) #store the max x-coordinate
    ymin=round(y0-rmax) #store the min y-coordinate
    ymax=round(y0+rmax) #store the max y-coordinate    
            
    #to check if a point belongs to a given cone or not, we compare the tan of the angle defined by the line connecting the point and the rotation center and the x-axis.
    #If that angles is in between angle[i] and angle[i+1], the point belongs to the i-cone.
    #For each pixel in the square (xmin:xmax,ymin:ymax) we check if it resides whitin rmin and rmax.
    #If that is true, then we assign the pixel to the correct region by iterating to the angles until it ends up in the correct region
        
    xvalid=[] #initialize the x-coordinates list of the pixel in the cones
    yvalid=[] #initialize the y-coordinates list of the pixel in the cones
    for i in range(ncones): #append an empty list to create a list of ncones list. We will place in the i-list the pixel belonging to the i-region
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
                                if angles[i] <= np.arctan2(j-y0,k-x0) <= angles[idx]: #the point angle must be in between angle[i] and angle[i+1]
                                    valid=True
                            elif angles[i] > 0 and angles[idx] < 0: #if angle[i] is positive and angle[i+1] is negative
                                if angles[i] <= np.arctan2(j-y0,k-x0) or np.arctan2(j-y0,k-x0) <= angles[idx]: #the point angle must be higher than angle[i] but lower than angle[i+1]
                                    valid=True
                            elif angles[i] < 0 and angles[idx] < 0: #if both angles are negative
                                if angles[idx] >= np.arctan2(j-y0,k-x0) >= angles[i]: #the point angle must be in between angle[i+1] and angle[i]
                                    valid=True
                            if valid:
                                xvalid[i].append(k) #append the x-coordinate
                                yvalid[i].append(j) #append the y-coordinate
                                pbar.update(1)
                                break
    return xvalid,yvalid
############################################################################################# 
def __get_reliability(pos_cat,neg_cat,snrmin,rel_threshold,**kwargs):
    """Calculate the reliability of the positive sources in a catalogue, by comparing their density in each position of the parameter space with the density of negative sources in the same position. All the fluxes must be divided by the rms of the data prior calling this function.

    Args:
        pos_cat (array-like): array-like of array where each array contains the flux per channel of a positive source
        neg_cat (array-like): array-like of array where each array contains the flux per channel of a negative source
        snrmin (float): minimum SNR of a detected line to be reliable
        rel_threshold (float): minimum value (from 0 to 1) of the reliability to consider a source reliable
        
    Kwargs:
        objname (string): name of the object (Default: '')
        outdir (string): output folder to save the plots
        outname (string): output file name
        plot_format (string): file format of the plots
        ctr_width (float): line width of the contours (Default: 2)
        verbose (bool): option to print messages and plot to terminal if True 
        
    Returns:
        Array with the values of the reliability for each positive source
        
    Raises:
        None
    """   
    #CHECK THE KWARGS#
    verbose=kwargs['verbose'] if 'verbose' in kwargs else False
    objname=kwargs['objname'] if 'objname' in kwargs else None
    outdir=kwargs['outdir'] if 'outdir' in kwargs else os.getcwd()+'/'
    outname=kwargs['outname'] if 'outname' in kwargs else 'reliability_result'
    if objname is None or objname == '':
        outname+=outdir
    else:
        outname=outdir+objname+'_'+outname
    format=kwargs['plot_format'] if 'plot_format' in kwargs else 'pdf'
    ctr_width=kwargs['ctr_width'] if 'ctr_width' in kwargs else 2
    
    #CALCULATE THE PARAMETERS FOR RELIABILITY CALCULATION
    #.ravel() transform the array into a 1D. The shape is (N,) and not (N,1) where N is the number of sources
    pos_peak=np.log10(np.nanmax(pos_cat,axis=1).reshape(pos_cat.shape[0],1)).ravel() #calculate log10(peak/rms) for positive sources
    pos_sum=np.log10(np.nansum(pos_cat,axis=1).reshape(pos_cat.shape[0],1)).ravel() #calculate log10(sum/rms) for positive sources
    pos_mean=np.log10(np.nanmean(pos_cat,axis=1).reshape(pos_cat.shape[0],1)).ravel() #calculate log10(mean/rms) for positive sources

    neg_peak=np.log10(np.nanmax(np.abs(neg_cat),axis=1).reshape(neg_cat.shape[0],1)).ravel() #calculate log10(peak/rms) for negative sources
    neg_sum=np.log10(np.nansum(np.abs(neg_cat),axis=1).reshape(neg_cat.shape[0],1)).ravel() #calculate log10(sum/rms) for negative sources
    neg_mean=np.log10(np.nanmean(np.abs(neg_cat),axis=1).reshape(neg_cat.shape[0],1)).ravel() #calculate log10(mean/rms) for negative sources

    #PREPARE THE DATA FOR KDE ESTIMATION
    tot_peak=np.concatenate((pos_peak,neg_peak)) #concatenate log10(peak/rms) of positive and negative into a single array
    tot_sum=np.concatenate((pos_sum,neg_sum)) #concatenate log10(sum/rms) of positive and negative into a single array
    tot_mean=np.concatenate((pos_mean,neg_mean)) #concatenate log10(mean/rms) of positive and negative into a single array
    rel_params=[tot_peak,tot_sum,tot_mean] #create the parameter space for all sources
    pos_rel_params=[pos_peak,pos_sum,pos_mean] #create the parameter space for positive sources
    neg_rel_params=[neg_peak,neg_sum,neg_mean] #create the parameter space for negative sources
    
    #PREPARE THE FIGURE TO PLOT THE RESULTS
    nrows=2 #number of rows in the figure
    ncols=3 #number of columns in the figure        
    fig=plt.figure(figsize=(6*ncols,6*nrows)) #create the figure
    ax=[] #initialize the plot list
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
        neg_m=np.vstack([neg_rel_params[idx1],neg_rel_params[idx2]]) #create the matrix where each row is the measurment on the negative sources of the idx-parameter (e.g., the peak flux density)
        pos_m=np.vstack([pos_rel_params[idx1],pos_rel_params[idx2]]) #create the matrix where each row is the measurment on the positive sources of the idx-parameter (e.g., the peak flux density)
        
        #CALCULATE THE DENSITY OF POSITIVE AND NEGATIVE SOURCES IN THE PARAMETER SPACE
        #This is LinkerPar_reliability of SoFiA2 until reliability calculation (line 1132 of LinkerPar.c of Sofia v2.5.1)
        X,Y=np.meshgrid(np.sort(rel_params[idx1]),np.sort(rel_params[idx2])) #create the x,y grid coordinates with the values of the variables. np.sort() is needed to create a monotonichally increasing coordinates
        points=np.vstack([X.ravel(),Y.ravel()]) #create the points matrix. If dim is the number of variables, and N the number of observations, this matrix has dimensions (dim,N**2)
        neg_kernel=kde(neg_m) #calculate the gaussian kernel density estimation of the negative sources. neg_kernel.covariance gives the covariance matrix for negative sources. neg_kernel.factor gives the scale kernel factor for kde estimation matrix for negative sources. neg_kernel.inv_cov gives the inverse covariance matrix for negative sources
        neg_kernel.factor=0.4 #set the kernel scale factor to a standard value (for SoFiA2)
        neg_density=np.reshape(neg_kernel(points).T,X.shape) #evaluate the kde on the points and return a (N,N) matrix with the value of the negative kde for each position in the parameter space
        pos_kernel=kde(pos_m) #calculate the gaussian kernel density estimation of the positive sources
        pos_kernel.factor=0.4 #set the kernel scale factor to a standard value (for SoFiA2)
        pos_density=np.reshape(pos_kernel(points).T,X.shape) #evaluate the kde on the points and return a (N,N) matrix with the value of the positive kde for each position in the parameter space
        
        #CALCULATE THE RELIABILITY OF THE POSITIVE SOURCES
        #This is LinkerPar_reliability of SoFiA2 reliability calculation (after line 1132 of LinkerPar.c of Sofia v2.5.1)
        rel=[] #initialize the reliability list
        for k in range(len(pos_cat)): #run over the list of positive sources
            n=len(np.where(~np.isnan(pos_cat[[k]]))[0]) #number of channels contributing to a source
            SNR=np.nansum(pos_cat[k])/np.sqrt(n) #SNR of the source
            if SNR>snrmin: #if the SNR is above the chosen minimum
                P=pos_kernel.evaluate((pos_rel_params[idx1][k],pos_rel_params[idx2][k])) #density of positive sources at the position of the k-positive source
                N=neg_kernel.evaluate((pos_rel_params[idx1][k],pos_rel_params[idx2][k])) #density of negative sources at the position of the k-positive source
                if P > N: #if the positive density is higher than the negative density
                    rel.append((P-N)/P) #calculate the reliability
                else:
                    rel.append(np.array([0])) #set to 0 the reliability
            else:
                rel.append(np.array([0])) #set to 0 the reliability
        rel=np.array(rel).ravel() #convert the reliability list into an array

        ax.append(fig.add_subplot(nrows,ncols,i+1)) #create the plot for the result of the iteration
        ax[i].contour(X,Y,neg_density,colors='red',linestyles='--',linewidths=ctr_width/2) #plot the contours of the negative density
        ax[i].contour(X,Y,pos_density,colors='blue',linestyles='--',linewidths=ctr_width/2) #plot the contours of the positive density
        ax[i].scatter(pos_rel_params[idx1],pos_rel_params[idx2],color='blue',label='Positive') #plot the positive sources
        ax[i].scatter(neg_rel_params[idx1],neg_rel_params[idx2],color='red',label='Negative') #plot the negative sources
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
                       color='gray',linestyle='--',linewidth=1,zorder=0)
            ax[i].set_title('Sum vs mean',pad=10)
            ax[i].set_xlabel('log(sum/rms)')
            ax[i].set_ylabel('log(mean/rms)')
            ax[i].text(0.95,0.05,'SNR min = {}'.format(snrmin),ha='right',transform=ax[i].transAxes)

        #UPDATE THE RELIABILITY ARRAY
        if i==0: #if we are at the first iteration
            total_rel=rel #the total reliability array is the reliability array
        else:
            total_rel=np.add(total_rel,rel) #sum the reliability to the total values

    rel_x=pos_peak[total_rel>=rel_threshold] #filter the reliable sources in the peak space
    rel_y=pos_sum[total_rel>=rel_threshold] #filter the reliable sources in the sum space
    rel_z=pos_mean[total_rel>=rel_threshold] #filter the reliable sources in the mean space

    for i in range(len(rel_params)): #update the scatter plots with the highlight of the reliable sources
        if i==0:
            ax[i].scatter(rel_x,rel_y,s=100,marker='s',facecolor='none',edgecolor='black',label='Reliable')
            ax[i].legend(loc='upper left')
        elif i==1:
            ax[i].scatter(rel_x,rel_z,s=100,marker='s',facecolor='none',edgecolor='black',label='Reliable')
            ax[i].legend(loc='upper left')
        else:
            ax[i].scatter(rel_y,rel_z,s=100,marker='s',facecolor='none',edgecolor='black',label='Reliable')
            ax[i].legend(loc='upper left')

    ax=fig.add_subplot(2,1,2,projection='3d') #plot the 3D parameter space
    ax.scatter(pos_peak,pos_sum,pos_mean,color='blue',label='Positive')
    ax.scatter(neg_peak,neg_sum,neg_mean,color='red',label='Negative')
    ax.scatter(rel_x,rel_y,rel_z,s=100,marker='s',facecolor='none',edgecolor='black',label='Reliable')
    ax.set_title('Peak vs sum vs mean',pad=10)
    ax.set_xlabel('log(peak/rms)')
    ax.set_ylabel('log(sum/rms)')
    ax.set_zlabel('log(mean/rms)')
    ax.legend(loc='upper right',prop={'size':11})
    
    fig.subplots_adjust(wspace=0.4,hspace=0.3)
    fig.savefig(outname+'.'+format,dpi=300,bbox_inches='tight')
    
    if verbose:
        plt.show()
    else:
        plt.close()
    
    return total_rel
        
############################################################################################# 
def __get_sources_catalogue(data,sources_id):
    """Given an array of data and an array of sources ID, returns two list of arrays: one with fluxes of all the positive sources and one of all the negative sources in the data. The length of each source is equal and uniformed by appending nans.

    Args:
        data (array): 1D array with the data
        sources_id (array): array with the sources IDs. Positive values denotes positive sources, negative values denotes negative sources.
        
    Kwargs:
        None
        
    Returns:
        List of array with same length: one with fluxes of all the positive sources and one of all the negative sources in the data.
        
    Raises:
        None
    """   
    #DEFINE THE MAX SPECTRAL LENGTH OF THE POSITIVE AND NEGATIVE SOURCES#
    #this is to uniform the spectral length of the sources to convert the list of sources into a numpy array
    pos_max_length=0 #initialize the max number of channels in a positive source
    neg_max_length=0 #initialize the max number of channels in a negative source
    for i in range(int(np.nanmin(sources_id)),int(np.nanmax(sources_id))+1): #run over the source indexes
        if i<0: #if negative sources
            if len(sources_id[sources_id==i]) > neg_max_length: #if the negative i-source is covering more channels than the current negative max length
                neg_max_length=len(sources_id[sources_id==i]) #recalculate the max number of channels in a negative source    
        elif i>0: #if positive sources
            if len(sources_id[sources_id==i]) > pos_max_length: #if the positive i-source is covering more channels than the current positive max length
                pos_max_length=len(sources_id[sources_id==i]) #recalculate the max number of channels in a positive source

    #CREATE THE POSITIVE AND NEGATIVE SOURCE CATALOGUES#
    pos_cat=[] #initialize the positive sources catalogue
    neg_cat=[] #initialize the negative sources catalogue
    for i in range(int(np.nanmin(sources_id)),int(np.nanmax(sources_id))+1): #run over the source indexes
        if i<0: #if negative sources
            source=data[sources_id==i] #store the source spectral flux
            while len(source)<neg_max_length: #add nans until the source spectral length match the widest source
                source=np.append(source,np.nan)
            neg_cat.append(source) #add it to the negative sources list
        elif i>0: #if positive sources
            source=data[sources_id==i] #store the source spectral flux
            while len(source)<pos_max_length: #add nans until the source spectral length match the widest source
                source=np.append(source,np.nan)
            pos_cat.append(source) #add it to the positive sources list
    pos_cat=np.array(pos_cat) #convert the list into an array
    neg_cat=np.array(neg_cat) #convert the list into an array 
    
    return pos_cat,neg_cat
       
#############################################################################################
def __load(data_to_load):
    """Load data provided in various type and return them as numpy.ndarray. If the data are fits, it returns also the header as astropy.fits.io.header object.

    Args:
        data_to_load (string/PrimaryHUD/HUDlist-like): name or path+name of the fits file, or PrimaryHUD or HUDList or array-like variable of the data
        
    Returns:
        data as numpy.ndarray
        
    Raises:
        ValueError: If input given in the wrong format
    """
    if type(data_to_load)==str: #if the data are given as a string, it is assumed it is the path to a fits file
        with fits.open(data_to_load) as Data:
            data=Data[0].data
            header=Data[0].header
        return data,header
    elif type(data_to_load)==fits.HDUList:
        data=Data[0].data
        header=Data[0].header
        return data,header
    elif type(data_to_load)==fits.PrimaryHDU:
        data=Data.data
        header=Data.header
        return data,header
    elif type(data_to_load)==list:
        data=np.array(data_to_load)
        header=None
        return data,header
    elif type(data_to_load)==np.ndarray or type(data_to_load)==np.array or type(data_to_load)==tuple:
        data=data_to_load
        header=None
        return data,header
    else:
        raise ValueError('Wrong data type: {}. Accepted types are\nstring, PrimaryHUD, HUDList, or array-like.'.format(type(data_to_load)))
                                        
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
        xshape=data.shape[2]
        yshape=data.shape[1]
    else: #if the data are 2D
        xshape=data.shape[1]
        yshape=data.shape[0]
    if box is None: #if no box is given
        xmin=ymin=0 #start from 0
        xmax=xshape #select until the last x-pixel
        ymax=yshape #select until the last y pixel
    elif len(box) != 4: #if the box has the wring size
        raise ValueError('ERROR: Please provide the box in the format [xmin,xmax,ymin,ymax]. Aborting!')
    else: #store the spatial box from the input
        xmin=box[0]
        xmax=box[1]
        ymin=box[2]
        ymax=box[3]
        if xmin < 0: #if xmin is negative
            warnings.warn('Lower x limit is negative ({}): set to 0.'.format(xmin))
            xmin=0 #set it to 0
        if xmax > xshape: #if xmax is too high
            warnings.warn('Max x is too high ({}): set to the size of x.'.format(xmax))
            xmax=xshape #set it to size of data
        if ymin < 0: #if ymin is negative
            warnings.warn('Lower y limit is negative ({}): set to 0.'.format(ymin))
            ymin=0 #set it to 0
        if ymax > yshape: #if ymax is too high
            warnings.warn('Max y is too high ({}): set to the size of y.'.format(ymax))
            ymax=yshape #set it to size of data
            
    return xmin,xmax,ymin,ymax

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
        string (string): file name or of a path to the file
        object (string): which object (datacube, maskcube, ...) is the string referring to
    
    Kwargs:
        path (string): (optional) path to the file. Used if 'string' is not a path to the file
        
    Returns:
        string with the path and the file name
        
    Raises:
        None
    """
    if string == '' or string is None: #if a string is not given
        if object in kwargs: #if the object is in kwargs
            if kwargs[object] == '' or kwargs[object] is None:
                raise ValueError('ERROR: {} is not set: aborting!'.format(object))
            else:
                string=kwargs[object] #store the string from the input kwargs
        else:
            raise ValueError('ERROR: {} is not set: aborting!'.format(object))
    if string[0] != '.': #if the string start with a . means that it is a path to the files (so differs from path parameter)
        if 'path' in kwargs: #if the path to the file is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the {} is set: aborting!'.format(object))
            else:
                string=kwargs['path']+string
        else:
            raise ValueError('ERROR: no path to the {} is set: aborting!'.format(object))
    return string
    
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
def __plot_kpcline(pixelres,asectokpc,xlim,leftmargin,topmargin):
    """Plot the kpc reference line on the upper left corner of the current axis.

    Args:
        pixelres (float): cube spatial resolution in arcsec
        asectokpc (float): arcsec to kpc conversion 
        xlim (array-like): x-axis limit of the plot
        leftmargin (float): width of the left margin in axis units (from 0 to 1)
        topmargin (float): width of the top margin in axis units (from 0 to 1)
        
    Returns:
        The current axis with the kpc reference line in the upper left corner
        
    Raises:
        None
    """    
    ax=plt.gca() #get the current axis
    kpcline=10/(asectokpc*pixelres) #length of the 10 kpc line in pixel
    kpcline=kpcline/(xlim[1]-xlim[0]) #lenght of the 10 kpc line in axes size
    ax.hlines(topmargin-0.015,leftmargin,leftmargin+kpcline,color='black',linewidth=2,transform=ax.transAxes) #add the 10 kpc line
    ax.text(leftmargin+(kpcline/2),topmargin,'10 kpc',ha='center',transform=ax.transAxes)
                    
    return ax
    
#############################################################################################
def __plot_stack_result(v,spectrum,rms,expected_rms,fluxunits,mask=None,aligned=False,chmin=None,chmax=None,nrows=1,ncols=2,color='blue',ylim=None,idx=1):
    """Plot the stacked spectrum and the rms as a function of the number of stacked spectra.

    Args:
        v (array-like): spectral axis
        spectrum (array-like): stacked spectrum
        rms (array-like): rms value after each stacking iteration
        expected_rms (array-like): expected rms after each stacking iteration
        fluxunits (string): flux units of the spectrum
        mask (array-like): optional mask. Set to None to disable
        aligned (boolean): tells if the spectrum is aligned (True) or not w.r.t. the redshift
        chmin (int): show the minimum channel to ignore in the rms calculation
        chmax (int): show the maximum channel to ignore in the rms calculation
        nrows (int): number of rows in the plot
        ncols (int): number of cols in the plot
        color (string): color of the spectrum and rms lines
        ylim (array-like): (optional) y-axis limits
        idx (int): index of the plot (idx <= nrows*ncols)
        
    Returns:
        Plot of the stacked spectrum, the measured and the expected rms as matplotlib.pyplot.figure.
        
    Raises:
        None
    """
    N=len(rms) #number of stacked spectra
    fig=plt.gcf() #get the figure
    
    ax=fig.add_subplot(nrows,ncols,idx) #create the subplot for the spectrum
    if idx==1: #if the plot is in the first row
        ax.set_title('Spectrum',fontsize=20,pad=10)
    ax=__plot_stack_spectrum(v,spectrum,mask,color,aligned,chmin,chmax,rms[-1],fluxunits)
    if ylim is not None:
        ax.set_ylim(ylim)

    if idx>1: #if idx is more than 1 means we are plotting multiple stacking results and we want to uniform the ymin and ymax to compare the spectra
        axes=plt.gcf().get_axes() #get the axes in the figure
        ylim_list=[] #initialize the ylim list
        for i in np.arange(0,idx,2): #access the axes corresponding to the spectrum
            ylim_list.append(np.array(axes[i].get_ylim())) #store the y axis limits
        ylim_array=np.array(ylim_list) #convert the list into an array
        ylim=[np.nanmin(ylim_array),np.nanmax(ylim_array)] #calculate the ney ylim
        for i in np.arange(0,idx,2): #access the axes corresponding to the spectrum
            axes[i].set_ylim(ylim) #set the ylim
            
    ax=fig.add_subplot(nrows,ncols,idx+1) #create the subplot for the rms
    if idx==1: #if the plot is in the first row
        ax.set_title('RMS',fontsize=20,pad=10)
    ax.plot(np.arange(1,N+1),rms,linewidth=2,color=color,label='Measured rms') #plot the stacked rms
    ax.plot(np.arange(1,N+1),expected_rms,linewidth=2,color='gray',linestyle='--',label='Expected rms') #plot the expected rms
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(which='both',direction='in',length=10,width=1) #change the ticks of the axes from external to internal, as well as the length and the width
    ax.tick_params(which='minor',length=5) #make the minor ticks slightly shorter
    ax.set_xlabel('Number of stacked spectra') #set the x-axis label
    ax.set_ylabel('RMS [{}]'.format(fluxunits)) #set the y-axis label
    ax.set_xlim(1,N) #set the xlim
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.legend(loc='lower left')
    
    return fig

#############################################################################################   
def __plot_stack_spectrum(v,spectrum,mask,color,aligned,chmin,chmax,rms,fluxunits):
    """Plot the stacked spectrum in the current matplotlib.pyplot.axis.

    Args:
        v (array-like): spectral axis
        spectrum (array-like): stacked spectrum
        mask (array-like): optional mask. Set to None to disable
        color (string): color of the spectrum
        aligned (boolean): tells if the spectrum is aligned (True) or not w.r.t. the redshift
        chmin (int): show the minimum channel to ignore in the rms calculation
        chmax (int): show the maximum channel to ignore in the rms calculation
        rms (array-like): rms value to plot
        fluxunits (string): flux units of the spectrum
        
    Returns:
        Plot of the stacked spectrum as matplotlib.pyplot.axis object.
        
    Raises:
        None
    """
    ax=plt.gca() #get the current axis
    if mask is not None: #if a mask is provided, invert it. We want the masked portion to be plotted as dashed line, and the non masked portion as a solid line
        mask[mask>0]=1 #set to 1 the positive values
        ax.plot(v,spectrum,color=color,linewidth=1,linestyle='--',label='Unreliable')
        ax.plot(v[chmin:chmax],mask[chmin:chmax]*spectrum[chmin:chmax],color=color,linewidth=2,label='Reliable') #plot the masked stacked spectrum
        ax.legend(loc='lower center',prop={'size':11}).set_zorder(1000)
    else:
        ax.plot(v,spectrum,color=color) #plot the stacked spectrum
    ax.axhline(y=0,linestyle='--',color='black',zorder=100) #draw the 0-flux line
    if aligned: #if the spectrum is the aligned spectrum
        ax.axvline(x=0,linestyle='-.',color='black',zorder=100) #draw the 0 velocity line
        ax.axvline(x=v[chmin],linestyle='-.',color='gray',alpha=0.5,zorder=100) #draw the chmin line
        ax.axvline(x=v[chmax],linestyle='-.',color='gray',alpha=0.5,zorder=100) #draw the chmax line
    ax.axhline(y=rms,linestyle='--',color='gray',zorder=100) #draw the final rms line
    ax.axhline(y=-rms,linestyle='--',color='gray',zorder=100) #draw the final rms line
    ax.tick_params(direction='in',length=10,width=1) #change the ticks of the axes from external to internal, as well as the length and the width
    ax.set_xlabel('Velocity [km/s]') #set the x-axis label
    ax.set_ylabel('<Flux> [{}]'.format(fluxunits)) #set the y-axis label
    ax.set_xlim(np.nanmin(v),np.nanmax(v))
    
    return ax

############################################################################################# 
def __source_finder(data,threshold):
    """Find the elements of a 1D array whose absolute value is above a threshold).

    Args:
        data (array): 1D array with the data
        threshold (float): value of the threshold. Elements of data whose absolute value is above the threshold are set to 1, the others to 0
        
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
def __source_linker(data,mask,kernel):
    """Assign to kernel-size consecutive positive/negative elements of a 1D array the same value representing the index of the source.

    Args:
        data (array): 1D array with the data
        mask (array): 1D array with the mask
        kernel (int): kernel size. It must be an odd number
        
    Kwargs:
        None
        
    Returns:
        Array of values representing the sources indexes.
        
    Raises:
        ValueError: If the kernel is not an odd number
    """
    if kernel%2 == 0: #if the kernel is an even number
        raise ValueError('Linker kernel must be an odd number.')
    linked_mask=mask.copy()*np.nan #initialize the linked mask
    pos_index=1 #initialize the positive source index
    neg_index=-1 #initialize the negative source index
    source=False #initialize the source check
    for i in range(len(mask)-kernel+1): #run over the mask
        if np.all(mask[i:i+kernel])>0: #if all the channels in the kernel are marked as a source:
            if not source: #if it is a new source
                start=i #store the channel index
                source=True #tell you have a source
            else: #if you are already looking at a source
                pass #do nothing
        else:
            if source: #if you were looking at a source
                if np.nansum(data[start:i+kernel-1])>0: #if the source is positive
                    linked_mask[start:i+kernel-1]=pos_index #assing the positive label to the source
                    pos_index+=1 #increase the positive label
                elif np.nansum(data[start:i+kernel-1])<0: #if the source is negative
                    linked_mask[start:i+kernel-1]=neg_index #assing the negative label to the source
                    neg_index-=1 #decrease the negative label
                else: #if the source has a total flux of 0
                    pass #do nothing
                source=False #tell you are no more looking for a source
                
    return linked_mask    

#############################################################################################    
def __stack(data,x,y,ignore=None,flip=False,diagnostic=False,**diagnostic_kwargs):
    """Stack the spectra for each sky position (x,y).

    Args:
        data (ndarray): 3D array with the data
        x,y (int): x,y pixel coordinates of the spectra to stack
        ignore (array-like): array-like with the min and max channel of the emission region to ignore for the rms computation
        flip (boolean): flip the cube along the spectral axis (set to True if the spectral resolution is negative)
        diagnostic (boolean): store all the diagnostic files and plots. Warning: the diagnostic might occupy large portions of the disk (default: False)
        
    Kwargs:
        v (array-like): spectral axis
        fluxunits (string): units of the flux
        color (string): color of the plot
        outdir (string): output directiory to store the plots
        format (string): file format of the plots

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
        v=diagnostic_kwargs['v'] #store the spectral axis
        fluxunits=diagnostic_kwargs['fluxunits'] #store the diagnostic plots format
        color=diagnostic_kwargs['color'] #store the spectrum color
        outdir=diagnostic_kwargs['outdir'] #store the diagnostic plots output folder
        format=diagnostic_kwargs['format'] #store the diagnostic plots format
        chmin=ignore[0] #store the lower channel of the galactic emission
        chmax=ignore[1] #store the upper channel of the galactic emission
        left_mean=[] #initialize the list of mean of the low velocities
        right_mean=[] #initialize the list of mean of the high velocities
        left_rms=[] #initialize the list of rms of the low velocities
        right_rms=[] #initialize the list of rms of the high velocities
        tot_mean=[] #initialize the total mean list
        tot_rms=[] #initialize the total rms list
        if not os.path.exists(outdir): #if the output folder does not exist
            os.makedirs(outdir) #create the folder
            
    if flip: #if the spectra must be flipped
        data=np.flip(data,axis=0) #flip the  cube along the spectral axis
      
    if diagnostic: #if the diagnostic plots must be made
        plt.ioff() #disable the interactive plotting 
        
    for i in tqdm(zip(y,x),desc='Spectra stacked',total=len(x)): #run over the pixels    
        if diagnostic: #if the diagnostic plots must be made
            dummy=data[:,i[0],i[1]].copy() #copy the spectrum used in the stacking
            left_mean.append(np.nanmean(dummy[:chmin])) #calculate the low velocities mean
            right_mean.append(np.nanmean(dummy[chmax:])) #calculate the high velocities mean
            left_rms.append(np.sqrt(np.nanmean(dummy[:chmin]**2))) #calculate the low velocities rms
            right_rms.append(np.sqrt(np.nanmean(dummy[chmax:]**2))) #calculate the high velocities rms
            dummy[chmin:chmax+1]=np.nan #blank the galactic emission
            tot_mean.append(np.nanmean(dummy)) #calculate the total mean
            tot_rms.append(np.sqrt(np.nanmean(dummy**2))) #calculate the total rms
        
            ncols=1 #number of columns in the plot
            nrows=3 #number of rows in the plot
            fig=plt.figure(figsize=(12*ncols,6*nrows)) #create the figure
            fig.suptitle('Pre stack, to be stacked and after stack spectrum comparison (N={})'.format(N),fontsize=24) #add the title
            
            ax=fig.add_subplot(nrows,ncols,1) #create the subplot for the pre-stack stacked spectrum
            if N==1: #if this is the first iteration
                ax=__plot_stack_spectrum(v,stack,None,color,True,chmin,chmax,0,fluxunits)
            else:
                ax=__plot_stack_spectrum(v,stack/weights,None,color,True,chmin,chmax,stack_rms[-1],fluxunits)
            ax.set_xlabel('')
            ax.xaxis.set_ticklabels([])
            if N>1: #after the first iteration
                ax.set_ylim(ylim)
            
            ax=fig.add_subplot(nrows,ncols,2) #create the subplot for the spectrum to be stacked
            ax=__plot_stack_spectrum(v,data[:,i[0],i[1]],None,color,True,chmin,chmax,tot_rms[-1],fluxunits)
            ax.set_xlabel('')
            ax.xaxis.set_ticklabels([])
            
        input_rms=np.sqrt(np.nanmean(data[:,i[0],i[1]]**2)) #calculate the rms of the input spectrum
        stack=np.nansum((stack,data[:,i[0],i[1]]/input_rms**2),axis=0) #stack the input spectrum
        weights+=(1/input_rms**2) #stack the weight
        dummy=stack.copy()/weights #create a dummy copy of the stacked spectrum. Here we divide for the number of stacked weights so far, since we want the rms as if we have stacked all the spectra
        if ignore is not None: #if the channel range over which the flux is ignored is provided
            dummy[ignore[0]+1:ignore[1]]=np.nan #blank the region not used for rms calculation
        stack_rms.append(np.sqrt(np.nanmean(dummy**2))) #calculate the rms of the stacked spectrum
        exp.append(stack_rms[0]/np.sqrt(N)) #expected value for the rms
        
        if diagnostic: #if the diagnostic plots must be made
            ax=fig.add_subplot(nrows,ncols,3) #create the subplot for the post-stack stacked spectrum
            ax=__plot_stack_spectrum(v,stack/weights,None,color,True,chmin,chmax,stack_rms[-1],fluxunits)
            if N==1: #after the first iteration
                ylim=ax.get_ylim() #get the ylim
            else:
                ax.set_ylim(ylim)
            fig.subplots_adjust(top=0.95,hspace=0.1)
            fig.savefig(outdir+'result_after_{}_stacked_spectra'.format(N)+format,dpi=300,bbox_inches='tight')
            plt.close(fig=fig)
        N+=1 #increase the number of stacked spectra
        
    if diagnostic:  #if the diagnostic plots must be made
        ncols=2 #number of columns in the plot
        nrows=1 #number of rows in the plot
        fig=plt.figure(figsize=(6*ncols,6*nrows)) #create the figure
        ax=fig.add_subplot(nrows,ncols,1) #create the subplot for the rms histogram
        ax.set_title('RMS',fontsize=20)
        hist(left_rms,bins='freedman',color='green',histtype='stepfilled',alpha=0.5,label='Low velocities') #plot the low velocities rms histogram
        hist(right_rms,bins='freedman',color='blue',histtype='stepfilled',alpha=0.5,label='High velocities') #plot the high velocities rms histogram
        hist(tot_rms,bins='freedman',color=None,edgecolor='black',histtype='step',linewidth=1.5,label='Both') #plot the total rms histogram
        ax.axvline(x=np.nanmean(tot_rms),linestyle='--',color='gray',label='Mean rms') #add the mean rms line 
        ax.tick_params(direction='in',length=10,width=1) #change the ticks of the axes from external to internal
        ax.set_xlabel('RMS') #set the x-axis label
        ax.set_ylabel('Counts') #set the y-axis label
        ax.legend(loc='upper right',prop={'size':11})
    
        ax=fig.add_subplot(nrows,ncols,2) #create the subplot for the mean histogram
        ax.set_title('Mean',fontsize=20)
        hist(left_mean,bins='freedman',color='green',histtype='stepfilled',alpha=0.5,label='Low velocities') #plot the low velocities mean histogram
        hist(right_mean,bins='freedman',color='blue',histtype='stepfilled',alpha=0.5,label='High velocities') #plot the high velocities mean histogram
        hist(tot_mean,bins='freedman',color=None,edgecolor='black',histtype='step',linewidth=1.5,label='Both') #plot the total mean histogram
        ax.axvline(x=0,linestyle='--',color='gray') #add the 0-mean line
        xl=np.nanmax((np.abs(left_mean),np.abs(right_mean)))*1.1
        ax.tick_params(direction='in',length=10,width=1) #change the ticks of the axes from external to internal
        ax.set_xlabel('Mean') #set the x-axis label
        ax.set_ylabel('Counts') #set the y-axis label
        ax.set_xlim(-xl,xl)
        ax.legend(loc='upper right',prop={'size':11})
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        fig.subplots_adjust(wspace=0.1)
        fig.savefig(outdir+'noise_histograms'+format,dpi=300,bbox_inches='tight')
        plt.close(fig=fig)
        plt.ion() #re-enable the interactive plotting
           
    return stack/weights,stack_rms,exp