"""A collection of functions for the analysis of astronomical data cubes and images.

                NICCI
New analysIs Code for Cubes and Images
"""
#############################################################################################
#
# FILENAME: main
# VERSION: 0.10
# DATE: 19/08/2022
# CHANGELOG:
#   v 0.10: - importparameters
#               - added the parameter chanbox
#               - added parameters to account changes in the other functions
#           - chanmap
#               - improved cube properties retrival
#               - improved automatic retrival of spatial and spectral info from the cube if not given
#               - added automatic conversion to arcsec and km/s
#               - improved ancillary information display
#               - improved automatic limits calculation
#               - improved systemic velocity import
#               - added the option to select a spatial box
#               - improved input arguments call
#               - aspect ratio is now 'equal' and no more 'auto'
#           - cubedo
#               - fixed wcs recalculation in operation 'cut'
#               - added check on type of HDU to avoid extend tables in operation 'extend'
#           - cubestat
#               - complete rework of the cube reading routine
#               - added units information to the output
#               - fixed sensitivity calculation
#               - fixed sensitivity print
#               - improved input arguments call
#           - gaussfit
#               - can now fit without a mask
#           - getpv
#               - no longer checks for pvangle if more than 2 points are given
#               - now accept undefined number of points
#               - improved data normalization
#               - improved contours color
#               - improved ancillary information display
#               - changed normalization from sqrt to cube root
#           - plotmom
#               - fixed sensitivity import according to new cubestat
#               - improved cube properties retrival
#               - improved automatic retrival of spatial and spectral info from the cube or maps if not given
#               - added automatic conversion from deg/arcimn and m/s to arcsec and km/s
#               - improved ancillary information display
#               - improved automatic limits calculation
#               - improved mom1 contours calculation
#               - changed mom0 normalization from sqrt to cube root
#               - aspect ratio is now 'equal' and no more 'auto'
#           - converttoHI
#               - accept various spatial and spectral units and automatically converts to arcsec and km/s
#           - getHImass
#               - accept various spatial units and automatically converts to arcsec
#           - flux
#               - accept various spatial and spectral units and automatically converts to arcsec and km/s
#               - fixed box call
#           - create_config
#               - added parameters to account changes in the other functions
#           - changed default spectral unit from km/s to m/s
#           - remove the redundant fits.close() when using with fits.open() ... as ...
#           - added checks on the boxes sizes
#           - improved documentation
#
#   TO DO:  - rotcurve, non fa quello chefa rotcurve in gipsy
#           - crea la funzione update_params che aggiorna il valore di un parametro nel dizionario dei parametri. Utile
#             per quando ad sempi non hai i valori del beam e usi cubestat per trovarli e cubestat li inserisce nel
#             dizionario
#           - aggiungere l'interpolazione in shuffle
#           - aggiungere galmod, fixhead di gipsy
#           - aggiungere il logger (con logger.)
#
#############################################################################################

from astropy import units as u
from astropy.coordinates import ICRS
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter as fit
from pvextractor import Path
from pvextractor import PathFromCenter
from pvextractor import extract_pv_slice
from scipy.stats import chi2 as statchi2
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
    """Read an ini-structured text file and store each row in a dictionary entry. See the text file for the description of each parameter.
    
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
    parameters['cubedoindir']=config.get('CUBEDO','inputdir') if config.has_option('CUBEDO','inputdir') else None #input directory. If empty, is the same of [INPUT] path. If it is not given, set it to None
    parameters['chanmin']=config.getint('CUBEDO','chanmin') if config.has_option('CUBEDO','chanmin') else 0 #first channel for the operations 'blank,cut,mom0'. If it is not given, set it to 0
    parameters['chanmax']=config.getint('CUBEDO','chanmax') if config.has_option('CUBEDO','chanmax') else None #last channel for the operations 'blank,cut,mom0'. If it is not given, set it to None
    parameters['cutbox']=config.get('CUBEDO','box').split(',') if config.has_option('CUBEDO','box') else None #comma-separated pixel edges of the box to extract for operation 'cut' in the format [xmin,xmax,ymin,ymax]. If it is not given, set it to None
    if parameters['cutbox'] != [''] and parameters['cutbox'] is not None: #if the box is given
        parameters['cutbox']=[int(i) for i in parameters['cutbox']] #convert string to float
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
    parameters['fixmaskindir']=config.get('FIXMASK','inputdir') if config.has_option('FIXMASK','inputdir') else None #input directory. If empty, is the same of [INPUT] path. If it is not given, set it to None
    parameters['fixmaskoutdir']=config.get('FIXMASK','outputdir') if config.has_option('FIXMASK','outputdir') else None #output directory to save the new mask. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['fixmaskoutname']=config.get('FIXMASK','outname') if config.has_option('FIXMASK','outname') else None #output name of the new mask including extension .fits. If empty, is the same of maskcube. If it is not given, set it to None
    
    #GAUSSFIT section
    parameters['GAUSSFIT']='----------------- GAUSSFIT -----------------'
    parameters['cubetofit']=config.get('GAUSSFIT','datacube') if config.has_option('GAUSSFIT','datacube') else None #name of the fits file of the data cube to fit including .fits. If empty, is the same of [FITS] datacube. If it is not given, set it to None
    parameters['gaussmask']=config.get('GAUSSFIT','gaussmask') if config.has_option('GAUSSFIT','gaussmask') else None #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube. The fit will be done inside the mask. If it is not given, set it to None
    parameters['gaussindir']=config.get('GAUSSFIT','inputdir') if config.has_option('GAUSSFIT','inputdir') else None #input directory. If empty, is the same of [INPUT] path. If it is not given, set it to None
    parameters['linefwhm']=config.getfloat('GAUSSFIT','linefwhm') if config.has_option('GAUSSFIT','linefwhm') else 15 #first guess on the fwhm of the line profile in km/s. If it is not given, set it to 15
    parameters['amp_thresh']=config.getfloat('GAUSSFIT','amp_thresh') if config.has_option('GAUSSFIT','amp_thresh') else 0 #amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum. If it is not given, set it to 0
    parameters['p_reject']=config.getfloat('GAUSSFIT','p_reject') if config.has_option('GAUSSFIT','p_reject') else 1 #p-value threshold for fit rejection. If a best-fit as p>p_reject, it will be rejected . If it is not given, set it to 1
    parameters['clipping']=config.getboolean('GAUSSFIT','clipping') if config.has_option('GAUSSFIT','clipping') else False #clip the spectrum to a % of the profile peak [True,False]. If it is not given, set it to False
    parameters['clipthreshold']=config.getfloat('GAUSSFIT','threshold') if config.has_option('GAUSSFIT','threshold') else 0.5 #clip threshold as % of the peak (0.5 is 50%) if clipping is True. If it is not given, set it to 0.5
    parameters['errors']=config.getboolean('GAUSSFIT','errors') if config.has_option('GAUSSFIT','errors') else False #compute the errors on the best-fit [True,False]. If it is not given, set it to False
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
    parameters['rotcurveoutdir']=config.get('ROTCURVE','outputdir') if config.has_option('ROTCURVE','outputdir') else None #output directory to save the plot. If empty, is the same as [INPUT] path. If it is not given, set it to None
    parameters['rotcurveoutname']=config.get('ROTCURVE','outname') if config.has_option('ROTCURVE','outname') else None #output name of the plot (including file extension *.jpg,*.png,*.pdf,...). If it is not given, set it to None
    
    #PLOTSTYLE section
    parameters['PLOTSTYLE']='----------------- PLOTSTYLE -----------------'
    parameters['ctr_width']=config.getfloat('PLOTSTYLE','ctr width') if config.has_option('PLOTSTYLE','ctr width') else 2 #width of the contours levels. If it is not given, set it to 2
            
    return parameters

#############################################################################################
def chanmap(datacube='',from_chan=0,to_chan=None,chansep=1,chanmask=False,chanmapoutdir='',chanmapoutname='',save=False,**kwargs):
    """Plot the channel map of a fits cube and (optionally) save it in a file.

    Args:
        datacube (string): name or path+name of the fits data cube
        from_chan (int): first channel to be plotted in the channel map
        to_chan (int): last channel to be plotted in the channel map
        chansep (int): channel separation in the channel map. The channels plotted are from from_chan to to_chan each chansep
        chanmask (bool): option to use a detection mask (True) or not (False) to highlight the 'real' emission in the channel map. If it is True, a 3D mask must be provided using the 'maskcube' kwarg (see kwargs arguments below)
        chanmapoutdir (string): the output folder name
        chanmapoutname (string): output file name
        save (bool): option to save the plot (True) or not (False)

    Kwargs:
        path (string): path to the data cube if the datacube is a name and not a path+name
        maskcube (string): name or path+name of the fits 3D mask cube. Used if chanmask=True
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
        chanbox (list/array): spatial box as [xmin,xmax,ymin,ymax] (Default: None)                     
        vsys (float): object systemic velocity in m/s (Default: 0)
        asectokpc (float): arcsec to kpc conversion to plot the spatial scale (Default: None)
        objname (string): name of the object (Default: '')
        contours (list/array): contours levels in terms of rms. They will replace the default levels (Default: None)                                    
        chansig (float): lowest contour level in terms of chansig*rms (Default: 3)
        ctr_width (float): line width of the contours (Default: 2)
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
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    if datacube == '' or datacube is None: #if a data cube is not given
        raise ValueError('ERROR: datacube is not set: aborting!')
    if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the data cube is set: aborting!')
            else:
                datacube=kwargs['path']+datacube
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')
    if chanmask: #if a mask cube is used
        if 'maskcube' in kwargs: #if the datacube is in kwargs
            if kwargs['maskcube'] == '' or kwargs['maskcube'] is None:
                raise ValueError('ERROR: you set to use a mask but no mask is provided: aborting!')
            else:
                maskcube=kwargs['maskcube'] #store the default data cube path from the input kwargs
        else:
            raise ValueError('ERROR: you set to use a mask but no mask is provided: aborting!')
        if maskcube[0] != '.': #if the mask cube name start with a . means that it is a path to the cube (so differs from path parameter)
            maskcube=kwargs['path']+maskcube
    outdir=chanmapoutdir #store the output directory from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=kwargs['path'] #the outdir is the path parameter
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder
    outname=chanmapoutname #store the output name from the input parameters
    if outname == '' or outname is None:  #if the outname is empty
        outname=datacube.replace('.fits','_chanmap.pdf')  #the outname is the object name plus chanmap.pdf
    if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    #CHECK THE KWARGS#
    if 'pixunits' in kwargs: #if the spatial units are in the input kwargs
        pixunits=kwargs['pixunits'] #store the spatial units from the input kwargs
    else:
        pixunits=None #set it to None   
    if 'specunits' in kwargs: #if the spectral units are in the input kwargs
        specunits=kwargs['specunits'] #store the spectral units from the input kwargs
    else:
        specunits=None #set it to None    
    if 'pixelres' in kwargs: #if the pixel resolution is in the input kwargs
        pixelres=kwargs['pixelres'] #store the pixel resolution from the input kwargs
    else:
        pixelres=None #set it to None
    if 'spectralres' in kwargs: #if the spectral resolution is in the input kwargs
        spectralres=kwargs['spectralres'] #store the spectral resolution from the input kwargs
    else:
        spectralres=None #set it to None
    if 'bmaj' in kwargs: #if the beam major axis is in the input kwargs
        bmaj=kwargs['bmaj'] #store the beam major axis from the input kwargs
    else:
        bmaj=None #set it to None
    if 'bmin' in kwargs: #if the beam minor axis is in the input kwargs
        bmin=kwargs['bmin'] #store the beam minor axis from the input kwargs
    else:
        bmin=None #set it to None
    if 'bpa' in kwargs: #if the beam position angle is in the input kwargs
        bpa=kwargs['bpa'] #store the beam position angle from the input kwargs
    else:
        bpa=None #set it to None
    if 'rms' in kwargs: #if the cube rms is provided
        rms=kwargs['rms'] #store the rms from the input kwargs
    else:
        rms=None #set it to None
    if 'chanbox' in kwargs: #if a box is provided
        chanbox=kwargs['chanbox'] #store the box from the input kwargs
    else:
        chanbox=None #set it to None
    if 'vsys' in kwargs: #if the systemic velocity is in the input kwargs
        if kwargs['vsys'] is not None: #if it is also not None
            vsys=kwargs['vsys'] #store the systemic velocity from the input kwargs
        else:
            vsys=0 #set it to 0
            warnings.warn('No systemic velocity is given: set it to 0 m/s!')
    else:
        vsys=0 #set it to 0
        warnings.warn('No systemic velocity is given: set it to 0 m/s!')
    if 'asectokpc' in kwargs: #if the arcsec-to-kpc conversion is in the input kwargs
        asectokpc=kwargs['asectokpc'] #store the arcsec-to-kpc conversion from the input kwargs
    else:
        asectokpc=None #set it to None
    if 'objname' in kwargs: #if the object name is in the input kwargs
        objname=kwargs['objname'] #store the object name from the input kwargs
        if objname is None: #if in kwargs but not set
            objname='' #set it to empty
    else:
        objname='' #set it to empty
    if 'contours' in kwargs: #if the conotur levels are in the input kwargs
        contours=kwargs['contours'] #store the contour levels from the input kwargs
    else:
        contours=None #set it to None
    if 'chansig' in kwargs: #if the lowest contours sigma is in the input kwargs
        chansig=kwargs['chansig'] #store the lowest contours sigma from the input kwargs
    else:
        chansig=3 #set it to 3
    if 'ctr_width' in kwargs: #if the contours width is in the input kwargs
        ctr_width=kwargs['ctr_width'] #store the contours width from the input kwargs
    else:
        ctr_width=2 #set it to 2
    if pixunits not in [None,'deg','arcmin','arcsec']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spatial units in deg, arcmin or arcsec. Aborting!')
    if specunits not in [None,'km/s','m/s','Hz']: #if wrong spatial units are given
        raise ValueError('ERROR: Please provide the spectral units km/s, m/s or Hz. Aborting!')
        
    #---------------   START THE FUNCTION   ---------------#    
    # NOW WE OPEN THE DATACUBE #
    with fits.open(datacube) as cube: #open the data cube
        data=cube[0].data #store the cube data
        header=cube[0].header #store the cube header
        
    # NOW WE CHECK FOR THE RELEVANT INFORMATION #
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
            raise ValueError('ERROR: I am still missing some information: {}. Please check the parameter!'.format(not_found))        
    #------------   CONVERT THE VALUES IN STANDARD UNITS   ------------ 
    if pixunits == 'deg': #if the spatial units are deg
        bmaj=bmaj*3600 #convert the beam major axis in arcsec
        bmin=bmin*3600 #convert the beam minor axis in arcsec
        pixelres=pixelres*3600 #convert the pixel size in arcsec^2
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        bmaj=bmaj*60 #convert the beam major axis in arcsec
        bmin=bmin*60 #convert the beam minor axis in arcsec
        pixelres=pixelres*60 #convert the pixel size in arcsec^2
    if specunits == 'm/s': #if the spectralunits are m/s
        spectralres=spectralres/1000 #convert the spectral resolution in km/s
    beamarea=1.13*(bmin*bmaj) #calculate the beam area
    if pixelres<0: #if the spatial resolution is negative
        pixelres=-pixelres #convert it to positive
        
    #-----------   SPECTRAL AXIS    -----------#
    if 'CRVAL3' in header: #if the header has the starting spectral value
        chan0=header['CRVAL3'] #store the starting spectral value
    else:
        raise ValueError('ERROR: no starting channel spectral value found: aborting!')
    if specunits=='m/s': #if the spectral units are m/s
        chan0=chan0/1000 #convert starting spectral value to km/s
        vsys=vsys/1000 #covert the systemic velocity in km/s
    wcs=WCS(header).dropaxis(2) #store the WCS info and drop the spectral axis

    # NOW WE CHECK IF A MASK IS USED AND OPEN IT #
    if chanmask: #if a mask must be used
        with fits.open(maskcube) as maskcube: #open the mask cube
            mask=maskcube[0].data #store the mask data
            if mask.shape[0] != data.shape[0] or mask.shape[1] != data.shape[1] or mask.shape[2] != data.shape[2]: #if the mask cube has different size than the data cube
                raise ValueError('ERROR: mask cube and data cube has different shapes:\n'
                                 '({}, {}, {}) the mask cube and ({}, {}, {}) the data cube. Aborting!'.format(mask.shape[0],mask.shape[1],mask.shape[2],data.shape[0],data.shape[1],data.shape[2]))
            mask[mask>1]=1 #convert the mask into a 0/1 array
            
    # NOW WE PREPARE THE CHANNELS VECTOR #  
    #WE CHECK THE CHANNELS
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
    if chanbox is None: #if no box is given
        xmin=ymin=0 #start from 0
        xmax=data.shape[2] #select unti the last x-pixel
        ymax=data.shape[1] #select unti the last y pixel
    elif len(chanbox) != 4: #if the box has the wring size
        raise ValueError('ERROR: Please provide the box in the format [xmin,xmax,ymin,ymax]. Aborting!')
    else: #store the spatial box from the input
        xmin=chanbox[0]
        xmax=chanbox[1]
        ymin=chanbox[2]
        ymax=chanbox[3]
        if xmin < 0: #if xmin is negative
            warnings.warn('Lower x limit is negative ({}): set to 0.'.format(xmin))
            xmin=0 #set it to 0
        if xmax > data.shape[2]: #if xmax is too high
            warnings.warn('Max x is too high ({}): set to the size of x.'.format(xmax))
            xmax=data.shape[2] #set it to size of data
        if ymin < 0: #if ymin is negative
            warnings.warn('Lower y limit is negative ({}): set to 0.'.format(ymin))
            ymin=0 #set it to 0
        if ymax > data.shape[1]: #if ymax is too high
            warnings.warn('Max y is too high ({}): set to the size of y.'.format(ymax))
            ymax=data.shape[1] #set it to size of data
                
    # NOW WE SETUP THE FIGURE #    
    nrows=int(np.ceil(np.sqrt(len(chans)))) #number of rows in the channel map
    ncols=int(np.floor(np.sqrt(len(chans)))) #number of columns in the channel map
    fig=plt.figure(figsize=(4*ncols,4*nrows)) #create the figure
    fig.suptitle('{} HI channel maps'.format(objname),fontsize=24) #add the title
    k=0 #index to run over the channels
    # NOW WE SETUP THE SUBPLOT PROPERTIES #    
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
    #it is highly raccomended to have an equal aspect ration for the moment maps. So we have to extend the shorter axis to match the larger
    if ylim[1]-ylim[0] > xlim[1]-xlim[0]: #if the y-axis is bigger than the x-axis
        extend=(ylim[1]-ylim[0]-xlim[1]+xlim[0])/2 #calculate how much to extend
        xlim[1]=xlim[1]+extend
        xlim[0]=xlim[0]-extend
    else: #if the x-axis is larger
        extend=(xlim[1]-xlim[0]-ylim[1]+ylim[0])/2 #calculate how much to extend
        ylim[1]=ylim[1]+extend
        ylim[0]=ylim[0]-extend
    #----------   NORMALIZATION  ---------#
    vmin=0.05*np.nanmin(data) #set the lower limit for the normalization
    vmax=0.15*np.nanmax(data) #set the upper limit for the normalization
    norm=cl.PowerNorm(gamma=0.3,vmin=vmin,vmax=vmax) #define the data normalization
    #------------   CONTOURS  ------------#
    if contours is None: #if contours levels are not given:
        maxsig=round(np.log(np.sqrt(vmax)/rms)/np.log(chansig)) #calculate the max sigma in the data
        ctr=np.power(chansig,np.arange(1,maxsig,2)) #contours level in units of nsigma^i (i from 1 to max sigma)
    else:
        ctr=contours #use the levels given in input
        
    # NOW WE DO THE PLOT # 
    for i in range(nrows):
        for j in range(ncols):
            fig.add_subplot(nrows,ncols,k+1,projection=wcs) #create the subplot
            ax=plt.gca() #get the current axes
            chanmap=data[chans[k],ymin:ymax,xmin:xmax] #select the channel
            im=ax.imshow(chanmap,cmap='Greys',norm=norm,aspect='equal') #plot the channel map in units of detection limit
            if chanmask: #if a mask must be used
                ax.contour(chanmap/rms,levels=ctr,cmap='Greys_r',linewidths=ctr_width/2,linestyles='solid') #add the contours
                ax.contour(chanmap/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width/2,linestyles='dashed') #add the negative contours
                ax.contour(chanmap*mask[chans[k],ymin:ymax,xmin:xmax]/rms,levels=ctr,cmap='Greys_r',linewidths=ctr_width,linestyles='solid') #add the contours within the mask
            else:
                ax.contour(chanmap/rms,levels=ctr,cmap='Greys_r',linewidths=ctr_width,linestyles='solid') #add the contours
                ax.contour(chanmap/rms,levels=-np.flip(ctr),colors='gray',linewidths=kwargs['ctr_width'],linestyles='dashed') #add the negative contours
            ax.tick_params(direction='in') #change the ticks of the axes from external to internal
            ax.set_xlabel('RA') #set the x-axis label
            ax.set_ylabel('DEC') #set the y-axis label
            if j != 0: #if not plotting the first column
               ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
            if i != nrows-1: #if not plotting the last row
               ax.coords[0].set_ticklabel_visible(False) #hide the x-axis ticklabels and labels
            ax.set_xlim(xlim) #set the xlim
            ax.set_ylim(ylim) #set the ylim
            ax.text(leftmargin,bottommargin,'V$_{{rad}}$: {:.2f} km/s'.format(chans[k]*spectralres+chan0-vsys),transform=ax.transAxes) #add the information of the channel velocity
            if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
                pxbeam=np.array([bmaj,bmin])/pixelres #beam in pixel
                box=patch.Rectangle((xlim[1]-2*pxbeam[0],ylim[0]),2*pxbeam[0],2*pxbeam[1],fill=None) #create the box for the beam. The box start at the bottom and at twice the beam size from the right edge and is twice the beam large. So, the beam is enclosed in a box that extend a beam radius around the beam patch
                beam=patch.Ellipse((xlim[1]-pxbeam[0],ylim[0]+pxbeam[1]),pxbeam[0],pxbeam[1],bpa,hatch='/////',fill=None) #create the beam patch. The beam center is at a beamsize (in pixel) from the plot border
                ax.add_patch(box) #add the beam box
                ax.add_patch(beam) #add the beam
            if j == 0: #in the fisr columns
                if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
                    kpcline=10/(asectokpc*pixelres) #length of the 10 kpc line in pixel
                    kpcline=kpcline/(xlim[1]-xlim[0]) #lenght of the 10 kpc line in axes size
                    if i == 0: #at the first iteration
                        leftmargin=leftmargin+0.05 #place the 10 kpc line and its label slightly more to the left and down to avoid being too much close to the axes
                        topmargin=topmargin-0.01
                    ax.hlines(topmargin-0.015,leftmargin,leftmargin+kpcline,color='black',linewidth=2,transform=ax.transAxes) #add the 10 kpc line
                    ax.text(leftmargin+(kpcline/2),topmargin,'10 kpc',ha='center',transform=ax.transAxes)
            k+=1
            
    fig.subplots_adjust(left=0.075, bottom=0.05, right=0.98, top=0.95, wspace=0.0, hspace=0.0) #fix the position of the subplots in the figure
    if save: #if the save switch is true
        fig.savefig(outname,dpi=300,bbox_inches='tight') #save the figure

    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    else:
        plt.close()

#############################################################################################
def cubedo(cubedo='',cubedoindir='',operation=None,chanmin=0,chanmax=None,cutbox=None,addchan=None,value='blank',withmask=False,
           cubedomask=None,cliplevel=0.5,xrot=None,yrot=None,zrot=None,cubedooutdir='',cubedooutname='',**kwargs):
    """Perform the selected operation between blank, clip, crop, cut, extend, mirror, mom0, shuffle, toint on a fits data cube.

    Args:
        cubedo (string): name or path+name of the fits data cube
        cubedoindir (string): name of the inpt directory
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
        cutbox (list/array): spatial cut box as [xmin,xmax,ymin,ymax]
        addchan (int): number of channels to add in operation extend. If < 0 the channels are added at the beginning of the spectral axis, else at the end
        value (float/string): value to give to the blanked pixel in operation blank. If string 'blank' it will be np.nan
        withmask (bool): option to use a detection mask (True) or not (False) for clip and mom0. If it is True, a 2D mask for clip or 3D mask for mom0 must be provided using the cubedomask  argument or the 'mask2d' and 'maskcube' kwarg (see kwargs arguments below)
        cubedomask (string): name or path+name of the fits 2D or 3D mask. Used if withmask=True
        cliplevel (float): clip threshold as % of the peak (0.5 is 50%) for clip
        xrot (float): x-coordinate of the rotational center for operation mirror in pixel
        yrot (float): y-coordinate of the rotational center for operation mirror in pixel
        zrot (int): z-coordinate of the rotational center for operation mirror in channels
        cubedooutdir (string): output folder name
        cubedooutname (string): output file name
        
    Kwargs:
        datacube (string): name or path+name of the fits data cube if cubedo is not given
        path (string): path to the data cube if the datacube is a name and not a path+name
        mom1map (string): name or path+name of the fits moment 1 map to be used for operation shuffle
        mask2d (string): name or path+name of the fits 2D mask for operation clip. Used if withmask=True
        maskcube (string): name or path+name of the fits 3D mask cube for operation mom0. Used if withmask=True
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
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    datacube=cubedo #store the data cube path from the input parameters
    if datacube == '' or datacube is None: #if a data cube is not given
        if 'datacube' in kwargs: #if the datacube is in kwargs
            if kwargs['datacube'] == '' or kwargs['datacube'] is None:
                raise ValueError('ERROR: datacube is not set: aborting!')
            else:
                datacube=kwargs['datacube'] #store the default data cube path from the input kwargs
        else:
            raise ValueError('ERROR: datacube is not set: aborting!')
    indir=cubedoindir #store the input folder from the input parameters
    if indir == '' or indir is None: #if the input folder is not given
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the data cube is set: aborting!')
            else:
                indir=kwargs['path']
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')
    elif not os.path.exists(indir): #if the input folder does not exist
        os.makedirs(indir) #create the folder
    if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
        datacube=indir+datacube
    if operation is None: #if no operation is set
        raise ValueError('ERROR: no operation set: aborting')
    if operation not in ['blank','clip','crop','cut','extend','mirror','mom0','shuffle','toint']: #if wrong operation is given
        raise ValueError("ERROR: wrong operation. Accepted values: ['blank','clip','crop','cut','extend','mirror','mom0','shuffle','toint']. Aborting")
    if operation in ['shuffle']: #if the datacube must be shuffled
        if 'mom1map' in kwargs: #if the moment 1 map name is in kwargs
            mom1map=kwargs['mom1map'] #store the moment 1 map path from the input kwargs
            if mom1map[0] != '.': #if the moment 1 map name start with a . means that it is a path to the map (so differs from path parameter)
                mom1map=indir+mom1map
        else:
            raise ValueError("ERROR: selected operation is '{}' but no velocity field is provided: aborting!".format(operation))
    if operation in ['mom0','shuffle']: #if the datacube must be shuffled or the moment 0 map created
        if 'spectralres' in kwargs: #if the spectral resolution is in kwargs
            spectralres=kwargs['spectralres'] #store the spectral resolution from the input parameters
        else:
            raise ValueError("ERROR: selected operation is '{}' but no spectral resolution is provided: aborting!".format(operation))
    if operation in ['clip']:  #if the datacube must be clipped
        usemask=withmask #store the use mask switch from the input parameters
        if usemask: #if a 2D mask is used
            mask2d=cubedomask #store the 2D mask from the input parameters
            if mask2d == '' or mask2d is None: #if a 2D mask is not provided
                if 'mask2d' in kwargs: #if the 2D mask name is in kwargs
                    if kwargs['mask2d'] == '' or kwargs['mask2d'] is None: #if the 2D mask name is still not set
                        raise ValueError("ERROR: selected operation is '{}' with mask but no mask is provided: aborting!".format(operation))
                    else:
                        mask2d=kwargs['mask2d'] #store the default 2D mask path from the input parameters
                else:
                    raise ValueError("ERROR: selected operation is '{}' with mask but no mask is provided: aborting!".format(operation))
            if mask2d[0] != '.': #if the 2D mask name start with a . means that it is a path to the mask (so differs from path parameter)
                mask2d=indir+mask2d
    if operation in ['mom0']:  #if the datacube must be clipped
        usemask=withmask #store the use mask switch from the input parameters
        if usemask: #if a 2D mask is used
            maskcube=cubedomask #store the 3D mask from the input parameters
            if maskcube == '' or maskcube is None: #if a 2D mask is not provided
                if 'maskcube' in kwargs: #if the mask cube name is in kwargs
                    if kwargs['maskcube'] == '' or kwargs['maskcube'] is None: #if the mask cube name is still not set
                        raise ValueError("ERROR: selected operation is '{}' with mask but no mask is provided: aborting!".format(operation))
                    else:
                        maskcube=kwargs['maskcube'] #store the default mask cube path from the input parameters
                else:
                    raise ValueError("ERROR: selected operation is '{}' with mask but no mask is provided: aborting!".format(operation))
            if maskcube[0] != '.': #if the mask cube name start with a . means that it is a path to the mask (so differs from path parameter)
                maskcube=indir+maskcube
        threshold=cliplevel #store the clipping threshold from the input parameters
    if operation in ['mirror']: #if the cube must be mirrored
        x0=xrot #store the x-axis rotation central pixel
        y0=yrot #store the y-axis rotation central pixel
        z0=zrot #store the z-axis rotation central channel
    outdir=cubedooutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the output folder is not given
        outdir=indir #the output folder is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder
    outname=cubedooutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        outname=datacube.replace('.fits','_'+operation+'.fits')
        mode='update' #set the fits open mode to update
    else:
        mode='readonly'
    if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    
    #---------------   START THE FUNCTION   ---------------#
    with fits.open(datacube,mode=mode) as cube: #open the datacube
        #------------   BLANK     ------------#  
        if operation == 'blank': #if the datacube must be blanked
            #WE CHECK THE CHANNELS
            if chanmin is None or chanmin < 0: #if not given or less than 0
                warnings.warn('Starting channel wrongly set. You give {} but should be at least 0. Set to 0'.format(chanmin))
                chanmin=0 #set it to 0
            if chanmax is None: #if the upper channel is not set
                chanmax=data.shape[0]+1 #blank until the last channel
            elif chanmax > data.shape[0]: #if the higher channel is larger than the size of the data
                warnings.warn('Use choose a too high last channel ({}) but the cube has {} channels. Last channel set to {}.'.format(chanmax,data.shape[0],data.shape[0]))
                chanmax=data.shape[0]+1 #blank until the last channel
            elif chanmax < chanmin: #if the higher channel is less than the lower
                warnings.warn('Last channel ({}) lower than starting channel ({}). Last channel set to {}.'.format(chanmax,chanmin,data.shape[0]))
                chanmax=data.shape[0]+1 #blank until the last channel
            cube[0].data[chanmin:chanmax]=np.nan #blank the data
        
        #------------   CLIP     ------------# 
        if operation == 'clip': #if the datacube must be clipped
            data=cube[0].data.copy() #import the data from the data cube
            clip_cube=data.copy()*0 #initialize the clipped cube as zeros
            if usemask: #if a mask is used
                with fits.open(mask2d) as Mask:
                    mask=Mask[0].data #open the 2D mask
                if mask.shape[0] != data.shape[1] or  mask.shape[1] != data.shape[2]: #if the mask cube has different size than the data cube
                    raise ValueError('ERROR: mask and data cube has different spatial shapes: {} the mask and ({},{}) the data cube. Aborting!'.format(mask.shape,data.shape[1],data.shape[2]))
                    x=np.where(mask > 0)[0] #store the x coordinate of the non-masked pixels
                    y=np.where(mask > 0)[1] #store the y coordinate of the non-masked pixels
            else:
                x=np.where(~np.isnan(data))[1] #store the x coordinate of the non-masked pixels
                y=np.where(~np.isnan(data))[2] #store the y coordinate of the non-masked pixels
            for i,j in zip(x,y): #run over the pixels
                spectrum=data[:,i,j] #extract the spectrum
                peak=np.nanmax(spectrum) #define the peak of the gaussian as the maximum of the spectrum
                spectrum[spectrum<(peak*(1-threshold))]=0 #clip the spectrum at the threshold of the max 
                clip_cube[:,i,j]=spectrum #store the result in the clipped cube
            cube[0].data=clip_cube #copy the clipped cube into the data cube
               
        #------------   CROP     ------------#
        if operation == 'crop': #if the datacube must be cropped
            data=cube[0].data.copy() #store the data
            xlim=np.where(~np.isnan(data))[1] #select the extreme x coordinates of non-NaN values
            ylim=np.where(~np.isnan(data))[2] #select the extreme y coordinates of non-NaN values
            data=data[:,np.min(xlim):np.max(xlim),np.min(ylim):np.max(ylim)] #crop the data
            wcs=WCS(cube[0].header) #store the wcs
            wcs=wcs[:,np.min(xlim):np.max(xlim),np.min(ylim):np.max(ylim)] #crop the wcs
            newheader=wcs.to_header() #write the wcs into a header
            cube[0].header['CRPIX1']=newheader['CRPIX1'] #update the header
            cube[0].header['CRPIX2']=newheader['CRPIX2'] #update the header
            cube[0].data=data #update the data

        #------------   CUT     ------------#   
        if operation == 'cut': #if the datacube must be cutted
            data=cube[0].data.copy() #store the data
            #WE CHECK THE CHANNELS
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
            if cutbox is None: #if no box is given
                xmin=ymin=0 #start from 0
                xmax=data.shape[2] #select unti the last x-pixel
                ymax=data.shape[1] #select unti the last y pixel
            elif len(cutbox) != 4: #if the box has the wring size
                raise ValueError('ERROR: Please provide the box in the format [xmin,xmax,ymin,ymax]. Aborting!')
            else: #store the spatial box from the input
                xmin=cutbox[0]
                xmax=cutbox[1]
                ymin=cutbox[2]
                ymax=cutbox[3]
                if xmin < 0: #if xmin is negative
                    warnings.warn('Lower x limit is negative ({}): set to 0.'.format(xmin))
                    xmin=0 #set it to 0
                if xmax > data.shape[2]: #if xmax is too high
                    warnings.warn('Max x is too high ({}): set to the size of x.'.format(xmax))
                    xmax=data.shape[2] #set it to size of data
                if ymin < 0: #if ymin is negative
                    warnings.warn('Lower y limit is negative ({}): set to 0.'.format(ymin))
                    ymin=0 #set it to 0
                if ymax > data.shape[1]: #if ymax is too high
                    warnings.warn('Max y is too high ({}): set to the size of y.'.format(ymax))
                    ymax=data.shape[1] #set it to size of data
            for i in range(len(cube)): #for each HDU of the fits file
                if type(cube[i]) == fits.hdu.image.PrimaryHDU: #if the HDU is a Primary HDU
                    cube[i].data=cube[i].data[chanmin:chanmax,ymin:ymax,xmin:xmax] #extract the  subcube
                    wcs=WCS(cube[0].header) #store the wcs
                    wcs=wcs[:,ymin:ymax,xmin:xmax] #crop the wcs
                    newheader=wcs.to_header() #write the wcs into a header
                    cube[i].header['CRPIX1']=newheader['CRPIX1'] #update the header
                    cube[i].header['CRPIX2']=newheader['CRPIX2'] #update the header
                    if 'CRVAL3' in cube[i].header and 'CDELT3' in cube[i].header: #if the spectral keywords exist
                        cube[i].header['CRVAL3']=cube[i].header['CRVAL3']+chanmin*cube[0].header['CDELT3'] #recalculate the spectral axis 
                    else:
                        raise ValueError('ERROR: no spectral keywords in the header. Cannot recalculate the spectral axis: aborting') 
                else:
                    cube[i].data=cube[i].data[chanmin:chanmax] #extract the  subcube
        
        #------------   EXTEND     ------------# 
        if operation == 'extend': #if the datacube must be cutted
            if value == 'blank': #if the value is blank
                value=np.nan #set it to blank
            wcs=WCS(cube[0].header) #store the wcs
            for i in range(len(cube)): #for each HDU of the fits file
                if type(cube[i]) == fits.hdu.image.PrimaryHDU: #if the HDU is a Primary HDU
                    if addchan < 0: #if the number of channels is less than 0
                        for j in range(abs(addchan)):
                            newplane=np.ones((1,cube[0].data.shape[1],cube[0].data.shape[2]))*value #create the new plane
                            cube[i].data=np.concatenate((newplane,cube[i].data)) #concatenate the plane to the left
                        if 'CRPIX3' in cube[i].header: #if the spectral keyword exists
                            cube[i].header['CRPIX3']=cube[i].header['CRPIX3']+abs(addchan) #recalculate the spectral axis 
                        else:
                            raise ValueError('ERROR: no spectral keywords in the header. Cannot recalculate the spectral axis: aborting')
                    else:
                        for j in range(addchan):
                            newplane=np.ones((1,cube[0].data.shape[1],cube[0].data.shape[2]))*value #create the new plane
                            cube[i].data=np.concatenate((cube[i].data,newplane)) #concatenate the plane to the left
        
        #------------   MIRROR     ------------# 
        if operation == 'mirror': #if the datacube must be mirrored
            data=cube[0].data #store the data
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
            cube[0].data=data #update the data
        
        #------------   MOM0     ------------# 
        if operation == 'mom0': #if the moment 0 map must be computed
            if usemask: #if a mask is used
                with fits.open(maskcube) as Mask:
                    mask=Mask[0].data #open the 3D mask
                    if mask.shape[0] != cube[0].data.shape[0] or mask.shape[1] != cube[0].data.shape[1] or mask.shape[2] != cube[0].data.shape[2]: #if the mask cube has different size than the data cube
                        raise ValueError('ERROR: mask cube and data cube has different shapes: {} the mask cube and {} the data cube. Aborting!'.format(mask.shape,cube[0].data.shape))
                    mask[mask>1]=1 #convert to a binary mask
                cube[0].data=cube[0].data*mask
            #WE CHECK THE CHANNELS
            if chanmin is None or chanmin < 0: #if not given or less than 0
                warnings.warn('Starting channel wrongly set. You give {} but should be at least 0. Set to 0'.format(chanmin))
                chanmin=0 #set it to 0
            if chanmax is None: #if the upper channel is not set
                chanmax=data.shape[0]+1 #integrate until the last channel
            elif chanmax > data.shape[0]: #if the higher channel is larger than the size of the data
                warnings.warn('Use choose a too high last channel ({}) but the cube has {} channels. Last channel set to {}.'.format(chanmax,data.shape[0],data.shape[0]))
                chanmax=data.shape[0]+1 #integrate until the last channel
            elif chanmax < chanmin: #if the higher channel is less than the lower
                warnings.warn('Last channel ({}) lower than starting channel ({}). Last channel set to {}.'.format(chanmax,chanmin,data.shape[0]))
                chanmax=data.shape[0]+1 #integrate until the last channel
            if spectralres is None:
                if 'CDELT3' in cube[0].header: #if the header has the starting spectral value
                    spectralres=cube[0].header['CDELT3'] #store the starting spectral value
                else:
                    raise ValueError('ERROR: no spectral resolution was found: aborting')
            if 'CRVAL3' in cube[0].header: #if the header has the starting spectral value
                v0=cube[0].header['CRVAL3'] #store the starting spectral value
            else:
                raise ValueError('ERROR: no spectral information was found: aborting')
            if 'CUNIT3' in cube[0].header: #if the spectral unit is in the header
                specunits=cube[0].header['CUNIT3'] #store the spectral unit
            else:
                specunits='m/s' #set the first channel to 0 km/s
                if verbose:
                    warnings.warn('No spectral unit was found: spectral unit set to m/s!')
            data=np.nansum(cube[0].data[chanmin:chanmax,:],axis=0)*spectralres #calculate the moment 0 map
            wcs=WCS(cube[0].header,naxis=2) #store the WCS
            header=wcs.to_header() #convert the WCS into a header
            if 'BUNIT' in cube[0].header: #if the spectral unit is in the header
                header['BUNIT']=cube[0].header['BUNIT']+'*'+specunits #store the units of the moment 0 map
            else:
                header['BUNIT']='Jy/beam*m/s' #set the first channel to 0 km/s
                if verbose:
                    warnings.warn('No flux unit was found: flux density unit set to Jy/beam*m/s!')
            hdu=fits.PrimaryHDU(data.astype('float32'),header=header) #create the primary HDU
            mom0=fits.HDUList([hdu]) #make the HDU list
            mom0.writeto(outname,overwrite=True) #save the moment 0 map
            return #exit the function
            
        #------------   SHUFFLE     ------------# 
        if operation == 'shuffle': #if the cube must be shuffled
            with fits.open(mom1map) as m1: #open the moment 1 map
                mom1=m1[0].data #store the velocity field
                if m1[0].header['BUNIT'] == 'm/s': #if the units are m/s
                    mom1=mom1/1000 #convert to km/s
            data=cube[0].data*np.nan #initialize the shuffled cube
            header=cube[0].header #store the header
            if spectralres is None:
                if 'CDELT3' in header: #if the header has the starting spectral value
                    spectralres=header['CDELT3'] #store the starting spectral value
                else:
                    raise ValueError('ERROR: no spectral resolution was found: aborting')
            if 'CRVAL3' in header: #if the header has the starting spectral value
                v0=header['CRVAL3'] #store the starting spectral value
            else:
                v0=0 #set the first channel to 0 km/s
                if verbose:
                    warnings.warn('No spectral information was found: starting velocity set to 0 km/s!')
            if 'CUNIT3' in header: #if the spectral unit is in the header
                specunits=header['CUNIT3'] #store the spectral unit
            else:
                specunits='m/s' #set the first channel to 0 km/s
                if verbose:
                    warnings.warn('No spectral unit was found: spectral unit set to m/s!')
            if specunits == 'm/s': #if the spectral units are m/s
                v0=v0/1000 #convert the starting velocity to km/s
            nchan=data.shape[0] #store the number of channels
            v=np.arange(v0,v0+nchan*spectralres,spectralres) #define the spectral axis
            X=np.where(~np.isnan(mom1))[1] #store the non-NaN x coordinates
            Y=np.where(~np.isnan(mom1))[0] #store the non-NaN y coordinates
            cen=nchan//2 #define the central channel
            for x,y in zip(X,Y): #run over the pixels
                loc=np.argmin(abs(v-mom1[y,x])) #define the spectral channel center of shuffle
                for z in range(nchan): #run over the spectral axis
                    n=round(loc-cen+z) #define the number of channels to be shuffled
                    if (n < 0) or (n >= nchan): #if negative or grater than the number of channels
                        pass #do nothing
                    else:
                        data[z,y,x]=cube[0].data[n,y,x] #perform the shuffle           
            cube[0].data=data #store the shuffle data
            cube[0].header['CRPIX3']=cen #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred
            cube[0].header['CRVAL3']=0. #update the header so that the velocity axis is 0 at the pixel at which profiles have been centred
            
        #------------   TOINT     ------------#     
        if operation == 'toint': #if the datacube must be converted into an integer cube
            for i in range(len(cube)): #for each HDU of the fits file
                cube[i].data[np.isnan(cube[i].data)]=0 #set to 0 the nans
                cube[i].data=cube[i].data.astype(int) #convert data into integers

        if mode == 'readonly': #if the open mode is read only
            cube.writeto(outname,overwrite=True) #write the new cube
                        
#############################################################################################
def cubestat(datacube='',**kwargs):
    """Calculate the detection limit of a data cube and (optional) its rms, spectral resolution, beam major axis, beam minor axis, beam position angle and beam area. It also computes the errors on rms and sensitivity.

    Args:
        datacube (string): string with name or path+name of the fits data cube

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
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    if datacube == '' or datacube is None:
        raise ValueError('ERROR: datacube is not set: aborting!')
    if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the data cube is set: aborting!')
            else:
                datacube=kwargs['path']+datacube
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')
    #CHECK THE KWARGS#
    if 'pixunits' in kwargs: #if the spatial units are in the input kwargs
        pixunits=kwargs['pixunits'] #store the spatial units from the input kwargs
    else:
        pixunits=None #set it to None   
    if 'specunits' in kwargs: #if the spectral units are in the input kwargs
        specunits=kwargs['specunits'] #store the spectral units from the input kwargs
    else:
        specunits=None #set it to None   
    if 'fluxunits' in kwargs: #if the flux units are in the input kwargs
        fluxunits=kwargs['fluxunits'] #store the flux units from the input kwargs
    else:
        fluxunits=None #set it to None   
    if 'pixelres' in kwargs: #if the pixel resolution is in the input kwargs
        pixelres=kwargs['pixelres'] #store the pixel resolution from the input kwargs
    else:
        pixelres=None #set it to None
    if 'spectralres' in kwargs: #if the spectral resolution is in the input kwargs
        spectralres=kwargs['spectralres'] #store the spectral resolution from the input kwargs
    else:
        spectralres=None #set it to None
    if 'bmaj' in kwargs: #if the beam major axis is in the input kwargs
        bmaj=kwargs['bmaj'] #store the beam major axis from the input kwargs
    else:
        bmaj=None #set it to None
    if 'bmin' in kwargs: #if the beam minor axis is in the input kwargs
        bmin=kwargs['bmin'] #store the beam minor axis from the input kwargs
    else:
        bmin=None #set it to None
    if 'bpa' in kwargs: #if the beam position angle is in the input kwargs
        bpa=kwargs['bpa'] #store the beam position angle from the input kwargs
    else:
        bpa=None #set it to None
    if 'rms' in kwargs: #if the cube rms is provided
        rms=kwargs['rms'] #store the rms from the input kwargs
    else:
        rms=None #set it to None
    if 'nsigma' in kwargs: #if the sigma-threshold for the detection limit is in kwargs
        nsigma=kwargs['nsigma'] #store it from input kwargs
    else:
        nsigma=3 #set it to 3
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False

    #---------------   START THE FUNCTION   ---------------#
    with fits.open(datacube) as cube: #open the data cube
        data=cube[0].data #store the cube data
        header=cube[0].header #store the cube header
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
        print('The median rms per channel is: {:.2e} {}'.format(rms,fluxunits))
        print('The {}\u03C3 1-channel detection limit is: {:.2e} {} i.e., {:.2e} {}'.format(int(nsigma),nsigma*rms,fluxunits,sensitivity,coldenunits))
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
    fits1=fits1 #store the first fits filename from the input parameters
    fits2=fits2 #store the second fits filename from the input parameters
    if fits1 == '' or fits1 is None or fits2 == '' or fits2 is None:
        raise ValueError('ERROR: one ore both fits file are not set: aborting')
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
def fixmask(refcube='',masktofix='',fixmaskindir='',fixmaskoutdir='',fixmaskoutname='',**kwargs):
    """Fix a source-finding mask by removing pixel corresponding to negative detections.
    
    Args:
        refcube (string): name or path+name of the fits data cube used to fix the mask
        masktofix (string): name or path+name of the fits 3D mask to be fixed
        fixmaskindir (string): name of the input directory
        fixmaskoutdir (string): output folder name
        fixmaskoutname (string): output file name
        
    Kwargs:
        datacube (string): name or path+name of the fits data cube if refcube is not given
        maskcube (string): name or path+name of the fits 3D mask cube. Used if chanmask=True                          
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
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    datacube=refcube #store the data cube path from the input parameters
    if datacube == '' or datacube is None: #if a data cube is not given
        if 'datacube' in kwargs: #if the datacube is in kwargs
            if kwargs['datacube'] == '' or kwargs['datacube'] is None:
                raise ValueError('ERROR: datacube is not set: aborting!')
            else:
                datacube=kwargs['datacube'] #store the default data cube path from the input kwargs
        else:
            raise ValueError('ERROR: datacube is not set: aborting!')
    maskcube=masktofix #store the mask cube name fro mthe input parameters
    if maskcube == '' or maskcube is None: #if a mask cube is not given
        if 'maskcube' in kwargs: #if the datacube is in kwargs
            if kwargs['maskcube'] == '' or kwargs['maskcube'] is None:
                raise ValueError('ERROR: mask cube is not set: aborting!')
            else:
                maskcube=kwargs['maskcube'] #store the default data cube path from the input kwargs
        else:
            raise ValueError('ERROR: mask cube is not set: aborting!')
    indir=fixmaskindir #store the input folder from the input parameters
    if indir == '' or indir is None: #if the input folder is not given
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the data cube is set: aborting!')
            else:
                indir=kwargs['path']
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')
    elif not os.path.exists(indir): #if the input folder does not exist
        os.makedirs(indir) #create the folder
    if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
        datacube=indir+datacube
    if maskcube[0] != '.': #if the mask cube name start with a . means that it is a path to the cube (so differs from path parameter)
        maskcube=indir+maskcube
    outdir=fixmaskoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=indir #the outdir is the input folder
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
    with fits.open(datacube) as cube: #open the data cube
        data=cube[0].data #store its data
    with fits.open(maskcube,mode=mode) as Maskcube: #open the mask cube
        mask=Maskcube[0].data.copy() #store the mask data
        mask[data<0]=0 #fix the mask by setting to 0 the pixel where the emission is negative (hence, no source)
        Maskcube[0].data=mask #overwrite the mask data
        if mode == 'readonly': #if the open mode is read only
            Maskcube.writeto(outname,overwrite=True) #write the new mask
        
#############################################################################################
def gaussfit(cubetofit='',gaussmask='',gaussindir='',spectralres=None,linefwhm=15,amp_thresh=0,p_reject=1,
             clipping=False,threshold=0.5,errors=False,gaussoutdir='',gaussoutname='',**kwargs):
    """Fit a gaussian profile to each spaxel of a cube based on a source-finding-produced mask.

    Args:
        cubetofit (string):name or path+name of the fits data cube to be fitted
        gaussmask (string): name or path+name of the fits 2D mask to be used in the fit
        gaussindir (string):name of the inpt directory
        spectralres (float): data spectral resolution in specunits
        linefwhm (float): first guess on the fwhm of the line profile in km/s
        amp_thresh (float): amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum
        p-reject (float): p-value threshold for fit rejection. If a best-fit has p > p-reject, it will be rejected
        clipping (bool): clip the spectrum to a % of the profile peak if True
        threshold (float): clip threshold as % of the peak (0.5 is 50%). Used if clipping is True
        errors (bool): compute the errors on the best-fit if True
        gaussoutdir (string): output folder name
        gaussoutname (string): output file name
    
    Kwargs:
        datacube (string): name or path+name of the fits data cube if cubetofit is not given
        mask2d (string): name or path+name of the fits 2D mask if gaussmask is not given
        path (string): path to the data cube if the datacube is a name and not a path+name
        verbose (bool): option to print messages to terminal if True (Default: False)
 
    Returns:
        Best-fit model cube as fits file
        
    Raises:
        ValueError: If no data cube is provided
        ValueError: If no path is provided
        ValueError: If mask cube and data cube dimensions do not match
        ValueError: If no spectral information is available through the input arguments or the cube header
    """
    #CHECK THE INPUT#
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    datacube=cubetofit #store the data cube path from the input parameters
    if datacube == '' or datacube is None: #if a data cube is not given
        if 'datacube' in kwargs: #if the datacube is in kwargs
            if kwargs['datacube'] == '' or kwargs['datacube'] is None:
                raise ValueError('ERROR: datacube is not set: aborting!')
            else:
                datacube=kwargs['datacube'] #store the default data cube path from the input kwargs
        else:
            raise ValueError('ERROR: datacube is not set: aborting!')
    mask2d=gaussmask #store the 2D mask path from the input parameters
    if mask2d == '' or mask2d is None: #if a 2D mask is not given
        if 'mask2d' in kwargs: #if the 2D mask is in kwargs
            mask2d=kwargs['mask2d'] #store the default data cube path from the input kwargs
    indir=gaussindir #store the input folder from the input parameters
    if indir == '' or indir is None: #if the input folder is not given
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the data cube is set: aborting!')
            else:
                indir=kwargs['path']
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')
    elif not os.path.exists(indir): #if the input folder does not exist
        os.makedirs(indir) #create the folder
    if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
        datacube=indir+datacube
    if mask2d[0] != '.': #if the 2D mask name start with a . means that it is a path to the mask (so differs from path parameter)
        mask2d=indir+mask2d
    outdir=gaussoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the outdir is empty
        outdir=indir #the outdir is the input folder
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder    
    outname=gaussoutname #store the output name from the input parameters
    if outname == '' or outname is None:  #if the outname is empty
        outname=datacube.replace('.fits','_gaussfit.fits')  #the outname is the object name plus chanmap.pdf
    if outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    
    #---------------   START THE FUNCTION   ---------------#
    with fits.open(datacube) as cube: #open the data cube
        data=cube[0].data.copy() #import the data from the data cube
        header=cube[0].header #import the data cube header
        if mask2d == '' or mask2d is None: #if no 2D mask is given
            mask=np.ones(data.shape)
        else:
            with fits.open(mask2d) as mask2d: #open the 2D mask
                mask=mask2d[0].data #import the 2D mask data
                if mask.shape[0] != data.shape[1] or  mask.shape[1] != data.shape[2]: #if the mask cube has different size than the data cube
                    raise ValueError('ERROR: mask and data cube has different spatial shapes: {} the mask and ({},{}) the data cube. Aborting!'.format(mask.shape,data.shape[1],data.shape[2]))
        model_cube=data.copy()*0 #initialize the model cube as zeros
        
        #WE CHECK THE REQUIRED INFORMATION#
        if spectralres is None and 'CDELT3' in header: #if the spectral resolution is not given but is in the header
            spectralres=header['CDELT3'] #store the spectral resolution from the cube header
        else:
            raise ValueError('ERROR: no spectral resolution is provided or found. Aborting!')
        if 'CRVAL3' in header: #if the header has the starting spectral value
            v0=header['CRVAL3'] #store the starting spectral value
        else:
            raise ValueError('ERROR: no spectral value for starting channel was found. Aborting!')
            
        #WE PREPARE THE SPECTRAL AXIS#
        nchan=np.shape(data)[0] #store the number of channels
        if spectralres>0: #if the spectral resolution is positive
            v=np.arange(v0,v0+nchan*spectralres,spectralres) #define the spectral axis
        else:
            v=np.arange(v0+nchan*spectralres,v0,spectralres) #define the spectral axis
        if len(v) > nchan: #!! sometimes an additional channel is created. For the moment, this is a workaround
            v=v[:-1]
            
        #WE PREPARE FOR THE FIT#
        width=linefwhm/spectralres #define the first guess fwhm of the line in km/s
        x=np.where(mask > 0)[0] #store the x coordinate of the non-masked pixels
        y=np.where(mask > 0)[1] #store the y coordinate of the non-masked pixels
        
        #WE START THE FITTING ROUTINE#
        for i,j in zip(x,y): #run over the pixels
            spectrum=data[:,i,j].copy() #extract the spectrum
            peak=np.nanmax(spectrum) #define the peak of the gaussian as the maximum of the spectrum
            if peak > amp_thresh: #if the peak is above the threshold
                vpeak=v0+np.argmax(spectrum)*spectralres #define the central velocity as the velocity of the peak
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
                        model_cube[:,i,j]=total_fit(v) #store the result in the model cube
            
        cube[0].data=model_cube #copy the model cube into the data cube
        cube.writeto(outname,overwrite=True) #save the model cube 
        
#############################################################################################
def getpv(pvcube='',pvwidth=None,pvpoints=None,pvangle=None,pvchmin=0,pvchmax=None,pvoutdir='',
          savefits=False,fitsoutname='',plot=False,saveplot=False,plotoutname='',**kwargs):
    """Extract the pv slice of a cube and (optinal) save it as a fits file and/or (optional) plot it or return it as PrimaryHUD object.

    Args:
        pvcube (string): name or path+name of the fits data cube
        pvwidth (float): width of the slice in arcsec. If not given, will be thebeam size
        pvpoints (list/array): ICRS RA-DEC comma-separated coordinate of the slice points in decimal degree. If two are given ([x,y]), it is assumed they are the center of the slice. Else, they need to be the starting and ending coordinates ([xmin,xmax,ymin,ymax])
        pvangle (float): position angle of the slice in degree when two points are given. If not given, will be the object position angle
        pvchmin (int): first channel of the slice
        pvchmax (int): last channel of the slice
        pvoutdir (string): string of the output folder name
        savefits (bool): save the slice as fits file if True
        fitsoutname (string): string of the output fits file name
        plot (bool): plot the slice if True
        saveplot (string): save the plot if True
        plotoutname (string): string of the output plot file name
    
    Kwargs:
        datacube (string): name or path+name of the fits data cube if pvcube is not given
        path (string): path to the data cube if the datacube is a name and not a path+name
        spectralres (float): cube spectral resolution in km/s
        bmaj (float): beam major axis in arcsec
        pa (float): object position angle in degree
        pvsig (float): lowest contour level in terms of pvsig*rms
        pixelres (float): pixel resolution of the data in arcsec
        figure (bool): create a plot figure if True
        position (int): position of the subplot in the figure as triplet of integers (111 = nrow 1, ncol 1, index 1)
        vsys (float): object systemic velocity in km/s
        rms (float): rms of the data cube in Jy/beam as a float. If not given (None), the function tries to calculate it
        asectokpc (float): arcsec to kpc conversion to plot the spatial scale
        objname (string): name of the object
        subtitle (string): subtitle of the pv plot
        lim (list/array): list or array of plot x and y limits as [xmin,xmax,ymin,ymax]. They will replace the default limits
        pv_ctr (list/array): contours level. They will replace the default levels                                    
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
        ValueError: If no output folder is given
    """
    #CHECK THE INPUT#
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    datacube=pvcube #store the data cube path from the input parameters
    if datacube == '' or datacube is None: #if a data cube is not given
        if 'datacube' in kwargs: #if the datacube is in kwargs
            if kwargs['datacube'] == '' or kwargs['datacube'] is None:
                raise ValueError('ERROR: datacube is not set: aborting!')
            else:
                datacube=kwargs['datacube'] #store the default data cube path from the input kwargs
        else:
            raise ValueError('ERROR: datacube is not set: aborting!')
    if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the data cube is set: aborting!')
            else:
                datacube=kwargs['path']+datacube
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')
    if pvwidth is None: #if width is not given
        if verbose:
            warnings.warn('Pv width not set. Trying to set it to the beam major axis!')
        if 'bmaj' in kwargs: #if the beam major axis is in kwargs
            if kwargs['bmaj'] == '' or kwargs['bmaj'] is None:
                warnings.warn('Beam major axis not set. Pv width will be 1 pixel!')
                pvwidth=1
            else:
                pvwidth=kwargs['bmaj']*u.arcsec #store the beam major axis from the input kwargs
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
    if outdir == '' or outdir is None:  #if the output folder is not given
        if 'path' in kwargs: #if the path to the output folder is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no output folder is set: aborting!')
            else:
                outdir=kwargs['path']
        else:
            raise ValueError('ERROR: no output folder is set: aborting!') 
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder   
    pvoutname=fitsoutname #store the output name from the input parameters
    if pvoutname == '' or pvoutname is None:  #if the outname is empty
        pvoutname=datacube.replace('.fits','_pvslice.fits') #the outname is the object name plus'_pvslice.fits
    plotoutname=plotoutname #store the output name from the input parameters
    if plotoutname == '' or plotoutname is None:  #if the outname is empty
        plotoutname=datacube.replace('.fits','_pvslice.pdf') #the outname is the object name plus'_pvslice.fits
    #CHECK THE KWARGS#
    if 'pvsig' in kwargs: #if the nsigma for the detection limit as nsigma*rms is in the input kwargs
        pvsig=kwargs['pvsig'] #store the nsigma for the detection limit as nsigma*rms from the input kwargs
    else:
        pvsig=3 #set it to 3
    if 'pixelres' in kwargs: #if the pixel resolution is in the input kwargs
        pixelres=kwargs['pixelres'] #store the pixel resolution from the input kwargs
    else:
        pixelres=None #set it to None
    if 'figure' in kwargs: #if the do figure option is in the input kwargs
        figure=kwargs['figure'] #store the do figure option from the input kwargs
    else:
        figure=False #set it to False
    if 'position' in kwargs: #if the subplot position is in the input kwargs
        position=kwargs['position'] #store the subplot position from the input kwargs
    else:
        position=111 #set it to 111
    if 'rms' in kwargs: #if the rms is in the input kwargs
        rms=kwargs['rms'] #store the rms from the input kwargs
    else:
        rms=None #set it to None
    if 'vsys' in kwargs: #if the systemic velocity is in the input kwargs
        vsys=kwargs['vsys'] #store the systemic velocity from the input kwargs
    else:
        vsys=None #set it to None
    if 'asectokpc' in kwargs: #if the arcsec-to-kpc conversion is in the input kwargs
        asectokpc=kwargs['asectokpc'] #store the arcsec-to-kpc conversion from the input kwargs
    else:
        asectokpc=None #set it to None
    if 'objname' in kwargs: #if the object name is in the input kwargs
        objname=kwargs['objname'] #store the object name from the input kwargs
        if objname is None: #if in kwargs but not set
            objname='' #set it to empty
    else:
        objname='' #set it to empty
    if 'subtitle' in kwargs: #if the arcsec-to-kpc conversion is in the input kwargs
        subtitle=kwargs['subtitle'] #store the arcsec-to-kpc conversion from the input kwargs
    else:
        subtitle=None #set it to None
    if 'lim' in kwargs: #if the plot limits are in the input kwargs
        lim=kwargs['lim'] #store the plot limits from the input kwargs
    else:
        lim=None #set it to None
    if 'pv_ctr' in kwargs: #if the moment 0 contour levels are in the input kwargs
        pv_ctr=kwargs['pv_ctr'] #store the moment 0 contour levels from the input kwargs
    else:
        pv_ctr=None #set it to None
    if 'ctr_width' in kwargs: #if the contours width is in the input kwargs
        ctr_width=kwargs['ctr_width'] #store the contours width from the input kwargs
    else:
        ctr_width=2 #set it to 2   

    #---------------   START THE FUNCTION   ---------------#
    with fits.open(datacube) as cube: #open the data cube
        data=cube[0].data #store the data.
        header=cube[0].header #store the cube header
    wcs=WCS(header) #store the wcs information
    
    #WE CHECK IF THE REQUIRED INFORMATION ARE PROVIDED#
    #-----------   SPATIAL AXIS    -----------#
    if pixelres is None: #if the pixelres is not given
        if 'CUNIT1' in header: #if the spatial unit is in the header
            spaceunits=header['CUNIT1'] #store the spatial unit
            if 'CDELT1' in header: #if the spatial resolution is in the header
                spaceres=abs(header['CDELT1']) #store the spatial resolution from the header
                if spaceunits == 'deg': #if the spatial unit is degree
                    pixelres=spaceres*3600 #convert into arcsec
                elif spaceunits == 'arcmin': #if the spatial unit is arcmin
                    pixelres=spaceres*60 #convert into arcsec
                elif spaceunits == 'arcsec': #if the spatial unit is arcsec
                    pixelres=spaceres #do nothing
        elif verbose:
            warnings.warn('No spatial unit was found: unable to calculate the pixel resolution! Path length set to 0.5 deg')
    
    #WE DEFINE THE PATH OF THE SLICE#
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
        
    #WE EXTRACT THE PVSLICE AND REFER THE SPATIAL AXIS TO THE SLICE CENTER#    
    pv=extract_pv_slice(data[chmin:chmax,:,:],pvpath,wcs=wcs[chmin:chmax,:,:]) #extract the pv slice
    pv.header['CRPIX1']=round(pv.header['NAXIS1']/2)+1 #fix the header in order to have the distance from the center as spatial dimension
    
    #WE SAVE THE SLICE IF NEEDED#
    if savefits: #if the slice must be saved
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
            ctr=np.power(pvsig,np.arange(1,9,2)) #4 contours level between nsigma and nsigma^8
            if verbose:
                print('Contours level: {} Jy/beam'.format(ctr*rms))
        else:
            ctr=pv_ctr #use those in input
        #WE DO THE PLOT#
        im=ax.imshow(data,cmap='Greys',norm=norm,aspect='auto') #plot the pv slice in units of rms
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        if pv.header['CUNIT2']=='m/s': #if the spectral units are m/s
            ax.coords[1].set_format_unit(u.km/u.s) #convert to km/s
        ax.coords[0].set_format_unit(u.arcmin) #convert x units to arcmin
        ax.set_xlabel('Offset from center [arcmin]') #set the x-axis label
        ax.set_ylabel('Velocity [km/s]') #set the y-axis label
        ax.axvline(x=pv.header['CRPIX1'],linestyle='-.',color='black') #draw the galactic center line
        #WE ADD THE CONTOURS#
        ax.contour(data/rms,levels=ctr,cmap='gnuplot',linewidths=ctr_width,linestyles='solid') #add the positive contours
        ax.contour(data/rms,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width,linestyles='dashed') #add the negative contours
        #WE ADD ANCILLARY INFORMATION#
        if vsys is not None: #if the systemic velocity is given
            if pv.header['CUNIT2']=='m/s': #if the spectral units are m/s
                vsys=vsys*1000 #convert the systemic velocity into m/s
            ax.axhline(y=((vsys-pv.header['CRVAL2'])/pv.header['CDELT2'])+pv.header['CRPIX2']-1,linestyle='--',color='black') #draw the systemic velocity line. -1 is due to python stupid 0-couting
        else:
            ax.axhline(y=((0-pv.header['CRVAL2'])/pv.header['CDELT2'])-pv.header['CRPIX2'],linestyle='--',color='black') #draw the systemic velocity line
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            kpcline=10/(asectokpc*pixelres) #length of the 10 kpc line in pixel
            kpcline=kpcline/(xlim[1]-xlim[0]) #lenght of the 10 kpc line in axes size
            ax.hlines(topmargin-0.015,leftmargin,leftmargin+kpcline,color='black',linewidth=2,transform=ax.transAxes) #add the 10 kpc line
            ax.text(leftmargin+(kpcline/2),topmargin,'10 kpc',ha='center',transform=ax.transAxes)
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
    else: #if no plot has to be made
        return pv #return the pvslice
         
#############################################################################################
def plotmom(which='all',mom0map='',mom1map='',mom2map='',plotmomoutdir='',plotmomoutname='',save=False,**kwargs):
    """Plot the moment maps (all or just one) of a fits cube and (optionally) save the plot.

    Args:
        which (string): what the function has to plot between:
            all, plot moment 0, moment 1 and moment 2 map in a single 1-row 3-columns figure
            mom0, plot moment 0 map
            mom1, plot moment 1 map
            mom2, plot moment 2 map
        mom0map (string): name or path+name of the fits moment 0
        mom1map (string): name or path+name of the fits moment 1
        mom2map (string): name or path+name of the fits moment 2
    plotmomoutdir (string): output folder name
    plotmomoutname (string): output file name
    save (bool): save the plot if True
    
    Kwargs:
        path (string): path to the moment map if the momXmap is a name and not a path+name
        pbcorr (bool): apply the primary beam correction if True. Note that in that case you must supply a beam cube (Default: False)
        beamcube (string): name or path+name of the fits beam cube if pbcorr is True
        use_cube (bool): use a data cube to get information like the rms, the spectral resolution and the beam if True (Default: False). Note that in that case you must supply a data cube
        datacube (string): name or path+name of the fits data cube if use_cube is True
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
        mom0_ctr (list/array): contours level. They will replace the default levels (Default: None)
        lim (list/array): list or array of plot x and y limits as [xmin,xmax,ymin,ymax]. They will replace the default limits (Default: None)
        ctr_width (float): line width of the contours (Default: 2)
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
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    if which not in ['all','mom0','mom1','mom2']: #if wrong data is given
        raise ValueError("ERROR: wrong data selection. Accepted values: ['all','mom0','mom1','mom2']. Aborting")
    if which in ['all','mom0']: #if all moment maps must be plotted or only the moment 0
        if mom0map == '' or mom0map is None: #if a data cube is not given
            raise ValueError('ERROR: moment 0 map is not set: aborting!')
        if mom0map[0] != '.': #if the moment 0 map name start with a . means that it is a path to the map (so differs from path parameter)
            if 'path' in kwargs: #if the path to the data cube is in kwargs
                if kwargs['path'] == '' or kwargs['path'] is None:
                    raise ValueError('ERROR: no path to the moment 0 map is set: aborting!')
                else:
                    mom0map=kwargs['path']+mom0map
            else:
                raise ValueError('ERROR: no path to the moment 0 map is set: aborting!')  
        #------------   PB CORRECTION     ------------#
        if 'pbcorr' in kwargs: #check if the pb correction keyword is in the kwargs
            pbcorr=kwargs['pbcorr'] #store the apply pb correction option
        if pbcorr: #if the primary beam correction is applied
            #------------   IS IN THE INPUT?     ------------# 
            if 'beamcube' in kwargs: #if the beamcube is in kwargs
                if kwargs['beamcube'] == '' or kwargs['beamcube'] is None: #if no beam cube is provided
                    if verbose:
                        warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')
                    pbcorr=False #set the pbcorr option to False
                else:
                    beamcube=kwargs['beamcube'] #store the data cube path from the input parameters.
                    #------------   IS IN THE INPUT, BUT THE PATH?     ------------# 
                    if beamcube[0] != '.': #if the primary beam cube name start with a . means that it is a path to the cube (so differs from path parameter)
                        if 'path' in kwargs: #if the path to the beam cube is in kwargs
                            if kwargs['path'] == '' or kwargs['path'] is None:
                                if verbose:
                                    warnings.warn('You have not provided a path to the beam cube. Cannot apply primary beam correction!')
                                pbcorr=False #set the pbcorr option to False
                            else:
                                indir=kwargs['path']
                                beamcube=indir+beamcube
                        else:
                            if verbose:
                                warnings.warn('You have not provided a path to the beam cube. Cannot apply primary beam correction!')
                            pbcorr=False #set the pbcorr option to False
            else:
                if verbose:
                    warnings.warn('You have not provided a beam cube. Cannot apply primary beam correction!')
                pbcorr=False #set the pbcorr option to False
    if which in ['all','mom1']: #if all moment maps must be plotted or only the moment 1 map
        if mom1map == '' or mom1map is None: #if a model cube is not given
            raise ValueError('ERROR: moment 1 map is not set: aborting!')
        if mom1map[0] != '.': #if the moment 1 map name start with a . means that it is a path to the map (so differs from path parameter)
            if 'path' in kwargs: #if the path to the data cube is in kwargs
                if kwargs['path'] == '' or kwargs['path'] is None:
                    raise ValueError('ERROR: no path to the moment 1 map is set: aborting!')
                else:
                    mom1map=kwargs['path']+mom1map
            else:
                raise ValueError('ERROR: no path to the moment 1 map is set: aborting!')  
        if 'vsys' in kwargs: #if the beam major axis is in the input kwargs
            vsys=kwargs['vsys'] #store the beam major axis from the input kwargs
        else:
            vsys=None #set it to None
    if which in ['all','mom2']: #if all moment maps must be plotted or only the moment 2 map
        if mom2map == '' or mom2map is None: #if a data cube is not given
            raise ValueError('ERROR: moment 2 map is not set: aborting!')
        if mom2map[0] != '.': #if the moment 2 map name start with a . means that it is a path to the map (so differs from path parameter)    
            if 'path' in kwargs: #if the path to the data cube is in kwargs
                if kwargs['path'] == '' or kwargs['path'] is None:
                    raise ValueError('ERROR: no path to the moment 2 map is set: aborting!')
                else:
                    mom2map=kwargs['path']+mom2map
            else:
                raise ValueError('ERROR: no path to the moment 2 map is set: aborting!')
    outdir=plotmomoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the output folder is not given
        if 'path' in kwargs: #if the path to the output folder is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no output folder is set: aborting!')
            else:
                outdir=kwargs['path']
        else:
            raise ValueError('ERROR: no output folder is set: aborting!') 
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
    if 'use_cube' in kwargs: #if the use cube option is in the input kwargs
        use_cube=kwargs['use_cube'] #store the use cube option from the input kwargs
    else:
        use_cube=False #set it to False
    if use_cube: #if the use cube option is True
        if 'datacube' in kwargs: #check if the data cube is in the kwargs
            datacube=kwargs['datacube'] #store the data cube from the input kwargs
            if datacube == '' or datacube is None: #if a data cube is not given
                raise ValueError('ERROR: you set to use a data cube but no data cube is provided: aborting!')
            if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
                if 'path' in kwargs: #if the path to the data cube is in kwargs
                    if kwargs['path'] == '' or kwargs['path'] is None:
                        raise ValueError('ERROR: no path to the data cube is set: aborting!')
                    else:
                        datacube=kwargs['path']+datacube
                else:
                    raise ValueError('ERROR: no path to the data cube is set: aborting!')
        else:
            raise ValueError('ERROR: you set to use a data cube but no data cube is provided: aborting!')
    if 'pixunits' in kwargs: #if the spatial units are in the input kwargs
        pixunits=kwargs['pixunits'] #store the spatial units from the input kwargs
    else:
        pixunits=None #set it to None   
    if 'specunits' in kwargs: #if the spectral units are in the input kwargs
        specunits=kwargs['specunits'] #store the spectral units from the input kwargs
    else:
        specunits=None #set it to None   
    if 'spectralres' in kwargs: #if the spectral resolution is in the input kwargs
        spectralres=kwargs['spectralres'] #store the spectral resolution from the input kwargs
    else:
        spectralres=None #set it to None       
    if 'bmaj' in kwargs: #if the beam major axis is in the input kwargs
        bmaj=kwargs['bmaj'] #store the beam major axis from the input kwargs
    else:
        bmaj=None #set it to None
    if 'bmin' in kwargs: #if the beam major axis is in the input kwargs
        bmin=kwargs['bmin'] #store the beam major axis from the input kwargs
    else:
        bmin=None #set it to None
    if 'bpa' in kwargs: #if the beam major axis is in the input kwargs
        bpa=kwargs['bpa'] #store the beam major axis from the input kwargs
    else:
        bpa=None #set it to None
    if 'nsigma' in kwargs: #if the nsigma for the detection limit as nsigma*rms is in the input kwargs
        nsigma=kwargs['nsigma'] #store the nsigma for the detection limit as nsigma*rms from the input kwargs
    else:
        nsigma=3 #set it to 3
    if 'rms' in kwargs: #if the rms is in the input kwargs
        rms=kwargs['rms'] #store the rms from the input kwargs
    else:
        rms=None #set it to None
    if 'pixelres' in kwargs: #if the pixel resolution is in the input kwargs
        pixelres=kwargs['pixelres'] #store the pixel resolution from the input kwargs
    else:
        pixelres=None #set it to None
    if 'asectokpc' in kwargs: #if the arcsec-to-kpc conversion is in the input kwargs
        asectokpc=kwargs['asectokpc'] #store the arcsec-to-kpc conversion from the input kwargs
    else:
        asectokpc=None #set it to None
    if 'objname' in kwargs: #if the object name is in the input kwargs
        objname=kwargs['objname'] #store the object name from the input kwargs
        if objname is None: #if in kwargs but not set
            objname='' #set it to empty
    else:
        objname='' #set it to empty
    if 'subtitle' in kwargs: #if the arcsec-to-kpc conversion is in the input kwargs
        subtitle=kwargs['subtitle'] #store the arcsec-to-kpc conversion from the input kwargs
    else:
        subtitle=None #set it to None
    if 'wcs' in kwargs: #if the wcs is in the input kwargs
        wcs=kwargs['wcs'] #store the wcs from the input kwargs
    else:
        wcs=None #set it to None
    if 'position' in kwargs: #if the subplot position is in the input kwargs
        position=kwargs['position'] #store the subplot position from the input kwargs
    else:
        position=111 #set it to 111
    if 'mom0_ctr' in kwargs: #if the moment 0 contour levels are in the input kwargs
        mom0_ctr=kwargs['mom0_ctr'] #store the moment 0 contour levels from the input kwargs
    else:
        mom0_ctr=None #set it to None
    if 'lim' in kwargs: #if the plot limits are in the input kwargs
        lim=kwargs['lim'] #store the plot limits from the input kwargs
    else:
        lim=None #set it to None
    if 'ctr_width' in kwargs: #if the contours width is in the input kwargs
        ctr_width=kwargs['ctr_width'] #store the contours width from the input kwargs
    else:
        ctr_width=2 #set it to 2    

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
        with fits.open(mom0map) as m0: #open the moment 0 map
            mom0=m0[0].data #store the data
            mom0header=m0[0].header #store the header
        #WE TRY TO GET THE PIXEL RESOLUTION#
        if pixelres is None and 'CDELT1' in mom0header: #if the pixelres is not given and is in the header
            pixelres=mom0header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if pixunits is None and 'CUNIT1' in mom0header: #if the spatial unit is in the header
            pixunits=mom0header['CUNIT1'] #store the spatial unit
        #WE DO THE PB CORRECTION IF NEEDED#    
        if pbcorr: #if the primary beam correction is applied
            with fits.open(beamcube) as pb_cube: #open the primary beam cube
                pb_slice=pb_cube[0].data[np.array(pb_cube[0].shape[0]/2).astype(int)] #extract the central plane
            mom0=mom0/pb_slice #apply the pb correction
            momunit=mom0header['BUNIT'].replace('/beam'.casefold(),'')
        else:
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
        if mom0header['BUNIT'].casefold()=='jy/beam*m/s' or mom0header['BUNIT'].casefold()=='jy*m/s': #if the units are in m/s
            mom0=mom0/1000 #convert to km/s
        mom0[mom0==0]=np.nan #convert the moment 0 zeros into nan
        
    #WE PREPARE THE MOMENT 1 MAP#
    if which == 'all' or which == 'mom1': #if all moment maps or moment 1 map must be plotted
        #WE GET THE DATA AND DO A SIMPLE CONVERSION#
        with fits.open(mom1map) as m1: #open the moment 1 map
            mom1=m1[0].data #store the data
            mom1header=m1[0].header #store the header
            if mom1header['BUNIT']=='m/s': #if the units are in m/s
                mom1=mom1/1000 #convert to km/s
        #WE TRY TO GET THE PIXEL RESOLUTION 
        if pixelres is None and 'CDELT1' in mom1header: #if the pixelres is not given and is in the header
            pixelres=mom1header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
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
        with fits.open(mom2map) as m2: #open the moment 2 map
            mom2=m2[0].data #store the data
            mom2header=m2[0].header #store the header
            if mom2header['BUNIT']=='m/s': #if the units are in m/s
                mom2=mom2/1000 #convert to km/s
        #WE TRY TO GET THE PIXEL RESOLUTION# 
        if pixelres is None and 'CDELT1' in mom2header: #if the pixelres is not given and is in the header
            pixelres=mom2header['CDELT1'] #store the spatial resolution from the header
        elif pixelres is None:
            if verbose:
                warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
        if pixunits is None and 'CUNIT1' in mom2header: #if the spatial unit is in the header
            pixunits=mom2header['CUNIT1'] #store the spatial unit
        #WE CALCULATE THE FWHM           
        disp=np.sqrt(np.nanmedian(mom2)*8*np.log(2)) #calculate the median velocity dispersion (FWHM)
    
    #WE CONVERT THE BEAM TO ARCSEC
    if pixunits == 'deg': #if the spatial units are deg
        bmaj=bmaj*3600 #convert the beam major axis in arcsec
        bmin=bmin*3600 #convert the beam minor axis in arcsec
    elif pixunits == 'arcmin': #if the spatial units are arcmin
        bmaj=bmaj*60 #convert the beam major axis in arcsec
        bmin=bmin*60 #convert the beam minor axis in arcsec
    beamarea=1.13*(bmin*bmaj) #calculate the beam area 
            
    #WE CONVERT THE SPATIAL DIMENSION TO ARCSEC
    if pixelres is not None and pixunits == 'deg': #if the spatial unit is degree
        pixelres=pixelres*3600 #convert into arcsec
    elif pixelres is not None and pixunits == 'arcmin': #if the spatial unit is arcmin
        pixelres=pixelres*60 #convert into arcsec
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
        im=ax.imshow(mom0,cmap='Greys',norm=norm,aspect='equal') #plot the moment 0 map
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        #WE ADD THE CONTOURS#
        if momunit == 'cm$^{-2}$': #if the moment 0 is column density
            ax.contour(mom0,levels=ctr,cmap='Greys_r',linewidths=ctr_width,linestyles='solid') #add the contours
        else:
            ax.contour(mom0,levels=ctr,cmap='Greys_r',linewidths=ctr_width,linestyles='solid',norm=norm) #add the contours
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
            ax.text(leftmargin,bottommargin,'Detection limit: {:.2e} {}'.format(sens*10.**(18),momunit),transform=ax.transAxes) #add the information of the detection limit
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            pxbeam=np.array([bmaj,bmin])/pixelres #beam in pixel
            box=patch.Rectangle((xlim[1]-2*pxbeam[0],ylim[0]),2*pxbeam[0],2*pxbeam[1],fill=None) #create the box for the beam. The box start at the bottom and at twice the beam size from the right edge and is twice the beam large. So, the beam is enclosed in a box that extend a beam radius around the beam patch
            beam=patch.Ellipse((xlim[1]-pxbeam[0],ylim[0]+pxbeam[1]),pxbeam[0],pxbeam[1],bpa,hatch='/////',fill=None) #create the beam patch. The beam center is at a beamsize (in pixel) from the plot border
            ax.add_patch(box) #add the beam box
            ax.add_patch(beam) #add the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            kpcline=10/(asectokpc*pixelres) #length of the 10 kpc line in pixel
            kpcline=kpcline/(xlim[1]-xlim[0]) #lenght of the 10 kpc line in axes size
            leftmargin=leftmargin+0.05 #place the 10 kpc line and its label slightly more to the left and down to avoid being too much close to the axes
            topmargin=topmargin-0.01
            ax.hlines(topmargin-0.015,leftmargin,leftmargin+kpcline,color='black',linewidth=2,transform=ax.transAxes) #add the 10 kpc line
            ax.text(leftmargin+(kpcline/2),topmargin,'10 kpc',ha='center',transform=ax.transAxes)
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
        ctr_res=np.floor(0.85*np.nanmax(np.abs(mom1))/15)*5 #we want the contours to be multiple of 5
        ctr_pos=np.arange(0,0.85*np.nanmax(np.abs(mom1)),ctr_res).astype(int) #positive radial velocity contours
        ctr_neg=np.flip(-ctr_pos) #negative radial velocity contours
        #WE DO THE PLOT#
        im=ax.imshow(mom1,cmap='bwr',norm=norm,aspect='equal') #plot the moment 1 map with a colormap centered on 0
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if which == 'all': #if it is an atlas
            ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
        #WE ADD THE CONTOURS#
        ax.contour(mom1,levels=ctr_pos,colors='black',linewidths=ctr_width,linestyles='solid') #add the contours
        ax.contour(mom1,levels=ctr_neg,colors='black',linewidths=ctr_width,linestyles='dashed') #add the contours
        ax.contour(mom1,levels=[0],colors='gray',linewidths=2*ctr_width,linestyles='solid') #add the contour for the 0-velocity
        #WE ADD THE COLORBAR#
        if which == 'all': #if it is an atlas
            cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='Radial velocity [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        else:
            cb=plt.colorbar(im,ax=ax,location='top',pad=0.0,label='Radial velocity [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        cb.ax.tick_params(direction='in',length=5) #change the ticks of the corobar from external to internal and made them longer
        cb.set_ticks(np.concatenate((ctr_neg,ctr_pos),axis=None)) #set the ticks of the colobar to the levels of the contours
        #WE ADD ANCILLARY INFORMATION#
        ax.text(leftmargin,bottommargin,'Systemic velocity: v$_{{sys}}$ = {:.2f} km/s'.format(vsys),transform=ax.transAxes) #add the information of the systemic velocity
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            pxbeam=np.array([bmaj,bmin])/pixelres #beam in pixel
            box=patch.Rectangle((xlim[1]-2*pxbeam[0],ylim[0]),2*pxbeam[0],2*pxbeam[1],fill=None) #create the box for the beam. The box start at the bottom and at twice the beam size from the right edge and is twice the beam large. So, the beam is enclosed in a box that extend a beam radius around the beam patch
            beam=patch.Ellipse((xlim[1]-pxbeam[0],ylim[0]+pxbeam[1]),pxbeam[0],pxbeam[1],bpa,hatch='/////',fill=None) #create the beam patch. The beam center is at a beamsize (in pixel) from the plot border
            ax.add_patch(box) #add the beam box
            ax.add_patch(beam) #add the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            kpcline=10/(asectokpc*pixelres) #length of the 10 kpc line in pixel
            kpcline=kpcline/(xlim[1]-xlim[0]) #lenght of the 10 kpc line in axes size
            leftmargin=leftmargin+0.05 #place the 10 kpc line and its label slightly more to the left and down to avoid being too much close to the axes
            topmargin=topmargin-0.01
            ax.hlines(topmargin-0.015,leftmargin,leftmargin+kpcline,color='black',linewidth=2,transform=ax.transAxes) #add the 10 kpc line
            ax.text(leftmargin+(kpcline/2),topmargin,'10 kpc',ha='center',transform=ax.transAxes)
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
        ctr=np.power(2,np.arange(0,5,0.5))*disp #contours level in units of velocity dispersion
        #WE DO THE PLOT#
        im=ax.imshow(mom2,cmap='YlGn',aspect='equal') #plot the moment 2 map with a square-root colormap and in units of velocity dispersion
        ax.tick_params(direction='in') #change the ticks of the axes from external to internal
        ax.set_xlim(xlim) #set the xlim
        ax.set_ylim(ylim) #set the ylim
        ax.set_xlabel('RA') #set the x-axis label
        ax.set_ylabel('DEC') #set the y-axis label
        if which == 'all': #if all moment maps must be plotted
            ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
        #WE ADD THE CONTOURS#
        ax.contour(mom2,levels=ctr,cmap='RdPu_r',linewidths=ctr_width,linestyles='solid') #add the contours
        #WE ADD THE COLORBAR#
        if which == 'all': #if all moment maps must be plotted
            cb=fig.colorbar(im,ax=ax,location='top',pad=0.0,label='Velocity dispersion [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        else:
            cb=plt.colorbar(im,ax=ax,location='top',pad=0.0,label='Velocity dispersion [km/s]',fraction=0.0476) #add the colorbar on top of the plot
        cb.ax.tick_params(direction='in',length=5) #change the ticks of the colorbar from external to internal and made them longer
        cb.set_ticks(ctr) #set the ticks of the colobar to the levels of the contours
        #WE ADD ANCILLARY INFORMATION#
        ax.text(leftmargin,bottommargin,'Median dispersion: FWHM = {:.2f} km/s'.format(disp),transform=ax.transAxes) #add the information of the velocity dispersion
        if pixelres is not None and bmaj is not None and bmin is not None and bpa is not None: #if the pixel resolution and the beam is given
            pxbeam=np.array([bmaj,bmin])/pixelres #beam in pixel
            box=patch.Rectangle((xlim[1]-2*pxbeam[0],ylim[0]),2*pxbeam[0],2*pxbeam[1],fill=None) #create the box for the beam. The box start at the bottom and at twice the beam size from the right edge and is twice the beam large. So, the beam is enclosed in a box that extend a beam radius around the beam patch
            beam=patch.Ellipse((xlim[1]-pxbeam[0],ylim[0]+pxbeam[1]),pxbeam[0],pxbeam[1],bpa,hatch='/////',fill=None) #create the beam patch. The beam center is at a beamsize (in pixel) from the plot border
            ax.add_patch(box) #add the beam box
            ax.add_patch(beam) #add the beam
        if pixelres is not None and asectokpc is not None: #if the pixel resolution and the arcsec-to-kpc conversion is given
            kpcline=10/(asectokpc*pixelres) #length of the 10 kpc line in pixel
            kpcline=kpcline/(xlim[1]-xlim[0]) #lenght of the 10 kpc line in axes size
            leftmargin=leftmargin+0.05 #place the 10 kpc line and its label slightly more to the left and down to avoid being too much close to the axes
            topmargin=topmargin-0.01
            ax.hlines(topmargin-0.015,leftmargin,leftmargin+kpcline,color='black',linewidth=2,transform=ax.transAxes) #add the 10 kpc line
            ax.text(leftmargin+(kpcline/2),topmargin,'10 kpc',ha='center',transform=ax.transAxes)
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
    
#############################################################################################
def removemod(datacube='',modelcube='',maskcube='',method='subtraction',blankthreshold=0,removemodoutdir='',
              removemodoutname='',**kwargs):
    """Remove a model from a data cube using five methods: data-model (subtraction), data blanking (blanking), data-model after data blanking (b+s), data-model and negative residual blanking (negblank) and a combination of blanking and negblank (all).

    Args:
        datacube (string): name or path+name of the fits data cube
        modelcube (string): name or path+name of the fits model cube
        maskcube (string): name or path+name of the fits 3D mask to be used in the removal
        method (string): method to remove the model between:
            all, apply blanking and negblank methods
            blanking, blank the pixel in the data cube whose value in the model cube is > than blankthreshold
            b+s, same has above but it subtracts the model from the data AFTER the blanking
            negblank, subtract the model from the data and blank the negative residuals
            subtraction, subtract the model from the data
        blankthreshold (float): flux threshold for all, blanking and b+s methods
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
        ValueError: If no output folder is given
    """
    #CHECK THE INPUT#
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    if datacube == '' or datacube is None: #if a data cube is not given
        raise ValueError('ERROR: data cube is not set: aborting!')
    if modelcube == '' or modelcube is None: #if a model cube is not given
        raise ValueError('ERROR: model cube is not set: aborting!')
    if maskcube == '' or maskcube is None: #if a mask cube is not given
        if verbose:
            warnings.warn('No mask cube provided: the removal will be done over the whole data cube')
    elif maskcube[0] != '.': #if the mask cube name start with a . means that it is a path to the cube (so differs from path parameter)
        if 'path' in kwargs: #if the path to the mask cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the mask cube is set: aborting!')
            else:
                maskcube=kwargs['path']+maskcube
        else:
            raise ValueError('ERROR: no path to the mask cube is set: aborting!')    
    if datacube[0] != '.': #if the data cube name start with a . means that it is a path to the cube (so differs from path parameter)
        if 'path' in kwargs: #if the path to the data cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the data cube is set: aborting!')
            else:
                datacube=kwargs['path']+datacube
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')    
    if modelcube[0] != '.': #if the model cube name start with a . means that it is a path to the cube (so differs from path parameter)
        if 'path' in kwargs: #if the path to the model cube is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the model cube is set: aborting!')
            else:
                modelcube=kwargs['path']+modelcube
        else:
            raise ValueError('ERROR: no path to the data cube is set: aborting!')
    if method not in ['all','blanking','b+s','negblank','subtraction']: #if wrong operation is given
        raise ValueError("ERROR: wrong method. Accepted values: ['all','blanking','b+s','negblank','subtraction']. Aborting")   
    threshold=blankthreshold #store the blanking threshold from the input parameters
    outdir=removemodoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the output folder is not given
        if 'path' in kwargs: #if the path to the output folder is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no output folder is set: aborting!')
            else:
                outdir=kwargs['path']
        else:
            raise ValueError('ERROR: no output folder is set: aborting!') 
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder
    outname=removemodoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        outname=datacube.replace('.fits','_'+method+'.fits')
    elif outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname

    #---------------   START THE FUNCTION   ---------------#    
    with fits.open(datacube) as cube: #open the data cube
        data=cube[0].data.copy() #store the data
        with fits.open(modelcube) as modelcube: #open the model cube
            model=modelcube[0].data #store the model data
        if maskcube == '' or maskcube is None: #if a mask cube is not given
            mask=np.ones(data.shape)
        else:
            with fits.open(maskcube) as maskcube: #open the mask cube
                mask=maskcube[0].data #store the mask data
        if method in ['all','blanking','b+s']: #if all methods, or the blanking, or the blanking and subtraction must be performed
            data[np.where(model>threshold)]=np.nan #blank the data where the model is above the threshold
        if method in ['all','b+s','subtraction','negblank']: #if all methods, or the blanking and subtraction, or the subtraction or the negative blanking must be performed
            emission=np.where(mask>0) #store the emission coordinates
            data[emission]=data[emission]-model[emission] #subtract the model
        if method in ['all','negblank']: #if all methods, or the negative blanking must be performed
            mask[np.where(mask>0)]=1 #convert the mask into a 1/0 mask
            masked_data=data*mask #mask the data
            data[np.where(masked_data<0)]=np.nan #blank the negative data-model pixels

        cube[0].data=data #copy the new cube into the data cube
        cube.writeto(outname,overwrite=True) #write the fits cube
        
#############################################################################################
def rotcurve(mom1map='',pa=None,rotcenter=None,rotcurveoutdir='',rotcurveoutname='',save_csv=False,**kwargs):
    """Compute the rotation curve of a galaxy and plot it and (optionally) save it as a csv file.

    Args:
        mom1map (string): name or path+name of the fits moment 1
        pa (float): object position angle in degree
        rotcenter (list/array): x-y comma-separated coordinates of the rotational center in pixel
        rotcurveoutdir (string):output folder name
        rotcurveoutname (string):output file name
        save_csv (bool):save the rotation curve as csv file if True
        
    Kwargs:
        path (string): path to the moment map if the momXmap is a name and not a path+name
        vsys (float): object systemic velocity in km/s
        pixelres (float): pixel resolution of the data in arcsec
        asectokpc (float): arcsec to kpc conversion to plot the spatial scale
        objname (string): name of the object
        verbose (bool): option to print messages and plot to terminal if True  

    Returns:
        Plot and(optional) csv file of the galactic rotation curve
        
    Raises:
        ValueError: If no velocity field is provided
        ValueError: If no path is provided
        ValueError: If no galactic center is set
        ValueError: If galactic center is  not given as x-y comma-separated coordinates in pixel
        ValueError: If no output folder is provided
    """    
    #CHECK THE INPUT#
    if 'verbose' in kwargs: #check if the verbose option is in kwargs
        verbose=kwargs['verbose'] #store the verbose option from the input kwargs
    else:
        verbose=False #set it to False
    if mom1map == '' or mom1map is None: #if a moment 1 map is not given
        raise ValueError('ERROR: velocity field is not set: aborting!')
    if mom1map[0] != '.': #if the moment 1 map name start with a . means that it is a path to the map (so differs from path parameter)
        if 'path' in kwargs: #if the path to the  moment 1 map is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no path to the velocity field is set: aborting!')
            else:
                mom1map=kwargs['path']+mom1map
        else:
            raise ValueError('ERROR: no path to the velocity field is set: aborting!')
    if pa is None: #if no position angle is given
        raise ValueError('ERROR: position angle is not set: aborting!')
    pa=np.radians(pa-90) #convert the position angle into effective angle in radians
    center=rotcenter #store the rotation center from the input parameters
    if center is None: #if no center is given
        raise ValueError('ERROR: no velocity field center is provided: aborting!')
    elif len(center) != 2: #if the center as wrong length
        raise ValueError('ERROR: wrongly velocity field center is provided. Use x-y comma-separated coordinates in pixel: aborting!')
    outdir=rotcurveoutdir #store the output folder from the input parameters
    if outdir == '' or outdir is None:  #if the output folder is not given
        if 'path' in kwargs: #if the path to the output folder is in kwargs
            if kwargs['path'] == '' or kwargs['path'] is None:
                raise ValueError('ERROR: no output folder is set: aborting!')
            else:
                outdir=kwargs['path']
        else:
            raise ValueError('ERROR: no output folder is set: aborting!') 
    elif not os.path.exists(outdir): #if the output folder does not exist
        os.makedirs(outdir) #create the folder
    outname=rotcurveoutname #store the outputname from the input parameters
    if outname == '' or outname is None: #if the outname is empty
        outname=mom1map.replace('.fits','_rotcurve.pdf')  #the outname is the object name plus rotvurve.pdf
    elif outname[0] != '.': #if the outname name start with a . means that it is a path to the cube (so differs from path parameter)
        outname=outdir+outname
    #CHECK THE KWARGS#
    if 'vsys' in kwargs: #if the systemic velocity is in the input kwargs
        vsys=kwargs['vsys'] #store the systemic velocity from the input kwargs
    else:
        vsys=None #set it to None
    if 'pixelres' in kwargs: #if the pixel resolution is in the input kwargs
        pixelres=kwargs['pixelres'] #store the pixel resolution from the input kwargs
    else:
        pixelres=None #set it to None
    if 'asectokpc' in kwargs: #if the arcsec-to-kpc conversion is in the input kwargs
        asectokpc=kwargs['asectokpc'] #store the arcsec-to-kpc conversion from the input kwargs
    else:
        asectokpc=None #set it to None
    if 'objname' in kwargs: #if the object name is in the input kwargs
        objname=kwargs['objname'] #store the object name from the input kwargs
        if objname is None: #if in kwargs but not set
            objname='' #set it to empty
    else:
        objname='' #set it to empty
           
    #---------------   START THE FUNCTION   ---------------#
    with fits.open(mom1map) as m1: #open the moment 1 map
        mom1=m1[0].data #store the data
        header=m1[0].header #store the header
        if header['BUNIT']=='m/s': #if the units are in m/s
            mom1=mom1/1000 #convert to km/s
            
    if vsys is None: #if the systemic velocity is not given
        vsys=np.nanmedian(mom1) #calculate the systemic velocity
    mom1=mom1-vsys #subract the result to the moment 1 map
    
    if pixelres is None: #if the pixelres is not given
        if 'CUNIT1' in header: #if the spatial unit is in the header
            spaceunits=header['CUNIT1'] #store the spatial unit
            if 'CDELT1' in header: #if the spatial resolution is in the header
                spaceres=abs(header['CDELT1']) #store the spatial resolution from the header
                if spaceunits == 'deg': #if the spatial unit is degree
                    pixelres=spaceres*3600 #convert into arcsec
                elif spaceunits == 'arcmin': #if the spatial unit is arcmin
                    pixelres=spaceres*60 #convert into arcsec
                elif spaceunits == 'arcsec': #if the spatial unit is arcsec
                    pixelres=spaceres #do nothing
        elif verbose:
            warnings.warn('No spatial unit was found: unable to calculate the pixel resolution!')
    
    q=center[1]-center[0]*np.tan(pa) #calculate the intercept of the y-axis
    r=np.sqrt(mom1.shape[0]**2+(mom1.shape[0]*np.tan(pa))**2) #calculate the radius of the galaxy
    x=np.arange(round(r))*np.cos(pa) #define the x-axis
    y=np.arange(round(r))*np.sin(pa)+q #define the y-axis
    if pixelres is None: #if no pixel resolution is given
        radius=(x-center[0]) #radius from center in pixel
        units='pixel' #set the units to pixel
    elif asectokpc is None: #if no arcsec-to-kpc conversion is provided
        radius=(x-center[0])*pixelres  #radius from center in arcsec
        units='arcsec' #set the units to arcsec
    else:
        radius=(x-center[0])*pixelres*asectokpc #convert the radius from pixel to kpc
        units='kpc' #set the units to kpc
    rotcurve=mom1[x.astype(int),y.astype(int)] #extract the rotation curve
    rec=rotcurve[rotcurve<=0] #store the receiding rotation curve
    app=rotcurve[rotcurve>=0] #store the approaching rotation curve
    
    nrows=1 #number of rows in the atlas
    ncols=1 #number of columns in the atlas
    
    fig=plt.figure(figsize=(6*ncols,7*nrows)) #create the figure
    fig.suptitle('{} rotation curve'.format(objname),fontsize=24) #add the title
    ax=fig.add_subplot(nrows,ncols,1) #create the subplot
    ax.plot(abs(radius[rotcurve<=0]),abs(rec),color='red',label='Receiding side',linewidth=1.5) #plot the receding rotation curve
    ax.plot(abs(radius[rotcurve>=0]),app,color='blue',label='Approaching side',linewidth=1.5) #plot the approaching rotation curve
    ax.set_xlabel('Radius from center [{}]'.format(units)) #set the x-axis label
    ax.set_ylabel('Velocity [km/s]') #set the y-axis label
    ax.legend(loc='lower right') #set the legend
    
    fig.subplots_adjust(left=0.17, bottom=0.1, right=0.93, top=0.85, wspace=0.0, hspace=0.0) #fix the position of the subplots in the figure   
    fig.savefig(outname,dpi=300,bbox_inches='tight') #save the figure
    
    if verbose: #if print-to-terminal option is true
        plt.show() #show the figure
    else:
        plt.close()
    
    if save_csv: #if the result must be saved to a csv file
        df=pd.DataFrame(columns=['RADIUS [kpc]','VROT [km/s]']) #create the dataframe
        df['RADIUS [kpc]']=radius #store the radius
        df['VROT [km/s]']=rotcurve #store the rotation curve
        df.dropna(subset=['VROT [km/s]'],inplace=True) #remove the NaNs
        df.to_csv(outname.replace('.pdf','.csv'),index=False) #convert into a csv
                        
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
        spectralres=spectralres/1000 #convert the spectral resolution in km/s
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
        print('For H₀={:.1f}, \u03A9ₘ={:.2f}, \u03A9ᵥ={:.2f}, z={}:'.format(H0,omega_matter,omega_vacuum,z))
        print('It is now {:.1f} Gyr since the Big Bang.'.format(age_Gyr))
        print('The age at redshift z was {:.1f} Gyr.'.format(zage_Gyr))
        print('The light travel time was {:.3f} Gyr.'.format(DTT_Gyr))
        print('The comoving radial distance, which goes into Hubbles law, is {:.1f} Mpc or {:.3f} Gly'.format(DCMR_Mpc,DCMR_Gyr))
        print('The comoving volume within redshift z is {:.3f} Gpc^3.'.format(V_Gpc))
        print('The angular size distance Dₐ is {:.1f} Mpc or {:.3f} Gly.'.format(DA_Mpc,DA_Gyr))
        print('This gives a scale of {:.3f} kpc/arcsec.'.format(kpc_DA))
        print('The luminosity distance Dₗ is {:.1f} Mpc or {:.3f} Gly.'.format(DL_Mpc,DL_Gyr))
        print('The distance modulus, m-M, is {:.2f}'.format(5*np.log10(DL_Mpc*10.**6)-5))

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
        box (list/array): region to compute the flux as [xmin,xmax,ymin,ymax]

    Kwargs:
        beamcube (string): name or path+name of the fits beam cube if pbcorr is True
        path (string): path to the beam cube if the beamcube is a name and not a path+name
        verbose (bool): option to print messages to terminal if True   
        
    Returns:
        HI mass over the given array with the uncertainty as float
        
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
        raise ValueError('ERROR: Wrong units provided. Accepted values: m/s, km/s. Aborting!')
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
    if box is not None: #if no region is provided        
        if len(box)!=4: #check that it has the correct format
            raise ValueError('ERROR: Please provide the box in the format [xmin,xmax,ymin,ymax]. Aborting!')
        else: #store the spatial box from the input
            xmin=box[0]
            xmax=box[1]
            ymin=box[2]
            ymax=box[3]
            if xmin < 0: #if xmin is negative
                warnings.warn('Lower x limit is negative ({}): set to 0.'.format(xmin))
                xmin=0 #set it to 0
            if xmax > data.shape[2]: #if xmax is too high
                warnings.warn('Max x is too high ({}): set to the size of x.'.format(xmax))
                xmax=data.shape[2] #set it to size of data
            if ymin < 0: #if ymin is negative
                warnings.warn('Lower y limit is negative ({}): set to 0.'.format(ymin))
                ymin=0 #set it to 0
            if ymax > data.shape[1]: #if ymax is too high
                warnings.warn('Max y is too high ({}): set to the size of y.'.format(ymax))
                ymax=data.shape[1] #set it to size of data
    #NOW WE CALCULATE THE MASS#  
        HI_mass=(2.35*10**5)*(distance**2)*pixelsize*np.nansum(darray[xmin:xmax,ymin:ymax])/beamarea #compute the HI masss
    else:
        HI_mass=(2.35*10**5)*(distance**2)*pixelsize*np.nansum(darray)/beamarea #compute the HI mass
    error=HI_mass/10 #the error on the mass is equal to the calibration error (typically of 10%)
    if verbose: #if the print-to-terminal option is True
        print('Total HI mass: {:.2e} solar masses'.format(HI_mass))
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
        box (list/array): region to compute the flux as [xmin,xmax,ymin,ymax]
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
        spectralres=spectralres/1000 #convert the spectral resolution in km/s
    #------------   BOX     ------------         
    if box is not None: #if no region is provided        
        if len(box)!=4: #check that it has the correct format
            raise ValueError('ERROR: Please provide the box in the format [xmin,xmax,ymin,ymax]. Aborting!')
        else: #store the spatial box from the input
            xmin=box[0]
            xmax=box[1]
            ymin=box[2]
            ymax=box[3]
            if xmin < 0: #if xmin is negative
                warnings.warn('Lower x limit is negative ({}): set to 0.'.format(xmin))
                xmin=0 #set it to 0
            if xmax > data.shape[2]: #if xmax is too high
                warnings.warn('Max x is too high ({}): set to the size of x.'.format(xmax))
                xmax=data.shape[2] #set it to size of data
            if ymin < 0: #if ymin is negative
                warnings.warn('Lower y limit is negative ({}): set to 0.'.format(ymin))
                ymin=0 #set it to 0
            if ymax > data.shape[1]: #if ymax is too high
                warnings.warn('Max y is too high ({}): set to the size of y.'.format(ymax))
                ymax=data.shape[1] #set it to size of data
    #NOW WE CALCULATE THE FLUX# 
        flux=np.nansum(darray[xmin:xmax,ymin:ymax])*spectralres*pixelsize/beamarea #calculate the flux
    else:
        flux=np.nansum(darray)*spectralres*pixelsize/beamarea #calculate the flux
    if verbose: #if the print-to-terminal option is True
        print('The flux is {:.2e} {}'.format(flux,units))
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
            'HO		=	   #Hubble parameter (default: 69.6)\n'
            'Omegam	=	   #Omega matter (default: 0.286)\n'
            'Omegav	=	   #Omega vacuum (default: 0.714)\n'
            '\n'
            '[GALAXY]\n'
            'objname		=	   #name of the object (default: None)\n'
            'distance	=	   #distance of the object in Mpc (default: None)\n'
            'redshift	=	   #redshift of the object (default: None)\n'
            'asectokpc	=	   #arcsec to kpc conversion (default: None)\n'
            'vsys		=	   #systemic velocity of the object in km/s (default: None)\n'
            'pa			=	   #position angle of the object in deg (default: None)\n'
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
            'inputdir	=	   #input directory. If empty, is the same of [INPUT] path (default: None)\n'
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
            'inputdir 	=	   #input directory. If empty, is the same of [INPUT] path (default: None)\n'
            'outputdir	=	   #output directory to save the new mask. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the new mask including extension .fits. If empty, is the same of maskcube (default: None)\n'
            '\n'
            '[GAUSSFIT]\n'
            'datacube	=	   #name of the fits file of the data cube to fit including .fits. If empty, is the same of [FITS] datacube (default: None)\n'
            'gaussmask	=	   #name of the fits file of the mask cube including .fits. If empty, is the same of [FITS] maskcube. The fit will be done inside the mask (default: None)\n'
            'inputdir	=	   #input directory. If empty, is the same of [INPUT] path (default: None)\n'
            'linefwhm	=	   #first guess on the fwhm of the line profile in km/s (default: 15)\n'
            'amp_thresh	=	   #amplitude threshold for the fit. If a profile peak is < threshold, the fit wont be performed on that spectrum (default: 0)\n'
            'p_reject	=	   #p-value threshold for fit rejection. If a best-fit as p>p_reject, it will be rejected (default: 1) \n'
            'clipping	=	   #clip the spectrum to a % of the profile peak [True,False] (default: False)\n'
            'threshold	=	   #clip threshold as % of the peak (0.5 is 50%) if clipping is True (default: 0.5)\n'
            'errors		=	   #compute the errors on the best-fit [True,False] (default: False)\n'
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
            'outputdir	=	   #output directory to save the plot. If empty, is the same as [INPUT] path (default: None)\n'
            'outname		=	   #output name of the plot (including file extension *.jpg,*.png,*.pdf,...) (default: None)\n'
            '\n'
            '[PLOTSTYLE]\n'
            'ctr width	=	   #width of the contours levels (default: 2)')

#############################################################################################
############################### Code template for atlas plot ################################   
# 
# nrows= #number of rows in the atlas
# ncols= #number of columns in the atlas
# outname='<Insert outname here>' #name of the outfile
# 
# fig=plt.figure(figsize=(6*ncols,7*nrows))
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
# fig.savefig(outname,dpi=300) #save the figure
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
        
#############################################################################################
#function: nicci
#input: parameters = keyworded arguments with the necessary variables passed thourgh the parameter dictionary
#description: MAIN FUNCTION. Calls the importparameters function to read the parameter file and execute a number of task based on the result of the reading
#output: none

def nicci(parameters):
    
    mode=parameters['mode'].split(',')
    for i in range(len(mode)):
        if mode[i] == 'cutcube':
            cutcube(parameters)
            
        elif mode[i] == 'cubestat':
            stat=cubestat(parameters)
            
            parameters['bmaj']=stat['beam major axis']
            parameters['bmin']=stat['beam minor axis']
            parameters['spectralres']=stat['spectral resolution']
            parameters['rms']=stat['rms']
            
            return parameters
            
        elif mode[i] == 'fixmask':
            fixmask(parameters)
            
        elif mode[i] == 'fixshufflemask':
            fixshufflemask(parameters)
            
        elif mode[i] == 'gaussfit':
            gaussfit(parameters)
            
        elif mode[i] == 'plotmom':
            plotmom(parameters)
            
        elif mode[i] == 'removedisc':
            removedisc(parameters)
"""