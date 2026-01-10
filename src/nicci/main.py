"""
A collection of functions for the analysis of astronomical data cubes and images.

                NICCI
Naive functIons Collection for Cubes and Images
"""
#############################################################################################
#
# FILENAME: main
# VERSION: 1.0
# DATE: 10/01/2026
# CHANGELOG:
#   - major revision of the entire package
#       - the following functions were removed:
#           - cosmo;
#           - create_config;
#           - cubedo;
#           - cubestat;
#           - fitsarith;
#           - fixmask;
#           - fit_muse;
#           - flux;
#           - gaussfit;
#           - import_parameters;
#           - radial_profile;
#           - removemod;
#           - rotcurve;
#           - stacking.
#
#       - the following functions were streamlined:
#           - chanmap;
#           - convert_to_HI;
#           - getpv. This function is now also split into pvslice and plot_pvslice;
#           - lines_finder;
#           - noise_variations. This function is now called plot_noise;
#           - plotmom. This function is now also split into plot_mom0, plot_mom1 and plot_mom2 and the helper _plot_mom;
#           - RMS;
#           - velfi.
#
#       - the following functions were added:
#           - convert_from_HI. Convert HI data values using a beam area and velocity integration.
#           - fits_cutout. Create a spatial (and optionally spectral) cutout from a FITS file.
#           - load_data. Load astronomical data from a FITS file, a NumPy array, or an (array, header) tuple
#                        and return a standardized dictionary with data, metadata, WCS information, and derived quantities.
#           - make_figure. Create a Matplotlib figure and axes with flexible layout options,
#                          including support for WCS projections, multiple panels, shared formatting, and aliases for common parameters.
#           - shuffle. Create a Matplotlib figure and axes with flexible layout options, including
#                      support for WCS projections, multiple panels, shared formatting, and aliases for common parameters.
#           - stack. Stack a set of 1D spectra using optional noise-based weighting.
#
#       - the following helpers were removed:
#           - _assign_to_concentric;
#           - _assign_to_cones;
#           - _load;
#           - _load_box;
#           - _parallel_gas_fit;
#           - _parallel_gauss_fit;
#           - _parallel_stellar_fit;
#           - _plot_bpt;
#           - _plot_stack_result;
#           - _read_string.
#
#       - the following helpers were streamlined:
#           - _assign_to_cells. It is now a public function.
#           - _assign_to_rings. It is now a public function.
#           - _auto_fit_fontsize;
#           - _covariance_to_error_ellipse;
#           - _normalize;
#           - _plot_beam;
#           - _plot_kpcline (now called _plot_kpc_scale);
#           - _plot_stack_spectrum (now called _plot_spectrum);
#           - _radius;
#           - _reliability;
#           - _source_finder;
#           - _source_linker;
#           - _stack. It is now a public function.
#
#       - the following helpers were added:
#           - _add_colorbar. Add a horizontal colorbar above an axes with optional contour annotations;
#           - _contours. Compute contour levels from a specification;
#           - _extend. Extend an array-like variable to a target length by repetition;
#           - _extract_reliability_params. Extracts logarithmic peak, summed, and mean flux parameters for sources;
#           - _kde_density. Evaluates a multivariate Gaussian kernel density estimate in blocks;
#           - _norm. Construct a normalization object for image display.;
#           - _plot_mom. It is the former plot_mom;
#           - _plot_reliability. Produces diagnostic plots for reliability parameter space;
#           - _plot_skellam. Plots the cumulative distribution of the Skellam parameter;
#           - _split_sources. Splits detected sources into positive and negative samples;
#           - _write_catalogue. Builds a source catalogue from detected spectral features.
#
#############################################################################################

from astropy import units as u
from astropy.convolution import convolve,Box1DKernel
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.transforms import Bbox
from pvextractor import Path
from pvextractor import extract_pv_slice
from scipy.special import erf
from scipy.stats import anderson
from tqdm.auto import tqdm

import matplotlib.colors as cl
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

def _full_warning(message,category,filename,lineno,file=None,line=None):
    return 'WARNING: %s:%s: %s: %s\n' %(filename,lineno,category.__name__,message)
def _simple_warning(message,category,filename,lineno,file=None,line=None):
    return 'WARNING: %s: %s\n' %(category.__name__,message)
    
warnings.formatwarning=_simple_warning

plt.rc('font',size=14) #set the default fontsize for the text in plots
plt.rcParams['mathtext.default']='regular'
plt.rcParams['xtick.direction']='in'
plt.rcParams['xtick.major.size']=10
plt.rcParams['xtick.major.width']=1
plt.rcParams['xtick.minor.size']=5
plt.rcParams['xtick.minor.width']=1
plt.rcParams['xtick.top']=True
plt.rcParams['ytick.direction']='in'
plt.rcParams['ytick.major.size']=10
plt.rcParams['ytick.major.width']=1
plt.rcParams['ytick.minor.size']=5
plt.rcParams['ytick.minor.width']=1
plt.rcParams['ytick.right']=True
plt.rcParams['axes.formatter.use_mathtext']=True

## --- CUSTOM COLORMAPS --- ##
if 'mom1' not in plt.colormaps(): #register the colormap for the moment 1 maps if not existing
    plt.colormaps.register(cmap=cl.LinearSegmentedColormap.from_list('mom1',[(95/255,235/255,255/255),(0/255,0/255,255/255),(0/255,0/255,125/255),(255/255,255/255,255/255),(125/255,0/255,0/255),(255/255,0/255,0/255),(255/255,95/255,135/255)]))
    
if 'mom1_alt' not in plt.colormaps(): #register the colormap for the moment 1 maps if not existing
    plt.colormaps.register(cmap=cl.LinearSegmentedColormap.from_list('mom1_alt',[(95/255,235/255,255/255),(0/255,0/255,255/255),(0/255,0/255,125/255),(0/255,255/255,0/255),(125/255,0/255,0/255),(255/255,0/255,0/255),(255/255,95/255,135/255)]))

if 'mom1_cosmos' not in plt.colormaps(): #register the colormap for the moment 1 maps if not existing
    from CosmosCanvas import velmap as vmap

    #set the saturation using Chroma values
    Cval_max=50.
    Cval_mid=0.
    Cval_4=40.
    Cval_min=35.

    #select Hues, using degrees on the colour wheel
    Hval_L=190. #left for lowest data value.  Turquoise.
    Hval_R=10.  #right for highest data value. Rose.
    Hval_1=210. #cyans
    Hval_2=230.
    Hval_3=40.  #while hue 50 is the complement to 230 it looks too brown so the colour selected is offset.
    Hval_4=30.

    #adjust luminosity
    Lval_1=61. #this is the default value in velmap.py file
    Lval_2=55.
    Lval_mid=70.
    Lval_3=55
    Lval_4=35
    Lval_min=25.

    #adjust range for minor axis segment. (Default=0.05; Recommended max = 0.15)\n",
    width=0.05  #this is the segment centred on div that will be grey (chroma = 0)\n",
    
    #create the cmap
    vmap.create_cmap_chromaVelocity(-50,50,name='mom1_cosmos',div=0,width=width,
                                    Cval_max=Cval_max,Cval_mid=Cval_mid,Cval_min=Cval_min,
                                    Hval_L=Hval_L,Hval_R=Hval_R,
                                    Hval_1=Hval_1,Hval_2=Hval_2,Hval_3=Hval_3,Hval_4=Hval_4,
                                    Lval_mid=Lval_mid,
                                    Lval_1=Lval_1,Lval_2=Lval_2,Lval_3=Lval_3,
                                    Lval_4=Lval_4,Lval_min=Lval_min,Cval_4=Cval_4)
    
if 'mom2' not in plt.colormaps(): #register the colormap for the moment 2 maps if not existing
    plt.colormaps.register(cmap=cl.LinearSegmentedColormap.from_list('mom2',[(1,1,1),(0/255,255/255,255/255,0.3),(255/255,255/255,0/255),(255/255,127/255,0/255),(200/255,0/255,0/255)])) 

if 'mom2_ctr' not in plt.colormaps(): #register the colormap for the moment 2 contours if not existing
    plt.colormaps.register(cmap=cl.LinearSegmentedColormap.from_list('mom2_ctr',[(0/255,155/255,255/255,0.75),(150/255,0/255,0/255)]))

#######################################################################################################

def assign_to_cells(data,
                    *,
                    nregions):
    """
    Partition a 2D array into a specified number of approximately equal, rectangular regions
    and return the pixel coordinates belonging to each region.

    The function constructs a near-square grid whose total number of cells is at least
    ``nregions``. Each cell has a uniform size determined by integer division of the input
    array dimensions. The grid is centered within the array, and only the first
    ``nregions`` cells (in row-major order) are used. For each selected cell, all pixel
    coordinates inside that cell are collected.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D array representing the input data from which regions are defined.
    nregions : int, keyword-only
        The number of regions (cells) to generate. Must be a positive integer.

    Returns
    -------
    xvalid : numpy.ndarray
        A 2D array of shape (nregions, N), where each row contains the flattened x-coordinates
        of pixels belonging to one region.
    yvalid : numpy.ndarray
        A 2D array of shape (nregions, N), where each row contains the flattened y-coordinates
        of pixels belonging to one region.

    Raises
    ------
    ValueError
        If ``nregions`` is not a positive integer.

    Notes
    -----
    - Regions are rectangular and all have the same size.
    - If the array dimensions are not exactly divisible by the grid dimensions,
    some border pixels may be excluded.
    - Regions are generated in row-major order (top to bottom, left to right).
    """

    #check that input number of regions is valid
    if nregions <= 0:
        raise ValueError("nregions must be a positive integer")

    #determine data shape
    height,width=data.shape

    #choose a near-square grid such that nrows * ncols >= nregions
    nrows=int(np.floor(np.sqrt(nregions))) #floor ensures that the number of rows is the largest integer not exceeding sqrt(nregions)
    ncols=int(np.ceil(nregions/nrows)) #ceil gives the maximum number of columns to cover the entire data array 

    #compute uniform cell size (integer division)
    cell_h=height//nrows
    cell_w=width//ncols

    #determine the portion of the array covered by the grid
    used_h=cell_h*nrows
    used_w=cell_w*ncols

    #center the grid within the array
    y0=(height-used_h)//2
    x0=(width-used_w)//2

    #initialize lists of valid pixel coordinates per region
    xvalid=[]
    yvalid=[]

    #track how many regions have been generated
    region_count=0

    #iterate over the grid cells
    for row in range(nrows):
        for col in range(ncols):
            #stop once the requested number of regions is reached
            if region_count >= nregions:
                break

            #compute cell bounds in array coordinates
            ys=y0+row*cell_h
            ye=ys+cell_h
            xs=x0+col*cell_w
            xe=xs+cell_w

            #generate all pixel coordinates inside the cell
            y,x=np.meshgrid(np.arange(ys,ye),
                            np.arange(xs,xe),
                            indexing="ij")

            #store flattened coordinates for the current region
            xvalid.append(x.ravel())
            yvalid.append(y.ravel())

            #increment region counter
            region_count+=1

    return np.array(xvalid),np.array(yvalid)
    
##########################################################################################################

def assign_to_rings(data,
                    *,
                    radii,pa,inc,xpos,ypos,
                    between_angles=None):
    """
    Assign pixels of a 2D array to a set of concentric (possibly elliptical) rings.

    This function determines, for each ring defined by an outer radius and
    associated geometric parameters, which pixels of the input 2D `data` array
    belong to that ring. Rings can be elliptical due to inclination and position
    angle, can have ring-dependent centers, and can optionally be restricted to
    specific angular sectors.

    Pixels are considered part of a ring if they lie between the inner and outer
    elliptical boundaries of that ring. The inner boundary of ring *i* is given by
    the outer boundary of ring *i-1* (or zero for the first ring).

    Parameters
    ----------
    data : 2D array-like
        Input data array. Only its shape is used to generate pixel coordinates;
        the data values themselves are not modified or inspected.

    radii : array-like of float
        Outer radii of the rings, in pixel units. The length of this array defines
        the number of rings. Radii are assumed to be ordered from inner to outer.

    pa : float or array-like of float
        Position angle(s) of the rings, in degrees. If a single value is provided,
        it is applied to all rings; otherwise, it must have one value per ring.

    inc : float or array-like of float
        Inclination angle(s) of the rings, in degrees. If a single value is provided,
        it is applied to all rings; otherwise, it must have one value per ring.

    xpos : float or array-like of float
        X-coordinate(s) of the ring centers, in pixel units. If a single value is
        provided, it is applied to all rings; otherwise, it must have one value per
        ring.

    ypos : float or array-like of float
        Y-coordinate(s) of the ring centers, in pixel units. If a single value is
        provided, it is applied to all rings; otherwise, it must have one value per
        ring.

    between_angles : array-like of shape (nrings, 2), optional
        Angular boundaries for each ring, in degrees, defining a sector
        [min_angle, max_angle] in the plane of the sky. Angles are converted to the
        position-angle reference frame of each ring and properly wrapped. If
        provided, only pixels whose polar angle falls within the specified range
        are considered part of the ring. If None, the full azimuthal extent is used.

    Returns
    -------
    xvalid : list of list of int
        For each ring, a list containing the x-indices (column indices) of pixels
        that belong to that ring.

    yvalid : list of list of int
        For each ring, a list containing the y-indices (row indices) of pixels
        that belong to that ring.

    Notes
    -----
    - Radii, positions, and pixel coordinates are all assumed to be in pixel units.
    - Angles (`pa`, `inc`, `between_angles`) are specified in degrees.
    - Rings may vary in geometry (radius, position angle, inclination, center)
    from one ring to another.
    - The returned pixel coordinates can be used to index the original `data`
    array as `data[y][x]`.
    """

    #ensure radii is an array
    radii=np.asarray(radii)

    #get the number of rings
    nrings=len(radii)
    
    #extend the relevant quantities
    pa=_extend(pa,nrings)
    inc=_extend(inc,nrings)
    xpos=_extend(xpos,nrings)
    ypos=_extend(ypos,nrings)

    #manage the angle boundaries
    if between_angles is not None:
        #convert angle boundaries to "position angle" frame.
        #+90 is needed to match python array orientation with position angle on sky convention
        angles=np.asarray(between_angles)+pa[:,None]+90

        #convert to radians
        angles=np.radians(angles)

        #normalise angle in the [-np.pi,np.pi) range
        angles=(angles+np.pi) % (2*np.pi)-np.pi

    #convert to radians
    pa=np.radians(pa)
    inc=np.radians(inc)

    #determine the data shape
    xshape,yshape=data.shape

    #generate the pixel coordinates
    xx,yy=np.meshgrid(np.arange(yshape),
                      np.arange(xshape),
                      indexing="ij")

    #initialize the list of valid pixel coordinates per ring
    xvalid=[[] for _ in range(nrings)]
    yvalid=[[] for _ in range(nrings)]

    #for each ring
    for ring in range(nrings):
        #determine the inner boundary radius, position angle and inclination
        r_min=0.0 if ring == 0 else radii[ring-1]
        pa_min=pa[ring-1] if ring > 0 else pa[ring]
        inc_min=inc[ring-1] if ring > 0 else inc[ring]

        #determine the outer boundary radius, position angle and inclination
        r_max=radii[ring]
        pa_max=pa[ring]
        inc_max=inc[ring]

        #compute the pixel positions w.r.t. the ring centers
        dx=xx-xpos[ring]
        dy=yy-ypos[ring]

        #if angle boundaries are provided
        if between_angles is not None:
            #compute the point angle
            theta=np.arctan2(dy, dx)

            #get the angle boundaries
            min_angle,max_angle=angles[ring]

            #mask the pixels outside the angle range
            mask=((max_angle-min_angle) % (2*np.pi) >= (theta-min_angle) % (2*np.pi))
        else:
            #all pixels are valid
            mask=np.ones_like(dx,dtype=bool)

        #calculate the inner elliptical radius of the pixel
        d_min=_radius(r_min,dx,dy,pa_min,inc_min)

        #calculate the outer elliptical radius of the pixel
        d_max=_radius(r_max,dx,dy,pa_max,inc_max)

        #if one of the distance is negative, means that the points resides between the innermost and outermost radius, hence, it is a valid pixel.
        #Indeed, if a pixel is inside rmin, then d1<0 but also d2<0 and so d1*d2>0.
        #If the pixel is outside rmax, then d1>0 but also d2>0 and so d1*d2>0.
        mask &= (np.round(d_min)*np.round(d_max)) <= 0

        #extract the pixels in the ring
        ys,xs=np.nonzero(mask)

        #store the pixel coordinates in the ring
        xvalid[ring]=xs.tolist()
        yvalid[ring]=ys.tolist()

    return xvalid,yvalid

#######################################################################################################
        
def chanmap(datacube,channel_step,
            *,
            first_channel=None,last_channel=None,n_channels=9,
            xmin=None, xmax=None,ymin=None,ymax=None,
            pixunits="deg",fluxunits='Jy/beam',specunits="m/s",pixelres=None,spectralres=None,
            z0=None,vsys=None,
            bmaj=None,bmin=None,bpa=None,
            use_column_density=True,
            fig_x_size=5,fig_y_size=5,
            wspace=0.01,hspace=0.01,
            dpi=300,
            wcs=None,
            cmap="Greys",
            vmin=0.001,vmax=0.9,norm="log",
            ctr="pow3",ctr_cmap="hot",ctr_width=1,
            left_loc=0.05,right_loc=0.95,lower_loc=0.05,upper_loc=0.9,
            kpc=10,asectokpc=None,
            color="black",y_offset=0.02,colorbar_label=None,
            output=None):
    """
    Create a channel map figure from a spectral-line data cube.

    This function extracts a subset of velocity channels from a 3D data cube
    (spectral, y, x) and displays them as a grid of 2D images. Each panel
    corresponds to one channel (or a stepped selection of channels) and can
    include intensity maps, contours, beam representation, velocity labels,
    a physical scale bar, and a shared colorbar. The function is primarily
    intended for astronomical spectral-line data (e.g., HI or CO cubes).

    Parameters
    ----------
    datacube : str or array-like
        Input data cube or a reference that can be understood by `load_data`.
        The cube is expected to have shape (nchan, ny, nx).
    channel_step : int
        Step size between displayed channels (e.g., 1 for consecutive channels,
        2 to skip every other channel).

    first_channel : int, optional
        Index of the first channel to display. If not provided, the range is
        determined from the systemic velocity or the cube center.
    last_channel : int, optional
        Index of the last channel to display. If not provided, the range is
        determined from the systemic velocity or the cube center.
    n_channels : int, optional
        Number of channels to display. Used only if `first_channel` or
        `last_channel` is not explicitly set.

    xmin, xmax, ymin, ymax : int, optional
        Spatial pixel bounds defining the sub-region to extract from the cube.
        Defaults to the full spatial extent.

    pixunits : {"deg", "arcsec"}, optional
        Units of the spatial pixel scale.
    fluxunits : str, optional
        Units of the flux density (e.g., "Jy/beam").
    specunits : {"m/s", "km/s"}, optional
        Units of the spectral axis.
    pixelres : float, optional
        Spatial pixel size in `pixunits`.
    spectralres : float, optional
        Spectral resolution per channel in `specunits`.

    z0 : float, optional
        Reference velocity (e.g., velocity of the first channel).
    vsys : float, optional
        Systemic velocity used to label channel velocities.

    bmaj, bmin : float, optional
        Major and minor axes of the synthesized beam in `pixunits`.
    bpa : float, optional
        Beam position angle in degrees.

    use_column_density : bool, optional
        If True, contours are interpreted and labeled as column density when
        sufficient beam and spectral information is available.

    fig_x_size, fig_y_size : float, optional
        Size of the figure in inches.
    wspace, hspace : float, optional
        Width and height spacing between subplots.
    dpi : int, optional
        Resolution of the output figure in dots per inch.

    wcs : astropy.wcs.WCS, optional
        World Coordinate System for the spatial axes.
    cmap : str or Colormap, optional
        Colormap used for the image panels.
    vmin, vmax : float, optional
        Minimum and maximum values for image normalization.
    norm : str, optional
        Normalization scheme (e.g., "log", "linear").

    ctr : str, list, or array-like, optional
        Contour definition. Can be a keyword specifying an automatic contour
        scheme or explicit contour levels.
    ctr_cmap : str or Colormap, optional
        Colormap used for contours.
    ctr_width : float, optional
        Line width of the contour lines.

    left_loc, right_loc, lower_loc, upper_loc : float, optional
        Axes-relative locations for annotations such as velocity labels,
        beam symbol, and scale bar.

    kpc : float, optional
        Physical length (in kpc) of the scale bar.
    asectokpc : float, optional
        Conversion factor from arcseconds to kiloparsecs.
    color : str, optional
        Color used for text and annotation elements.
    y_offset : float, optional
        Vertical offset for annotation placement.
    colorbar_label : str, optional
        Label for the colorbar. If not provided, it is inferred from `fluxunits`.

    output : str, optional
        Path to save the generated figure. If None, the figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure.
    axis_list : list of matplotlib.axes.Axes
        List of axes corresponding to the channel map panels.

    Notes
    -----
    - If velocity information is available, each panel is annotated with the
    line-of-sight velocity relative to `vsys`.
    - Negative contours are automatically added as dashed gray lines.
    - Unused subplots are hidden if the grid contains more panels than channels.
    """

    #load the data
    data_dict=load_data(datacube)
    
    #store the relevant info
    data=data_dict['data']
    if wcs is None: wcs=data_dict['wcs']
    if wcs is not None and wcs.naxis == 3: wcs=wcs.dropaxis(2)
    if bmaj is None: bmaj=data_dict['bmaj']
    if bmin is None: bmin=data_dict['bmin']
    if bpa is None: bpa=data_dict['bpa']
    if fluxunits is None: fluxunits=data_dict['fluxunits']
    if pixunits is None: pixunits=data_dict['pixunits']
    if specunits is None: specunits=data_dict['specunits']
    if pixelres is None: pixelres=data_dict['pixelres']
    if spectralres is None: spectralres=data_dict['spectralres']
    if z0 is None: z0=data_dict['z0']
    if colorbar_label is None: colorbar_label=f'Flux [{fluxunits}]'
    if any(x is None for x in (bmaj,bmin,spectralres)): use_column_density=False

    #convert spatial units
    pixres_arcsec=pixelres
    bmaj_arcsec=bmaj
    bmin_arcsec=bmin
    if pixunits == "deg":
        if pixres_arcsec is not None: pixres_arcsec*=3600
        if bmaj_arcsec   is not None: bmaj_arcsec*=3600
        if bmin_arcsec   is not None: bmin_arcsec*=3600

    #convert spectral units
    specres_kms=spectralres
    z0_kms=z0
    vsys_kms=vsys
    if specunits == "m/s":
        if specres_kms is not None: specres_kms/=1000
        if z0_kms      is not None: z0_kms/=1000
        if vsys_kms    is not None: vsys_kms/=1000
    
    #create the velocity range
    v=z0_kms+np.arange(data.shape[0])*specres_kms if all(x is not None for x in (z0_kms,specres_kms)) else None    
    
    #use full spatial range if not specified
    if xmin is None or xmin < 0: xmin=0
    if xmax is None or xmax > data.shape[2]: xmax=data.shape[2]
    if ymin is None or ymin < 0: ymin=0
    if ymax is None or ymax > data.shape[1]: ymax=data.shape[1]

    #determine the channel range
    if first_channel is not None and first_channel < 0: first_channel=0
    if last_channel is not None and last_channel > data.shape[0]: last_channel=data.shape[0]
    if any(x is None for x in (first_channel,last_channel)): #if no channel range is set, determine it from the central channel
        central_channel=np.argmin(np.abs(v-vsys_kms)) if all(x is not None for x in (v,vsys_kms)) else data.shape[0]//2
        first_channel=central_channel-n_channels//2*channel_step
        last_channel=central_channel+n_channels//2*channel_step

    #extract the subcube
    data=data[first_channel:last_channel+1:channel_step,ymin:ymax,xmin:xmax]
    if wcs is not None: wcs=wcs[ymin:ymax,xmin:xmax]
    if v is not None: v=v[first_channel:last_channel+1:channel_step]

    #determine number of channels and grid layout
    ncols=int(np.ceil(np.sqrt(n_channels)))
    nrows=int(np.ceil(n_channels/ncols))
    
    #define the normalization
    norm=_norm(norm,data,vmin,vmax)
        
    #define the contours
    if isinstance(ctr,(list,np.ndarray)) and use_column_density:
        ctr=convert_from_HI(np.asarray(ctr)*10.**19,beamarea=1.13*bmaj_arcsec*bmin_arcsec,spectralres=np.abs(specres_kms),from_moment_map=False) #convert back into Jy/beam
    else:
        ctr=_contours(ctr,data,vmax=vmax,norm=norm)

    #create the figure
    fig,axis_list=make_figure(nrows=nrows,ncols=ncols,
                              fig_x_size=fig_x_size,fig_y_size=fig_y_size,
                              wspace=wspace,hspace=hspace,
                              wcs=wcs,
                              colorbar=True)
    
    #hide unused axes
    for i in range(n_channels,len(axis_list)):
        axis_list[i].set_visible(False)

    #loop over channels and plot each panel
    for idx in range(n_channels):
        row=idx//ncols
        col=idx%ncols
        
        ax=axis_list[idx]
        im=ax.imshow(data[idx],origin='lower',cmap=cmap,norm=norm,aspect='equal')
        #add the contours
        ax.contour(data[idx],levels=ctr,cmap=ctr_cmap,linewidths=ctr_width,linestyles='solid')
        ax.contour(data[idx],levels=-np.flip(ctr),colors='gray',linewidths=ctr_width,linestyles='dashed') #add the negative contours
        #add the channel velocity
        if vsys_kms is None: vsys_kms=0
        if v is not None: ax.text(left_loc,lower_loc,f'v$_{{los}}$: {v[idx]-vsys_kms:.1f} km/s',color=color,transform=ax.transAxes)
        #add the beam
        if all(x is not None for x in (pixres_arcsec,bmaj_arcsec,bmin_arcsec,bpa)):
            ax=_plot_beam(ax,pixres_arcsec,bmaj_arcsec,bmin_arcsec,bpa,
                           xloc=right_loc,yloc=lower_loc,color=color)
        #add the kpc line
        if all(x is not None for x in (pixres_arcsec,asectokpc)):
            if col == 0: #in the first column
                    ax=_plot_kpc_scale(ax,pixres_arcsec,asectokpc,
                                        xloc=left_loc,yloc=upper_loc,kpc=kpc,color=color,y_offset=y_offset)
               
        #add the colorbar
        if row == 0: #in the first row
            if use_column_density and all(x is not None for x in (bmaj_arcsec,bmin_arcsec,specres_kms)):
                #convert the contour levels to column density in units of 10^19 cm^-2
                ctr_label=np.around(
                          convert_to_HI(ctr,beamarea=1.13*bmaj_arcsec*bmin_arcsec,spectralres=np.abs(specres_kms),from_moment_map=False)/10.**19
                          ,2)
                colorbar_label=r'HI column density [$10^{19}$ cm$^{-2}$]'
            else:
                ctr_label=None
            ax=_add_colorbar(fig,ax,im,norm,ctr,
                             label=colorbar_label,
                             ctr_cmap=ctr_cmap,ctr_label=ctr_label)

    if output is not None:
        fig.savefig(output,bbox_inches='tight',dpi=dpi) #save the figure
    return fig,axis_list #return the figure and axis list
    
########################################################################################################

def convert_to_HI(data,beamarea,
                  *,
                  spectralres=None,from_moment_map=False):
    """
    Convert input data to HI column density units.

    This function converts spectral-line data or moment-0 maps into HI column
    density, applying the appropriate velocity integration and beam-area
    normalization.

    Parameters
    ----------
    data : array_like
        Input data values. This can be a spectral data cube/slice (e.g. brightness
        temperature per channel) or a moment-0 map, depending on `from_moment_map`.
    beamarea : float
        Beam area in the same angular units used for the data. Must be a positive
        number.
    spectralres : float, optional
        Spectral resolution (channel width) in velocity units. Required when
        `from_moment_map` is False. The absolute value is used.
    from_moment_map : bool, optional
        If True, the input `data` is assumed to be a moment-0 map already integrated
        over velocity. If False (default), the data are assumed to be per-channel
        values and will be integrated using `spectralres`.

    Returns
    -------
    numpy.ndarray
        Array of HI column density values derived from the input data.

    Raises
    ------
    ValueError
        If `beamarea` is not a positive number.
    ValueError
        If `spectralres` is not provided when `from_moment_map` is False.

    Notes
    -----
    The conversion uses a fixed multiplicative constant of 1.25 × 10^24 and applies
    a velocity integration factor `dv`. For moment-0 maps, `dv` is set to 1.0,
    assuming the data are already integrated over velocity.
    """

    if beamarea is None or beamarea <= 0:
        raise ValueError("Beam area must be a positive number.")

    if not from_moment_map:
        if spectralres is None:
            raise ValueError("spectralres must be provided when from_moment_map=False.")
        dv=np.abs(spectralres)
    else:
        dv=1.0  #moment-0 maps are already integrated over velocity

    constant=1.25*10.**24

    return np.asarray(data)*dv*constant/beamarea

#########################################################################################################
    
def convert_from_HI(data,beamarea,
                    *,
                    spectralres=None,from_moment_map=False):
    """
    Convert HI data values using a beam area and velocity integration.

    This function converts input data (e.g., HI intensity or moment values)
    into physical units using a fixed conversion constant, the beam area, and
    optionally the spectral resolution. It supports both spectral-cube data
    and pre-integrated moment-0 maps.

    Parameters
    ----------
    data : array_like
        Input data values to be converted. This can be a scalar or any array-like
        object and will be converted to a NumPy array.
    beamarea : float
        Beam area associated with the data. Must be a positive number.
    spectralres : float, optional
        Spectral resolution (channel width) of the data, typically in velocity
        units. Required when `from_moment_map` is False.
    from_moment_map : bool, optional
        If True, the input data are assumed to come from a moment-0 map and are
        already integrated over velocity. In this case, `spectralres` is ignored.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Converted data values as a NumPy array with the same shape as the input.

    Raises
    ------
    ValueError
        If `beamarea` is None or not positive.
        If `from_moment_map` is False and `spectralres` is not provided.

    Notes
    -----
    When `from_moment_map` is False, the absolute value of `spectralres` is used
    as the velocity width per channel. A fixed conversion constant of
    1.25 × 10^24 is applied internally.
    """

    if beamarea is None or beamarea <= 0:
        raise ValueError("Beam area must be a positive number.")

    if not from_moment_map:
        if spectralres is None:
            raise ValueError("spectralres must be provided when from_moment_map=False.")
        dv=np.abs(spectralres)
    else:
        dv=1.0  #moment-0 maps are already integrated over velocity

    constant=1.25*10.**24

    return np.asarray(data)*beamarea/(dv*constant)

##########################################################################################################

def fits_cutout(input_fits,
                *,
                xmin=None,xmax=None,ymin=None,ymax=None,
                zmin=None,zmax=None,
                output=None):
    """
    Create a spatial (and optionally spectral) cutout from a FITS file.

    This function loads a FITS file, extracts a subregion of the data array
    based on the provided index ranges, updates the WCS information in the
    header accordingly, and either writes the result to a new FITS file or
    returns the cutout data and header.

    The function supports both 2D (y, x) and 3D (z, y, x) FITS data. Any
    unspecified or out-of-bounds limits default to the full valid range
    along the corresponding axis.

    Parameters
    ----------
    input_fits : str
        Path to the input FITS file to be cut.
    xmin, xmax : int or None, optional
        Minimum and maximum x-axis indices for the cutout. If None or
        out of bounds, the full valid x-range is used.
    ymin, ymax : int or None, optional
        Minimum and maximum y-axis indices for the cutout. If None or
        out of bounds, the full valid y-range is used.
    zmin, zmax : int or None, optional
        Minimum and maximum z-axis indices for the cutout (only used for
        3D data). If None or out of bounds, the full valid z-range is used.
    output : str or None, optional
        Path to the output FITS file. If provided, the cutout is written
        to disk and nothing is returned. If None, the cutout data and
        updated header are returned.

    Returns
    -------
    data : numpy.ndarray
        The cutout data array. Only returned if `output` is None.
    header : astropy.io.fits.Header or None
        The updated FITS header containing the modified WCS. Only returned
        if `output` is None.

    Notes
    -----
    - Indexing follows standard NumPy slicing semantics (start inclusive,
    stop exclusive).
    - If a header is present, the WCS is recalculated for the cutout region
    and written back into the header.
    - Existing files at the output path are overwritten if `output` is set.
    """

    #load the data
    data_dict=load_data(input_fits)
    
    #store the relevant info
    data,header=data_dict['data'],data_dict['header']

    #use full ranges if not specified
    if xmin is None or xmin < 0: xmin=0
    if ymin is None or ymin < 0: ymin=0
    if zmin is None or zmin < 0: zmin=0

    if data.ndim == 3:
        if zmax is None or zmax > data.shape[0]: zmax=data.shape[0]
        if xmax is None or xmax > data.shape[2]: xmax=data.shape[2]
        if ymax is None or ymax > data.shape[1]: ymax=data.shape[1]
    elif data.ndim == 2:
        if xmax is None or xmax > data.shape[1]: xmax=data.shape[1]
        if ymax is None or ymax > data.shape[0]: ymax=data.shape[0]

    #extract the subdata
    if data.ndim == 3:
        data=data[zmin:zmax,ymin:ymax,xmin:xmax]
    elif data.ndim == 2:
        data=data[ymin:ymax,xmin:xmax]
    
    #extract the new wcs and update the header
    if header is not None:
        if data.ndim == 3:
            wcs=WCS(header)[zmin:zmax,ymin:ymax,xmin:xmax] #store the wcs
        elif data.ndim == 2:
            wcs=WCS(header)[ymin:ymax,xmin:xmax] #store the wcs
        newheader=wcs.to_header() #write the wcs into a header
        for key, value in newheader.items():
            header[key] = value
    
    #write the output
    if output is not None:
        fits.PrimaryHDU(data,header=header).writeto(output,overwrite=True)
    #return the data and header  
    else:
        return data,header

##########################################################################################################

def hi_mass(data,
            *,
            beamarea=None,distance=None,pixelres=None,
            xmin=None,xmax=None,ymin=None,ymax=None):
    """
    Compute the total HI mass from an integrated HI intensity map.

    This function converts an HI moment-0 map (or a subregion of it) into a total
    neutral hydrogen (HI) mass using standard radio astronomy conventions.

    Parameters
    ----------
    data : array_like
        Integrated HI intensity in units of Jy/beam·km/s. Can be a 2D array
        representing a spatial map or any array that can be summed.
    beamarea : float, keyword-only
        Beam area in arcsec². Must be provided and non-zero.
    distance : float, keyword-only
        Distance to the source in Mpc. Must be provided and non-zero.
    pixelres : float, keyword-only
        Pixel angular size in arcsec.
    xmin, xmax : int, optional
        Minimum and maximum x pixel indices defining the subregion to use.
        If not provided, the full x-range is used.
    ymin, ymax : int, optional
        Minimum and maximum y pixel indices defining the subregion to use.
        If not provided, the full y-range is used.

    Returns
    -------
    float
        Total HI mass in solar masses (M☉).

    Raises
    ------
    ValueError
        If `beamarea`, `distance`, or `pixelres` are not provided or are zero.

    Notes
    -----
    NaN values in the input data are ignored when summing the flux. The conversion
    uses a standard factor of 2.35 × 10⁵ to convert integrated flux and distance
    into HI mass.
    """

    #check the input
    if beamarea is None or beamarea == 0:
        raise ValueError("Provide the beam area in arcsec^2.")
    if beamarea < 0:
        beamarea=np.abs(beamarea)
    if distance is None or distance == 0:
        raise ValueError("Provide the distance in Mpc.")
    if pixelres is None or pixelres == 0:
        raise ValueError("Provide the pixel resolution in arcsec.")

    #convert the data into an array
    data=np.asarray(data)

    if data.ndim == 2:
        #use full spatial range if not specified
        if xmin is None: xmin=0
        if xmax is None: xmax=data.shape[1]
        if ymin is None: ymin=0
        if ymax is None: ymax=data.shape[0]
        
        #extract the subregion
        data=data[ymin:ymax, xmin:xmax]
    
    #calculate the total flux
    flux=np.nansum(data)
        
    #units conversion factor
    constant=2.35*10.**5

    return constant*distance**2*pixelres**2*flux/beamarea

##########################################################################################################

def lines_finder(spectrum,
                 *,
                 spectrum_rms=None,
                 smooth_kernel=1,
                 finder_threshold=4,
                 finder_replace=None,
                 finder_fluxrange='negative',
                 finder_statistic='mad',
                 linker_kernel=1,
                 linker_minsize=1,
                 reliability=False,
                 reliability_threshold=0.8,
                 reliability_snrmin=5,
                 reliability_kernel=0.4,
                 reliability_tolerance=1e-12,
                 reliability_plot=False,
                 reliability_skellam=False,
                 gaussianity=False,
                 gaussianity_fluxrange='negative',
                 gaussianity_statistic='mad',
                 gaussianity_tries=10000,
                 keep_negative=False,
                 write_catalogue=False,
                 output=None,
                 **kwargs):
    """
    Detects and characterizes spectral line features in one or more 1D spectra.

    This is the main high-level routine of the pipeline. It performs noise
    estimation, optional smoothing, threshold-based source finding, source
    linking, optional Gaussianity tests, reliability estimation based on
    positive/negative detections, and optional catalogue generation.

    Parameters
    ----------
    spectrum : array-like
        Input spectrum or collection of spectra. Can be a 1D NumPy array,
        a 2D array (one spectrum per row), or a list/tuple of arrays.
    spectrum_rms : float or array-like, optional
        Noise (RMS) estimate per spectrum. If not provided or NaN, the RMS
        is estimated internally from the data.
    smooth_kernel : int or array-like, default=1
        Boxcar kernel size(s) used for smoothing during source finding.
        Multiple kernel sizes can be provided.
    finder_threshold : float, default=4
        Detection threshold in units of RMS.
    finder_replace : float, optional
        Replacement value (in units of RMS) used to mask detected channels
        prior to smoothing, to avoid source smearing.
    finder_fluxrange : {'negative', 'positive', 'all'}, default='negative'
        Flux range used for RMS estimation during source finding.
    finder_statistic : {'mad', 'std'}, default='mad'
        Statistic used to estimate the RMS.
    linker_kernel : int, default=1
        Maximum allowed gap (in channels) between detections when linking
        them into a single source.
    linker_minsize : int, default=1
        Minimum number of detected channels required for a valid source.
    reliability : bool, default=False
        If True, compute source reliability using positive/negative symmetry.
    reliability_threshold : float, default=0.8
        Minimum reliability value for a source to be considered reliable.
    reliability_snrmin : float, default=5
        Minimum signal-to-noise ratio required for a source to be considered
        in the reliability analysis.
    reliability_kernel : float, default=0.4
        Scaling factor applied to the covariance matrix in reliability KDE.
    reliability_tolerance : float, default=1e-12
        Diagonal regularization term added to the covariance matrix.
    reliability_plot : bool, default=False
        If True, generate diagnostic reliability plots.
    reliability_skellam : bool, default=False
        If True, generate a Skellam distribution diagnostic plot.
    gaussianity : bool, default=False
        If True, perform Gaussianity tests on the noise.
    gaussianity_fluxrange : {'negative', 'positive', 'all'}, default='negative'
        Flux range used for RMS estimation in Gaussianity tests.
    gaussianity_statistic : {'mad', 'std'}, default='mad'
        Statistic used for RMS estimation in Gaussianity tests.
    gaussianity_tries : int, default=10000
        Number of Monte Carlo realizations used to calibrate the Kanekar test.
    keep_negative : bool, default=False
        If True, retain negative detections in the output mask.
    write_catalogue : bool, default=False
        If True, write a source catalogue to disk.
    output : str, optional
        Output filename prefix for plots and catalogues.
    **kwargs
        Additional keyword arguments passed to plotting or reliability routines. See _reliability.

    Returns
    -------
    mask : ndarray
        Binary mask array with the same shape as the input spectra, where
        detected source channels are marked with 1 (and optionally -1 for
        negative detections).
    """

    #determine the number of spectra to process
    if isinstance(spectrum,np.ndarray):
        nspectra=len(spectrum) if spectrum.ndim > 1 or spectrum.dtype == object else 1
    elif isinstance(spectrum,(list,tuple)):
        nspectra=len(spectrum) if any(isinstance(x, (list,tuple,np.ndarray)) for x in spectrum) else 1

    #ensure input spectrum is an iterable
    if spectrum.ndim==1: spectrum=np.array([spectrum])
    
    #ensure there is at least a noise value per spectrum
    noise=_extend(spectrum_rms,nspectra)

    #ensure kernel list is an array
    kernels=np.atleast_1d(smooth_kernel)

    if finder_replace is None: finder_replace=finder_threshold/2
    if output is None and reliability:
        reliability_plot=False
        reliability_skellam=False

    if gaussianity:
        #initialise the gaussianity checks
        gaussian_for_anderson=np.zeros(nspectra)
        gaussian_for_kanekar=np.zeros(nspectra)
        gaussianity_kernel=Box1DKernel(10)
        gaussianity_nchans=0

    #initialize the list of mask and source IDs
    mask_list=[]
    source_ids=[]

    #do the source finding
    for i in tqdm(range(len(spectrum)),desc='Spectra analysed',total=len(spectrum)): #for each input spectrum
        #extract the spectrum
        spec=spectrum[i].copy()
        
        #extract the noise value
        sigma=noise[i]

        #if not given, calculate and store into the noise array for later use
        if sigma is None or np.isnan(sigma):
            sigma=RMS(spec,mode=finder_statistic,fluxrange=finder_fluxrange)
            noise[i]=sigma
        
        #run the source finder
        mask=_source_finding(spec,sigma,
                             smooth_kernel=kernels,
                             finder_threshold=finder_threshold,
                             finder_replace=finder_replace,
                             finder_fluxrange=finder_fluxrange,
                             finder_statistic=finder_statistic)

        #run the source linker
        mask=_source_linker(spec,mask,linker_kernel,linker_minsize)

        #store the current mask, which cointains the sources IDs, to the sources IDs list
        source_ids.append(mask.copy())

        #perform the gaussianity tests
        if gaussianity:
            #the Kanekar parameter has expected value of 1.
            #But the finite number of channels and the correlation introduced by smoothing create a range of values.
            #Let check what is the expected range of values (which depends on the number of channels) 

            #if the spectrum has a different number of channels with respect to the others
            if len(spec) != gaussianity_nchans:
                #store the new number of channels
                gaussianity_nchans=len(spec)
                
                #initialize the test result list
                test=[]
                for tries in range(gaussianity_tries):
                    #create a Gaussian spectrum
                    spec=np.random.normal(size=gaussianity_nchans)

                    #compute the unsmoothed spectrum
                    rms_0=RMS(spec)

                    #convolve the spectrum.
                    #Wrap the boundaries to fix convolution gliches at edges and preserve noise properties
                    spec=convolve(spec,gaussianity_kernel,boundary='wrap')

                    #compute the rms of the smoothed spectrum
                    rms_1=RMS(spec)

                    #store the expected Kanekar parameter
                    test.append(rms_1*np.sqrt(10)/rms_0)

                #compute the 95% range
                lower,upper=np.percentile(np.array(test),[2.5,97.5])

            #test for Gaussianity for anderson-darlin and kanekar
            anders,kanekar=_gaussianity(spec,mask,
                                        statistic=gaussianity_statistic,
                                        fluxrange=gaussianity_fluxrange)

            #Gaussian for Kanekar if within the range.
            #anders is already boolean
            k=(lower < kanekar and kanekar < upper)

            #store the results
            gaussian_for_anderson[i]=anders
            gaussian_for_kanekar[i]=k

            #convert into array
            gaussian_for_anderson=np.array(gaussian_for_anderson,dtype=bool)
            gaussian_for_kanekar=np.array(gaussian_for_kanekar,dtype=bool)
        else:
            gaussian_for_anderson,gaussian_for_kanekar=None,None

                
        #make the mask binary
        mask[mask < 0]=-1 if keep_negative else 0 
        mask[mask > 0]=1 

        #store the mask
        mask_list.append(mask)

    #calculate the reliability
    if reliability:
        #split the positive and negative sources for the reliability computation
        pos,neg=_split_sources(spectrum,noise,source_ids)

        #run the reliability computation
        reliability_values=_reliability(pos,neg,reliability_snrmin,
                                        kernel=reliability_kernel,
                                        eps=reliability_tolerance,
                                        threshold=reliability_threshold,
                                        plot=reliability_plot,
                                        skellam=reliability_skellam,
                                        output=output,
                                        **kwargs)

        #remove unreliable sources from the mask
        k=0
        
        #run through the source_ids array.
        #Each element i contains the sources IDs for each channel of spectrum i
        for i,src_ids in enumerate(source_ids):
            #skip if no positive sources
            if np.all(src_ids <= 0):
                continue

            #positive source IDs
            positive_sources=src_ids[src_ids > 0]

            #for each positive source
            for j in positive_sources:
                #if unreliable
                if reliability_values[k+j-1] < reliability_threshold:
                    #channels belonging to source j
                    mask=(src_ids == j)
                    idx=np.where(mask)[0]

                    #minimum, maximum and total channel
                    chanmin=idx[0]
                    chanmax=idx[-1]

                    #remove the unrealiable source j from the mask
                    mask_list[i][chanmin:chanmax+1]=0

            #update the reliability index
            k+=np.nanmax(positive_sources)
    else:
        reliability_values=None

    #if a catalogue must be written
    if write_catalogue and output:
        catalogue=_write_catalogue(spectrum,source_ids,noise,
                                   gaussianity=gaussianity,
                                   gaussian_for_anderson=gaussian_for_anderson,
                                   gaussian_for_kanekar=gaussian_for_kanekar,
                                   reliability=reliability,
                                   reliability_values=reliability_values)

        #create a copy of the catalogue to retain only the reliable source
        filtered_catalogue=catalogue.copy()
        if reliability:
            #filter unreliable sources
            filtered_catalogue=filtered_catalogue.drop(filtered_catalogue[filtered_catalogue['Reliability']<reliability_threshold].index)
        if gaussianity:
            #filter non-Gaussian spectra
            filtered_catalogue=filtered_catalogue.drop(filtered_catalogue[(filtered_catalogue['Anderson test']) & (filtered_catalogue['Kanekar test'])].index)
        
        #save the catalouges
        catalogue.to_csv(output+'_sources_catalogue.csv',index=False)
        filtered_catalogue.to_csv(output+'_sources_catalogue_filtered.csv',index=False) 

    return np.array(mask_list)

##############################################################################################################

def load_data(input_data,
              *,
              ext=None):
    """
    Load astronomical data from a FITS file, a NumPy array, or an (array, header) tuple
    and return a standardized dictionary with data, metadata, WCS information, and
    derived quantities.

    Parameters
    ----------
    input_data : str or numpy.ndarray or tuple
        Input data source. Can be:
        - A string path to a FITS file.
        - A NumPy array containing the data.
        - A tuple of the form (data, header), where `data` is a NumPy array and
        `header` is an astropy.io.fits.Header instance.
    ext : int or str, optional, keyword-only
        FITS extension to load when `input_data` is a FITS file. If None, the first
        HDU containing array data is automatically selected.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data' : numpy.ndarray
            The loaded data array.
        - 'header' : astropy.io.fits.Header or None
            FITS header associated with the data, if available.
        - 'wcs' : astropy.wcs.WCS or None
            World Coordinate System constructed from the header.
        - 'pixelres' : float or None
            Spatial pixel resolution derived from the WCS.
        - 'spectralres' : float or None
            Spectral resolution (for 3D data cubes only).
        - 'x0', 'y0', 'z0' : float or None
            Reference world coordinates at the image/cube center.
        - 'fluxunits' : str
            Flux units from the FITS header (BUNIT).
        - 'pixunits' : str
            Spatial units from the FITS header (CUNIT1).
        - 'specunits' : str
            Spectral units from the FITS header (CUNIT3).
        - 'bmaj', 'bmin', 'bpa' : float or None
            Beam major axis, minor axis, and position angle.

    Raises
    ------
    TypeError
        If `input_data` is not a supported type.
    ValueError
        If a FITS file contains no HDU with data.
    """

    #check if the input is a string (assume it's a path to a FITS file)
    if isinstance(input_data, str):
        #open the FITS file
        with fits.open(input_data) as hdul:
            #if the user provided an extension, use it
            if ext is not None:
                hdu=hdul[ext]
            else:
                #auto-detect first extension with data
                for h in hdul:
                    if isinstance(h.data,np.ndarray):
                        hdu=h
                        break
                else:
                    raise ValueError("No HDU with data found in FITS file.")
            #the data is usually in the first HDU (Header/Data Unit)
            data=hdu.data
            #the header contains metadata about the observation
            header=hdu.header
        del hdul
    
    #check if the input is a NumPy array
    elif isinstance(input_data,np.ndarray):
        #use the array as the data
        data=input_data.copy()
        #no header is available
        header=None

    #check if the input is a tuple
    elif isinstance(input_data,tuple):
        if isinstance(input_data[0],np.ndarray):
            data=input_data[0].copy()
        else:
            raise TypeError("Input must be a path to a FITS file (string), a NumPy array or a (array,header) tuple.")
        if isinstance(input_data[1],fits.header.Header):
            header=input_data[1].copy()
        else:
            raise TypeError("Input must be a path to a FITS file (string), a NumPy array or a (array,header) tuple.")
    
    #if the input is neither a string nor a NumPy array, raise an error
    else:
        raise TypeError("Input must be a path to a FITS file (string), a NumPy array or a (array,header) tuple.")
    
    #create the WCS object from the header
    wcs=WCS(header) if header is not None else None

    #compute the pixel scale and spectral resolution
    if wcs is not None:
        scales=np.diag(wcs.pixel_scale_matrix) #extract the pixel scales
        pixelres=np.abs(scales[0]) #pixel resolution
        if wcs.naxis == 2:
            spectralres=None
        elif wcs.naxis == 3:
            spectralres=(scales[2]) #spectral resolution
    else:
        pixelres,spectralres=None,None

    #create the data dictionary
    data_dict={
        # data and metadata
        'data': data,
        'header': header,
        'wcs': wcs,

        #spatial and spectral resolution
        'pixelres': pixelres,
        'spectralres': spectralres,

        #image center and spectral reference
        'x0': header.get('CRVAL1')+(header.get('CRPIX1')-1)*scales[0] if header is not None else None,
        'y0': header.get('CRVAL2')+(header.get('CRPIX2')-1)*scales[1] if header is not None else None,
        'z0': header.get('CRVAL3')+(header.get('CRPIX3')-1)*spectralres if all(x is not None for x in (header,spectralres)) else None,

        #units
        'fluxunits': header.get('BUNIT') if header is not None else '',
        'pixunits': header.get('CUNIT1') if header is not None else '',
        'specunits': header.get('CUNIT3') if header is not None else '',

        #beam parameters
        'bmaj': header.get('BMAJ') if header is not None else None,
        'bmin': header.get('BMIN') if header is not None else None,
        'bpa': header.get('BPA') if header is not None else None,
    }
    
    return data_dict

###########################################################################################################
 
def make_figure(**kwargs):
    """
    Create a Matplotlib figure and axes with flexible layout options, including
    support for WCS projections, multiple panels, shared formatting, and aliases
    for common parameters.

    Parameters
    ----------
    **kwargs
        Keyword arguments controlling figure and axes creation. Supported options
        include (but are not limited to):

        Figure layout:
        - fig_x_size, fig_y_size : float
            Base size of each panel in inches.
        - nrows, ncols : int
            Number of rows and columns of subplots.
        - wspace, hspace : float
            Width and height spacing between subplots.

        Axes and projection:
        - wcs : astropy.wcs.WCS or None
            WCS projection to apply to all panels.

        Labels and ticks:
        - xlabels, ylabels : str or list of str
            Axis labels (single value or per-panel list).
        - xticks, yticks : array-like
            Tick locations.
        - xticklabels, yticklabels : array-like
            Tick labels.

        Limits and scaling:
        - xlim, ylim : tuple
            Axis limits.
        - xscale, yscale : str
            Axis scaling (e.g., 'linear', 'log').

        Miscellaneous:
        - colorbar : bool
            Whether a colorbar will be added externally (affects label placement).

        Common aliases such as `figx`, `figy`, `nrow`, `ncol`, `xlabel`, `ylabel`,
        `xticklabel`, and `yticklabel` are also supported.

    Returns
    -------
    (fig, ax) or (fig, axes)
        - fig : matplotlib.figure.Figure
            The created figure.
        - ax : matplotlib.axes.Axes
            Single axes object if nrows == ncols == 1.
        - axes : list of matplotlib.axes.Axes
            List of axes objects for multi-panel figures.
    """
    #handle parameter
    aliases={
            'figx': 'fig_x_size',
            'figy': 'fig_y_size',
            'nrow': 'nrows',
            'ncol': 'ncols',
            'xlabel': 'xlabels',
            'ylabel': 'ylabels',
            'xticklabel': 'xticklabels',
            'yticklabel': 'yticklabels'
    }

    #map aliases to canonical names
    for alias, canonical in aliases.items():
        if alias in kwargs and canonical not in kwargs:
            kwargs[canonical]=kwargs[alias]

    #extract parameters
    figx=kwargs.get("fig_x_size",7)
    figy=kwargs.get("fig_y_size",7)
    nrows=kwargs.get("nrows",1)
    ncols=kwargs.get("ncols",1)
    wcs=kwargs.get("wcs",None)
    wspace=kwargs.get("wspace",0.01)
    hspace=kwargs.get("hspace",0.01)
    xticks=kwargs.get("xticks",None)
    yticks=kwargs.get("yticks",None)
    xlabel=kwargs.get("xlabels",None)
    ylabel=kwargs.get("ylabels",None)
    xticklabels=kwargs.get("xticklabels",None)
    yticklabels=kwargs.get("yticklabels",None)
    xlim=kwargs.get("xlim",None)
    ylim=kwargs.get("ylim",None)
    xscale=kwargs.get("xscale",None)
    yscale=kwargs.get("yscale",None)
    colorbar=kwargs.get("colorbar",False)

    #helper function to pick the appropriate label for each panel
    def pick(label,idx):
        #if multiple labels are provided
        if isinstance(label,list):
            #return the corresponding label or the last one if idx exceeds the list length
            return label[idx] if idx < len(label) else label[-1]
        #if a single label is provided
        else:
            return label
        
    #create figure
    fig=plt.figure(figsize=(figx*ncols,figy*nrows))

    #single panel case
    if nrows == 1 and ncols == 1:
        #create the axis
        ax=fig.add_subplot(projection=wcs)

        #manage axis labels
        ax.set_xlabel(xlabel) if xlabel is not None else ax.set_xlabel('RA [J2000]' if wcs is not None else 'X [px]')
        ax.set_ylabel(ylabel) if ylabel is not None else ax.set_ylabel('DEC [J2000]' if wcs is not None else 'Y [px]')

        #manage ticks
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)

        #manage ticklabels
        if xticklabels is not None: ax.set_xticklabels(xticklabels)
        if yticklabels is not None: ax.set_yticklabels(yticklabels)

        #manage limits
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)

        #manage the axis scale
        if xscale is not None: ax.set_xscale(xscale)
        if yscale is not None: ax.set_yscale(yscale)

        return fig,ax
    
    #multiple panel case
    elif nrows > 1 or ncols > 1:
        #create grid specification
        gs=fig.add_gridspec(nrows,ncols,wspace=wspace,hspace=hspace)
        
        #create the panels
        axis_list=[] #list to store the axes
        xlabel_idx,ylabel_idx=0,0 #indices to track xlabel and ylabel for multiple labels
        for idx in range(nrows*ncols):
            #determine row and column index
            row=idx//ncols
            col=idx%ncols
            
            #create the axis
            ax=fig.add_subplot(gs[row,col],projection=wcs)

            #manage axis labels
            if not colorbar and row == 0: #if plotting the first row and a colorbar will not be added
                if wcs is not None:
                    ax.coords[0].set_ticklabel_position('t') #move the x-axis label to the top
                    ax.coords[0].set_axislabel_position('t') #move the x-axis tick labels to the top
                else:
                    ax.xaxis.set_label_position("top") #move the x-axis label to the top
                    ax.xaxis.tick_top() #move the x-axis tick labels to the top
            elif row == nrows-1: #if plotting the last row
                ax.set_xlabel(pick(xlabel,xlabel_idx)) if xlabel is not None else ax.set_xlabel('RA [J2000]' if wcs is not None else 'X [px]')
                xlabel_idx+=1
            else:
                #hide the x-axis ticklabels and labels
                if wcs is not None:
                    ax.coords[0].set_ticklabel_visible(False)
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
            
            if col == 0: #if plotting the first column
                ax.set_ylabel(pick(ylabel,ylabel_idx)) if ylabel is not None else ax.set_ylabel('DEC [J2000]' if wcs is not None else 'Y [px]')
                ylabel_idx+=1
            elif col == ncols-1: #if plotting the last column
                ax.set_ylabel(pick(ylabel,ylabel_idx)) if ylabel is not None else ax.set_ylabel('DEC [J2000]' if wcs is not None else 'Y [px]')
                ylabel_idx+=1
                if wcs is not None:
                    ax.coords[1].set_ticklabel_position('r') #move the y-axis label to the right
                    ax.coords[1].set_axislabel_position('r') #move the y-axis tick labels to the right
                else:
                    ax.yaxis.set_label_position("right") #move the y-axis label to the right
                    ax.yaxis.tick_right() #move the y-axis tick labels to the right
            else:
                if wcs is not None:
                    ax.coords[1].set_ticklabel_visible(False) #hide the y-axis ticklabels and labels
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])  

            if wcs is None:
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')

            #manage ticks
            if xticks is not None: ax.set_xticks(xticks)
            if yticks is not None: ax.set_yticks(yticks)

            #manage ticklabels
            if xticklabels is not None: ax.set_xticklabels(xticklabels)
            if yticklabels is not None: ax.set_yticklabels(yticklabels)

            #manage limits
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)

            #manage the axis scale
            if xscale is not None: ax.set_xscale(xscale)
            if yscale is not None: ax.set_yscale(yscale)
                
            axis_list.append(ax)
        return fig,axis_list
    
    #just the figure case
    else:
        return fig

############################################################################################################

def pvslice(datacube,
            *,
            width=3,
            start_point=None,end_point=None,
            length=None,angle=None,center=None,
            first_channel=None,last_channel=None,
            xmin=None,xmax=None,ymin=None,ymax=None,
            output=None,plot=False,
            **kwargs):
    """
    Extract a position–velocity (PV) slice from a spectral data cube and optionally
    write the result to a FITS file or plot it.

    Parameters
    ----------
    datacube : str or numpy.ndarray or tuple
        Input data cube, passed to `load_data`. Can be a FITS file path, a NumPy array,
        or an (array, header) tuple.
    width : int, optional, keyword-only
        Width of the PV slice in pixels.
    start_point, end_point : tuple of float, optional, keyword-only
        (x, y) pixel coordinates defining the start and end points of the PV slice.
    length : float, optional, keyword-only
        Length of the PV slice in pixels (used if start/end points are not provided).
    angle : float, optional, keyword-only
        Position angle of the PV slice in degrees (used with `length`).
    center : tuple of float, optional, keyword-only
        (x, y) pixel coordinates of the slice center (defaults to cube center).
    first_channel, last_channel : int, optional, keyword-only
        Channel range to include in the extraction.
    xmin, xmax, ymin, ymax : int, optional, keyword-only
        Spatial bounds for subsetting the data cube before extraction.
    output : str, optional, keyword-only
        Path to write the extracted PV slice as a FITS file.
    plot : bool, optional, keyword-only
        If True, plot the PV slice using `plot_pvslice`.
    **kwargs
        Additional keyword arguments passed to the plotting function plot_pvslice.

    Returns
    -------
    astropy.io.fits.PrimaryHDU or tuple
        If `plot` is False:
            - pv : astropy.io.fits.PrimaryHDU
                PV slice data and header.
        If `plot` is True:
            - fig : matplotlib.figure.Figure
            - ax : matplotlib.axes.Axes
            - pv : astropy.io.fits.PrimaryHDU

    Raises
    ------
    ValueError
        If insufficient parameters are provided to define the PV slice geometry.
    """

    #load the data
    data_dict=load_data(datacube)
    
    #store the relevant info
    data,header,wcs=data_dict['data'],data_dict['header'],data_dict['wcs']

    #define the channel range
    if first_channel is None or first_channel < 0: first_channel=0
    if last_channel is None or last_channel > data.shape[0]: last_channel=data.shape[0]

    #use full ranges if not specified
    if xmin is None or xmin < 0: xmin=0
    if ymin is None or ymin < 0: ymin=0
    if xmax is None or xmax > data.shape[2]: xmax=data.shape[2]
    if ymax is None or ymax > data.shape[1]: ymax=data.shape[1]    

    #extract the subcube
    data=data[first_channel:last_channel+1,ymin:ymax,xmin:xmax]
    wcs=wcs[first_channel:last_channel+1,ymin:ymax,xmin:xmax]

    #define the pv path
    if start_point is None or end_point is None:
        if length is None or angle is None:
            raise ValueError("If start_point and end_point are not provided, length and angle must be provided.")
        
        #calculate the start and end points from length, angle, and center
        if center is None:
            x0,y0=data.shape[2]//2,data.shape[1]//2
        else:
            x0,y0=center
        theta=np.radians(angle+90) #+90 accounts for the pvextractor angle definition
        
        #calculate the start and end points
        x1=x0-(length/2)*np.cos(theta)
        y1=y0-(length/2)*np.sin(theta)
        x2=x0+(length/2)*np.cos(theta)
        y2=y0+(length/2)*np.sin(theta)

        start_point,end_point=(x1,y1),(x2,y2)

    #extract the pv slice
    pv_path=Path([start_point,end_point],width=width)
    pv=extract_pv_slice(data,pv_path,wcs)

    #fix the header in order to have the distance from the path center as spatial dimension
    pv.header['CRPIX1']=round(pv.header['NAXIS1']/2)+1

    #add the flux units
    pv.header['BUNIT']=header.get('BUNIT')

    #write the pv slice to a fits file
    if output is not None:
        pv.writeto(output,overwrite=True)

    #plot the pv slice if requested
    if plot:
        fig,ax=plot_pvslice(pv.data,pv.header,**kwargs)
        return fig,ax,pv #return the figure, axis, pv data and header
    else:
        return pv #return the pv data and header

############################################################################################################

def plot_mom0(data,**kwargs):
    """
    Plot a moment-0 (integrated intensity) map.

    This is a convenience wrapper around `_plot_mom` with `which="mom0"`.
    All keyword arguments are forwarded to `_plot_mom`.

    Parameters
    ----------
    data : str or array-like
        Input data or path understood by `load_data`.
    **kwargs
        Additional keyword arguments passed to `_plot_mom`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
        
    return _plot_mom(data,which="mom0",**kwargs)

#############################################################################################################

def plot_mom1(data,**kwargs):
    """
    Plot a moment-1 (intensity-weighted velocity) map.

    This is a convenience wrapper around `_plot_mom` with `which="mom1"`.
    All keyword arguments are forwarded to `_plot_mom`.

    Parameters
    ----------
    data : str or array-like
        Input data or path understood by `load_data`.
    **kwargs
        Additional keyword arguments passed to `_plot_mom`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """

    return _plot_mom(data,which="mom1",**kwargs)

#############################################################################################################

def plot_mom2(data,**kwargs):
    """
    Plot a moment-2 (velocity dispersion) map.

    This is a convenience wrapper around `_plot_mom` with `which="mom2"`.
    All keyword arguments are forwarded to `_plot_mom`.

    Parameters
    ----------
    data : str or array-like
        Input data or path understood by `load_data`.
    **kwargs
        Additional keyword arguments passed to `_plot_mom`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """

    return _plot_mom(data,which="mom2",**kwargs)

############################################################################################################

def plot_noise(datacube,
               *,
               statistics='mad',flux_range='negative',
               norm='linear',vmin=0,vmax=1,
               cmap='Greys',color='blue',
               colorbar=True,flux_units='',
               fig_x_size=7,fig_y_size=7,
               mult_by=1,add_to=0,
               dpi=300,output=None):
    """
    Compute and visualize spatial and spectral noise properties of a 3D datacube.

    This function estimates noise using either the standard deviation or the
    median absolute deviation (MAD), optionally restricted to positive or
    negative flux values. It produces a two-panel figure showing the spatial
    RMS map (collapsed over channels) and the spectral RMS as a function of
    channel.

    Parameters
    ----------
    datacube : str or array-like
        Input datacube or path to a FITS file readable by ``load_data``.
    statistics : {'mad', 'std'}, optional
        Statistical estimator used to compute the RMS noise.
    flux_range : {'positive', 'negative'}, optional
        Flux range to consider when computing the noise.
    norm : str, optional
        Normalization scheme passed to the internal normalization helper.
    vmin, vmax : float, optional
        Lower and upper bounds for the image normalization.
    cmap : str, optional
        Colormap used for the spatial RMS map.
    color : str, optional
        Line color for the spectral RMS plot.
    colorbar : bool, optional
        Whether to add a colorbar to the spatial RMS map.
    flux_units : str, optional
        Flux units label; inferred from the FITS header if not provided.
    fig_x_size, fig_y_size : float, optional
        Base size of the figure in inches.
    mult_by, add_to : float, optional
        Scaling applied to the data prior to noise computation.
    dpi : int, optional
        Resolution of the output figure.
    output : str or None, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """

    #load the data
    data_dict=load_data(datacube)

    #store the relevant info
    data,header=data_dict['data'],data_dict['header']
    
    nchan,nx,ny=data.shape
    if flux_units == '' and header is not None: flux_units=header['BUNIT']
    
    #apply optional scaling
    data=data*mult_by+add_to

    #compute the spectral noise
    spectral_rms=np.zeros(nchan)

    for i in tqdm(range(nchan),desc="Computing spectral RMS"):
        spectral_rms[i]=RMS(data[i],mode=statistics,fluxrange=flux_range)

    #compute the spatial noise
    spatial_rms=np.zeros((nx,ny))
    flat_data=data.reshape(nchan,nx*ny)

    for idx in tqdm(range(nx*ny),desc="Computing spatial RMS"):
        spatial_rms.flat[idx]=RMS(flat_data[:,idx],mode=statistics,fluxrange=flux_range)

    #clean up glitchy pixels
    spatial_rms[spatial_rms == 0]=np.nan

    #create the figure
    nrows,ncols=2,1
    fig=plt.figure(figsize=(fig_x_size*ncols,fig_y_size*nrows))
    gs=fig.add_gridspec(nrows+1,ncols,hspace=0.1)
    ax1=fig.add_subplot(gs[0:2,0])
    ax1.set_xlabel('X [px]')
    ax1.set_ylabel('Y [px]')

    ax2=fig.add_subplot(gs[2,0])
    ax2.set_xlabel('Channel')
    ax2.set_ylabel(f'Noise [{flux_units}]')
    
    #define the normalization
    norm=_norm(norm,data,vmin,vmax)
    
    #plot the spatial rms
    im=ax1.imshow(spatial_rms,origin='lower',cmap=cmap)

    if colorbar:
        _add_colorbar(fig,ax1,im,norm,None,label=f'Noise [{flux_units}]')

    #plot the spectral rms
    ax2.plot(np.arange(nchan)+1,spectral_rms,c=color)    

    if output is not None:
        fig.savefig(output,bbox_inches='tight',dpi=dpi) #save the figure
    return fig

##########################################################################################################

def plot_pvslice(pv_data,pv_header,
                 *,
                 mult_by=1,add_to=0,
                 pixunits="deg",fluxunits='Jy/beam',pixelres=None,
                 vsys=None,
                 fig=None,ax=None,
                 fig_x_size=7,fig_y_size=7,
                 dpi=300,
                 cmap="Greys",
                 vmin=0.001,vmax=0.9,norm="log",
                 ctr="pow3",ctr_cmap="hot",ctr_width=1,
                 left_loc=0.05,upper_loc=0.9,
                 kpc=10,asectokpc=None,
                 color="black",y_offset=0.02,
                 colorbar=True,colorbar_label=None,
                 output=None):
    """
    Plot a position–velocity (PV) slice with optional contours, WCS axes,
    physical scale annotations, and colorbar.

    The function supports linear or logarithmic normalization, positive and
    negative contours, systemic velocity correction, and conversion of spatial
    units to physical scales (e.g., kpc).

    Parameters
    ----------
    pv_data : array-like
        2D PV slice data array.
    pv_header : astropy.io.fits.Header
        FITS header describing the PV slice WCS.
    mult_by, add_to : float, optional
        Scaling applied to the data before plotting.
    pixunits : str, optional
        Units of the spatial axis pixels.
    fluxunits : str, optional
        Flux units label.
    pixelres : float or None, optional
        Spatial pixel resolution; inferred from the WCS if not provided.
    vsys : float or None, optional
        Systemic velocity to subtract from the velocity axis.
    fig, ax : matplotlib objects, optional
        Existing figure and axis to draw on.
    fig_x_size, fig_y_size : float, optional
        Size of the figure in inches if a new figure is created.
    dpi : int, optional
        Resolution of the output figure.
    cmap : str, optional
        Colormap for the PV image.
    vmin, vmax : float, optional
        Normalization bounds for the image.
    norm : str, optional
        Normalization scheme.
    ctr : str, optional
        Contour definition scheme.
    ctr_cmap : str, optional
        Colormap used for contours.
    ctr_width : float, optional
        Line width of the contours.
    left_loc, upper_loc : float, optional
        Location of the physical scale bar in axes coordinates.
    kpc : float, optional
        Length of the physical scale bar in kiloparsecs.
    asectokpc : float or None, optional
        Conversion factor from arcseconds to kiloparsecs.
    color : str, optional
        Color used for annotations.
    y_offset : float, optional
        Vertical offset for annotation placement.
    colorbar : bool, optional
        Whether to add a colorbar.
    colorbar_label : str or None, optional
        Label for the colorbar.
    output : str or None, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the PV plot.
    """

    #copy the header to avoid modifying the original
    header=pv_header.copy()

    #apply data scaling
    data=pv_data.copy()
    data=data*mult_by+add_to

    #store the flux units
    if fluxunits is None: fluxunits=header.get('BUNIT')
    if colorbar_label is None: colorbar_label=f'Flux [{fluxunits}]'

    #create the WCS object from the header
    wcs=WCS(header)

    #compute the pixel scale
    if pixelres is None:
        scales=np.diag(wcs.pixel_scale_matrix) #extract the pixel scales
        pixelres=np.abs(scales[0]) #pixel resolution

    #set the pixel units
    if pixunits is None: pixunits=header.get('CUNIT1')

    #create the figure and axis if not provided
    if fig is None:
        fig=plt.figure(figsize=(fig_x_size,fig_y_size))
    if ax is None:
        if vsys is not None:
            header['CRVAL2']-=vsys #adjust the systemic velocity in the header
        ax=fig.add_subplot(projection=WCS(header))

    #define the normalization
    norm=_norm(norm,data,vmin,vmax)
        
    #define the contours
    ctr=_contours(ctr,data,vmax=vmax,norm=norm)

    #plot the pv data
    im=ax.imshow(data,aspect='auto',cmap=cmap,norm=norm)

    #add the contours
    ax.contour(data,levels=ctr,cmap=ctr_cmap,linewidths=ctr_width,linestyles='solid') #add the contours
    ax.contour(data,levels=-np.flip(ctr),colors='gray',linewidths=ctr_width,linestyles='dashed') #add the negative contours

    #set x-axis labels
    if asectokpc is not None:
        ax.coords[0].set_format_unit((u.arcsec / asectokpc)) #convert x units to kpc
        ax.set_xlabel('Distance [kpc]')
    else:
        ax.coords[0].set_format_unit(u.arcsec) #convert x units to arcsec
        ax.set_xlabel('Distance [arcsec]')

    #draw the path center line
    ax.axvline(x=header['CRPIX1']-1,ls='-.',c='black') #-1 is due to python stupid 0-couting
    
    #set y-axis labels
    if header['CUNIT2']=='m/s': #if the spectral units are m/s
            ax.coords[1].set_format_unit(u.km/u.s) #convert to km/s
            ax.set_ylabel(r'LoS velocity [km s$^{-1}$]')
    if vsys is not None: #if the systemic velocity is given
        ax.axhline(y=((0-header['CRVAL2'])/header['CDELT2'])+header['CRPIX2']-1,ls='--',c='black') #draw the systemic velocity line
        ax.set_ylabel(r'LoS velocity w.r.t. v$_{sys}$ [km s$^{-1}$]')     

    #add the kpc line
    pixres_arcsec=pixelres
    if pixunits == "deg":
        #convert spatial units
        if pixres_arcsec is not None: pixres_arcsec*=3600
    if all(x is not None for x in (pixres_arcsec,asectokpc)):
        ax=_plot_kpc_scale(ax,pixres_arcsec,asectokpc,
                            xloc=left_loc,yloc=upper_loc,kpc=kpc,color=color,y_offset=y_offset) #draw the kpc line
            
    #add the colorbar
    if colorbar:
        if header is not None and colorbar_label is None: colorbar_label=header.get('BUNIT')
        ax=_add_colorbar(fig,ax,im,norm,ctr,
                         label=colorbar_label,
                         ctr_cmap=ctr_cmap)

    #save the figure
    if output is not None:
        fig.savefig(output,bbox_inches='tight',dpi=dpi)
    return fig,ax #return the figure and axis list

############################################################################################################

def RMS(data,
        *,
        mode='mad',fluxrange='negative',
        if_empty='nan'):
    """
    Estimate the root-mean-square (RMS) noise of an array.

    The noise is assumed to be Gaussian and centered on zero. The RMS can be
    computed either via the standard deviation or the median absolute deviation
    (MAD), optionally filtering the data by flux sign.

    Parameters
    ----------
    data : array-like
        Input data array.
    mode : {'std', 'mad'}, optional
        Method used to estimate the RMS.
    fluxrange : {'positive', 'negative'}, optional
        Restrict the calculation to positive or negative values.
    if_empty : {'nan', 'error'}, optional
        Behavior when filtering removes all data.

    Returns
    -------
    float
        Estimated RMS value.

    Raises
    ------
    ValueError
        If an invalid mode is specified or if no data remain and
        ``if_empty='error'``.
    """

    #ensure data is an array
    data=np.asarray(data)

    #the noise is assumed to be gaussian and centered on 0
    value=0

    #flux filtering
    if fluxrange == 'positive':
        data=data[data > 0]
    elif fluxrange == 'negative':
        data=data[data < 0]

    if data.size == 0:
        if if_empty == 'error':
            raise ValueError("Filtering removed all data.")
        return np.nan

    if mode == "std":
        return np.sqrt(np.nanmean((data-value)**2))
    elif mode == "mad":
        return 1.4826*np.nanmedian(np.abs(data-value))
    else:
        raise ValueError("mode must be 'std' or 'mad'.")
    
############################################################################################################

def shuffle(input_fits,velocity_field,
            *,
            output_fits=None,spectralres=None,v0=None,fill_value=np.nan):
    """
    Spectrally align a datacube using a velocity field by circularly shifting
    each spectrum.

    For each spatial pixel, the spectrum is shifted such that the channel
    corresponding to the local velocity field value is aligned to the central
    channel. The operation preserves noise properties via circular shifts.

    Parameters
    ----------
    input_fits : str
        Input datacube FITS file.
    velocity_field : str
        FITS file containing the velocity field.
    output_fits : str or None, optional
        Path to write the shuffled datacube. If None, the data are returned.
    spectralres : float or None, optional
        Spectral resolution; inferred from the header if not provided.
    v0 : float or None, optional
        Starting velocity of the first channel.
    fill_value : float, optional
        Value used to replace NaNs in the velocity field before shuffling.

    Returns
    -------
    shuffled : ndarray
        Shuffled datacube (only if ``output_fits`` is None).
    header : astropy.io.fits.Header
        Updated FITS header (only if ``output_fits`` is None).

    Raises
    ------
    ValueError
        If required spectral information is missing.
    """

    #load the input data and velocity field
    data_dict=load_data(input_fits)
    vfield_dict=load_data(velocity_field)

    #store the relevant info
    data=data_dict['data']
    header=data_dict['header']
    vfield=vfield_dict['data']

    #determine spectral resolution
    if spectralres is None:
        spectralres=data_dict.get('spectralres') or header.get('CDELT3')
        if spectralres is None:
            raise ValueError("No spectral resolution found.")

    #determine starting velocity v0
    if v0 is None:
        if 'CRVAL3' in header and 'CRPIX3' in header:
            v0=header['CRVAL3']+(header['CRPIX3']-1)*spectralres
        else:
            raise ValueError("No starting velocity found.")
    
    #create the velocity array
    n_channel=data.shape[0]
    v=v0+np.arange(n_channel)*spectralres
    mid_channel=n_channel//2

    #initialize the shuffled data array
    shuffled=np.full_like(data, np.nan)

    #handle fill value for NaNs in velocity field
    if fill_value != np.nan:
        vfield[np.isnan(vfield)]=fill_value

    #find valid pixels in the velocity field
    valid_pixels=np.argwhere(~np.isnan(vfield))

    #loop over each pixel and shuffle the spectrum
    for y,x in tqdm(valid_pixels):
        spectrum=data[:,y,x]
        if np.all(np.isnan(spectrum)):
            continue
        #find the target channel,i.e., the channel whose velocity is closest to the velocity field value
        target_v=vfield[y,x]
        target_chan=np.argmin(np.abs(v-target_v))

        #compute the shift needed to align the target channel to the mid channel
        shift=mid_channel-target_chan

        #apply the shift using np.roll to preserve the noise properties
        shuffled[:,y,x]=np.roll(spectrum,shift)

    #update the header for the shuffled cube if a new FITS file is to be written
    if output_fits:
        header['CRPIX3']=mid_channel+1
        header['CRVAL3']=0.0
        fits.PrimaryHDU(shuffled,header=header).writeto(output_fits,overwrite=True)
    else:
        return shuffled,header

############################################################################################################

def stack(spectra,
          *,
          weighting='none',
          statistics='mad',
          fluxrange='negative',
          output=None,
          run_finder=False,
          plot=False,
          **kwargs):
    """
    Stack a set of 1D spectra using optional noise-based weighting.

    This function combines multiple spectra into a single stacked spectrum,
    optionally weighting each input spectrum by the inverse square of its RMS.
    During stacking, the RMS of the combined spectrum and its expected evolution
    are tracked. Optionally, a line-finding algorithm can be applied to the final
    stacked spectrum, and the result can be plotted and/or saved to disk.

    Parameters
    ----------
    spectra : sequence of array_like
        List or iterable of 1D spectra to be stacked. All spectra must have the
        same length. Spectra containing only NaNs are ignored.
    weighting : {'none', 'rms'}, optional
        Weighting scheme for stacking. If 'none', all spectra are weighted
        equally. If 'rms', each spectrum is weighted by the inverse square of
        its RMS. Default is 'none'.
    statistics : str, optional
        Statistic used to estimate the RMS (passed to the RMS function).
        Default is 'mad'.
    fluxrange : str, optional
        Flux range used for RMS estimation (passed to the RMS function).
        Default is 'negative'.
    output : str or None, optional
        Path to save a plot of the stacked spectrum. If None, no file is saved.
        Default is None.
    run_finder : bool, optional
        If True, run a line-finding algorithm on the final stacked spectrum.
        Default is False.
    plot : bool, optional
        If True, display a plot of the stacked spectrum. Default is False.
    **kwargs
        Additional keyword arguments passed to the RMS function, line finder,
        and plotting utilities. See RMS, lines_finder and _plot_spectrum.

    Returns
    -------
    stack : ndarray
        The final stacked spectrum.
    stack_rms : ndarray
        RMS of the stacked spectrum after each stacking step.
    exp : ndarray
        Expected RMS evolution assuming ideal noise reduction
        (∝ 1/sqrt(N)).
    """
    
    #initialize the number of stacked spectra
    N=1

    #initialize the weights
    weights=0

    #initialize the stacked spectrum
    stack=np.zeros(spectra[0].shape[0])

    #initialize the stacked rms list
    stack_rms=[]

    #initialize the expected rms list
    exp=[]
      
    #stack the spectra
    for i in tqdm(range(len(spectra)),desc='Spectra stacked',total=len(spectra)):
        #extract the spectrum 
        spectrum=spectra[i]

        #if the spectrum to be stacked is empty
        if np.all(np.isnan(spectrum)):
            continue

        #if no weighting
        if weighting == 'none':
            #the weight is 1
            w=1
        #if noise weighting
        elif weighting == 'rms': 
            #the weight is 1/rms**2 of the input spectrum
            w=1/RMS(spectrum,mode=statistics,fluxrange=fluxrange)**2
        
        #stack the input spectrum
        stack=np.nansum((stack,spectrum*w),axis=0)

        #stack the weight
        weights+=w

        #calculate the rms of the stacked spectrum
        stack_rms.append(RMS(stack/weights,mode=statistics,fluxrange=fluxrange)) 

        #calculate the expected value for the rms       
        exp.append(stack_rms[0]/np.sqrt(N)) 
        
        #increase the number of stacked spectra
        N+=1

    #apply the weights
    stack/=weights
    
    #run the line finder
    if run_finder:
        finder_output=kwargs.get('finder_output',None)
        mask=lines_finder(stack,spectrum_rms=stack_rms[0],output=finder_output,**kwargs)[0]
    else:
        mask=None

    #save stacked spectrum
    if output:
        #store the kwargs
        dpi=kwargs.get('dpi',300)

        #do the plot
        fig,ax=_plot_spectrum(stack,mask=mask,**kwargs)
        
        #save the figure
        fig.savefig(output,dpi=dpi,bbox_inches='tight') 

    #plot the stacked spectrum
    if plot:
        #do the plot
        fig,ax=_plot_spectrum(stack,mask=mask,**kwargs)
        plt.show(fig)
        
    return stack,np.array(stack_rms),np.array(exp)

##########################################################################################################

def velfi(velocity_field,
          *,
          radii,pa,inc,vrad,vsys,vrot,
          pixelres=None,center=None,
          extend_only=True,
          plot=True,
          cmap='mom1_cosmos',output=None,
          **kwargs):
    """
    Extend or model a 2D velocity field using a tilted-ring formalism.

    This function computes a model velocity field based on concentric rings
    defined by radius, position angle, inclination, rotation velocity,
    radial velocity, and systemic velocity. The model can be used either to
    extend an existing velocity field into masked regions or to generate a
    full modeled field.

    Parameters
    ----------
    velocity_field : str or array_like
        Input velocity field or path to a data file readable by `load_data`.
    radii : array_like
        Radii of the tilted rings in the same physical units as the input data.
    pa : float or array_like
        Position angle(s) of the rings in degrees.
    inc : float or array_like
        Inclination angle(s) of the rings in degrees.
    vrad : float or array_like
        Radial velocity component(s) of the rings.
    vsys : float or array_like
        Systemic velocity component(s) of the rings.
    vrot : float or array_like
        Rotational velocity component(s) of the rings.
    pixelres : float, optional
        Pixel resolution used to convert physical units to pixels. If None,
        it is read from the input metadata.
    center : tuple of int, optional
        Rotational center (x, y) in pixel coordinates. If None, the image
        center is used.
    extend_only : bool, optional
        If True, only fill NaN pixels in the input velocity field. If False,
        generate a full modeled field. Default is True.
    plot : bool, optional
        If True, plot the original and extended velocity fields. Default is True.
    cmap : str, optional
        Colormap used for plotting. Default is 'mom1_cosmos'.
    output : str or None, optional
        Path to save the resulting velocity field as a FITS file. If None,
        no file is written.
    **kwargs
        Additional keyword arguments passed to make_figure.

    Returns
    -------
    result : ndarray
        The extended or modeled velocity field.
    fig, axes : matplotlib objects, optional
        Returned only if plotting is enabled.
    """

    #load the input velocity field
    data_dict=load_data(velocity_field)
    data=data_dict['data']

    #store the relevant metadata
    if pixelres is None: pixelres=data_dict['pixelres']
    if center is None: center=(data.shape[1]//2,data.shape[0]//2)
    if np.asarray(center).shape !=(2,):
        raise ValueError('Provide the rotational center as a tuple (x,y) in pixel units')
    y0,x0=center

    #convert the radii into array
    radii=np.asarray(radii)

    #get the number of radii
    nradii=len(radii) 
    
    #extend the relevant quantities
    pa=_extend(pa,nradii)
    inc=_extend(inc,nradii)
    vrad=_extend(vrad,nradii)
    vsys=_extend(vsys,nradii)
    vrot=_extend(vrot,nradii)

    #convert into radians
    pa=np.radians(pa)
    inc=np.radians(inc)

    #convert from data units to pixel units
    radii/=np.abs(pixelres)

    #initialize the result            
    if extend_only:
        result=data.copy()
    else:
        result=data.copy()*np.nan
     
    #define the spatial range boundaries
    rmax=np.nanmax(radii)
    xmin=int(np.around(x0-rmax,0))
    xmax=int(np.around(x0+rmax,0))
    ymin=int(np.around(y0-rmax,0))
    ymax=int(np.around(y0+rmax,0))

    #clip the boundaries
    if xmin<0: xmin=0
    if xmax >= data.shape[1]: xmax=data.shape[1]
    if ymin<0: ymin=0
    if ymax >= data.shape[0]: ymax=data.shape[0]

    #define the spatial range
    xrange=np.arange(xmin,xmax)
    yrange=np.arange(ymin,ymax)

    #compute the velocity field extension
    with tqdm(desc='Pixel computed',total=len(xrange)*len(yrange)) as pbar:
        for i in yrange:
            for j in xrange:
                #if the value in (i,j) is empty
                if np.isnan(result[i,j]):
                    #convert the pixel coordinates w.r.t. the center
                    x=(j-x0)
                    y=(i-y0)

                    #calculate the distance from the center
                    r=np.sqrt(x**2+y**2)

                    #if the distance is less or equal the outer radius
                    if r <= radii[nradii-1]:
                        #calculate the radius of (x,y) compared to the innermost radius
                        d1=_radius(radii[0],x,y,pa[0],inc[0])

                        #calculate the radius of (x,y) compared to the outermost radius
                        d2=_radius(radii[nradii-1],x,y,pa[nradii-1],inc[nradii-1])

                        #if one of the distance is negative, means that the points resides between the innermost and outermost radius, hence, it is okay to calculate the rotation velocity at that point
                        if d1*d2 <= 0:
                            #find the radius[i] corresponding to that point
                            for k in np.arange(1,nradii+1,1):
                                #keep calculate the distance between the radius of (x,y) and the k-radius
                                d2=_radius(radii[k],x,y,pa[k],inc[k])

                                #when both distances are positive, means that we have reached the outermost radius
                                if d1*d2 > 0:
                                    #set the two distances to be equal
                                    d1=d2
                                else:
                                    break
                            #do the interpolation on adjacent radii to smooth the result

                            R=(radii[k]*d1-radii[k-1]*d2)/(d1-d2)

                            #calculate the weight of the radius k-1
                            w1=(radii[k-1]-R)/(radii[k-1]-radii[k])

                            #calculate the weight of the radius k
                            w2=1-w1

                            #calculated the weighted quantities
                            posai=w1*pa[k]+w2*pa[k-1]
                            incli=w1*inc[k]+w2*inc[k-1]
                            vroti=w1*vrot[k]+w2*vrot[k-1]
                            vradi=w1*vrad[k]+w2*vrad[k-1]
                            vsysi=w1*vsys[k]+w2*vsys[k-1]
                            if R == 0:
                                cost=1
                                sint=0
                            else:
                                cost=(-x*np.sin(posai)+y*np.cos(posai))/R
                                sint=((-x*np.cos(posai)-y*np.sin(posai))/R)/np.cos(incli)

                            #calculate the rotation velocity
                            result[i,j]=vsysi+(vroti*cost+vradi*sint)*np.sin(incli)
                            pbar.update(1)

    if plot:
        fig,axes=make_figure(ncols=2,nrows=1,wcs=WCS(data_dict['header']),colorbar=True,**kwargs)
        fig,axes[0]=plot_mom1(data,cmap=cmap,fig=fig,ax=axes[0])
        fig,axes[1]=plot_mom1(result,cmap=cmap,fig=fig,ax=axes[1])
        axes[0].set_title('Original field')
        axes[1].set_title('Extended field')
    
    if output is not None:
        fits.PrimaryHDU(result,header=data_dict['header']).writeto(output,overwrite=True)
    
    if output is not None and not plot:
        return result
    if output is not None and plot:
        return fig,axes
    if plot:
        return result,fig,axes

##########################################################################################################

def _add_colorbar(fig,ax,im,norm,ctr,
                  *,
                  label=None,
                  ctr_cmap=None,ctr_label=None,
                  mode='colormap',
                  **kwargs):
    """
    Add a horizontal colorbar above an axes with optional contour annotations.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to which the colorbar is added.
    ax : matplotlib.axes.Axes
        Axes associated with the image.
    im : matplotlib.image.AxesImage
        Image returned by `imshow`.
    norm : matplotlib.colors.Normalize
        Normalization used for the image.
    ctr : array-like or None
        Contour levels to be annotated on the colorbar.
    label : str, optional
        Colorbar label.
    ctr_cmap : str or matplotlib colormap, optional
        Colormap used to color contour markers on the colorbar.
    ctr_label : array-like of str, optional
        Custom tick labels for contour levels.
    mode : {'colormap', 'posneg'}, optional
        Mode for drawing contour markers. In 'posneg', positive and negative
        contours are drawn with different styles/colors.
    **kwargs
        Additional options used in 'posneg' mode (e.g. `ctr_pos_color`,
        `ctr_neg_color`).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axes.
    """

    #create colorbar axis
    cax=ax.inset_axes([0,1,1,0.05])

    #add the colorbar
    cb=fig.colorbar(im,cax=cax,orientation='horizontal')

    #set the ticks position
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.tick_top()

    #add the title
    if label is not None: cb.set_label(label)

    #set the countours
    if ctr is not None:
        cb.set_ticks(ctr)
        cb.set_ticklabels(np.around(ctr,2)) if ctr_label is None else cb.set_ticklabels(ctr_label)
        if mode == 'colormap' and ctr_cmap is not None:
            #assign the color of the countour levels to the correspondong tick
            for tick,tick_color in zip(ctr,plt.cm.get_cmap(ctr_cmap)(_normalize(ctr,0,1))):
                #create the line
                cb.ax.plot([norm(tick),norm(tick)],[0,1],c=tick_color,lw=1,transform=cb.ax.transAxes)
        elif mode == 'posneg':
            ctr_pos=ctr[ctr>0] #get the positive contours
            ctr_neg=ctr[ctr<0] #get the negative contours
            ctr_pos_color=kwargs.get('ctr_pos_color','black') #color for positive contours
            ctr_neg_color=kwargs.get('ctr_neg_color','gray') #color for negative contours
            #for each positive contour
            for tick in ctr_pos:
                #create the line for each contour level
                cb.ax.axvline(tick,c=ctr_pos_color,lw=1)
            #for each negative contour
            for tick in ctr_neg:
                #create the line for each contour level
                cb.ax.axvline(tick,c=ctr_neg_color,lw=1,ls='dashed')
    
    return ax

#############################################################################################

def _auto_fit_fontsize(text,
                       *,
                       width=None,height=None,
                       fig=None,ax=None,
                       min_fontsize=1):
    """
    Automatically adjust the font size of a text artist to fit within given
    width and/or height constraints in data coordinates.

    The font size is iteratively reduced until the rendered text bounding box
    satisfies the specified constraints or a minimum font size is reached.

    Parameters
    ----------
    text : matplotlib.text.Text
        Text artist whose font size is to be adjusted.
    width : float or None, optional
        Maximum allowed width in data coordinates.
    height : float or None, optional
        Maximum allowed height in data coordinates.
    fig : matplotlib.figure.Figure or None, optional
        Figure containing the text.
    ax : matplotlib.axes.Axes or None, optional
        Axes containing the text.
    min_fontsize : int, optional
        Minimum allowable font size.

    Returns
    -------
    int
        The final font size that satisfies the constraints.

    Raises
    ------
    ValueError
        If the text cannot fit within the given dimensions.
    """

    #get the figure and axes to use
    fig=fig or plt.gcf()
    ax=ax or plt.gca()

    #force a draw so that a renderer exists and text size can be measured reliably
    fig.canvas.draw()

    #obtain the renderer responsible for converting artists into pixel-based representations
    renderer=fig.canvas.get_renderer()

    #get the current font size 
    fontsize=text.get_fontsize()

    #iteratively decrease the font size it until the text fits within the constraints or the minimum font size is reached
    while fontsize >= min_fontsize:
        #measure the text bounding box in display (pixel) coordinates
        bbox=text.get_window_extent(renderer=renderer)

        #convert the bounding box from display coordinates into the axes data coordinate system
        bbox=Bbox(ax.transData.inverted().transform(bbox))

        #determine whether the text width satisfies the constraint if one is specified
        fits_width=True if width is None else bbox.width <= width

        #determine whether the text height satisfies the constraint if one is specified
        fits_height=True if height is None else bbox.height <= height

        #exit successfully once the text fits within both width and height constraints
        if fits_width and fits_height:
            return fontsize

        #decrease the font size by one point and apply it to the text object
        fontsize-=1
        text.set_fontsize(fontsize)

    #raise error if the text cannot fit within the constraints at any reasonable size
    raise ValueError("Text cannot fit within the given dimensions")

#############################################################################################

def _contours(ctr,data,
              *,
              vmax=None,norm=None):
    """
    Compute contour levels from a specification.

    Parameters
    ----------
    ctr : str or array-like
        Contour specification. Supported strings include:
        - 'even' for evenly spaced contours in normalized space
        - strings containing 'pow' to generate power-law spaced contours
        Otherwise, interpreted as explicit contour levels.
    data : array-like
        Data used to derive contour levels.
    vmax : float, optional
        Maximum value used when computing power-law contours.
    norm : matplotlib.colors.Normalize, optional
        Normalization used when `ctr='even'`.

    Returns
    -------
    ctr : numpy.ndarray
        Array of contour levels.
    """

    if isinstance(ctr,str) and 'pow' in ctr:
        #calculate the rms
        rms=RMS(data)

        #calculate the maximum sigma in the data
        max_sigma=int(np.around(vmax*np.nanmax(data/rms),0))

        #determine the power
        power=int(ctr.split('pow')[-1])

        #five evenly distributed powers between 'power' and max_sigma
        ctr=np.around(power**np.linspace(1,np.log(0.9*max_sigma)/np.log(power),5),0)
        
        #contour levels in terms of rms
        ctr*=rms
    elif isinstance(ctr,str) and ctr == 'even':
        #define five evenly spaced contours in the used normalization
        ctr=norm.inverse(np.linspace(0,1,7))[1:-1]
    else:
        #convert into an array
        ctr=np.asarray(ctr)

    return ctr

#############################################################################################################

def _covariance_to_error_ellipse(covariance_matrix,idx1,idx2):
    """
    Convert a covariance matrix into error ellipse parameters.

    This utility computes the semi-major axis, semi-minor axis, and position
    angle of the error ellipse corresponding to a 2D covariance subspace
    defined by two indices.

    Parameters
    ----------
    covariance_matrix : ndarray
        Full covariance matrix.
    idx1 : int
        Index of the first parameter.
    idx2 : int
        Index of the second parameter.

    Returns
    -------
    radius_maj : float
        Standard deviation along the major axis of the error ellipse.
    radius_min : float
        Standard deviation along the minor axis of the error ellipse.
    pa : float
        Position angle of the major axis in degrees.
    """

    #extract the xx component of the covariance matrix
    cov_xx=covariance_matrix[idx1,idx1]

    #extract the yy component of the covariance matrix
    cov_yy=covariance_matrix[idx2,idx2]

    #extract the xy component of the covariance matrix
    cov_xy=covariance_matrix[idx1,idx2]

    #calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues,eigenvectors=np.linalg.eig(covariance_matrix)

    #sort the eigenvalues and eigenvectors in descending order
    sorted_indices=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[sorted_indices]
    eigenvectors=eigenvectors[:,sorted_indices]
    
    #standard deviation along one axis
    radius_maj=np.sqrt(eigenvalues[0]) if eigenvalues[0]>eigenvalues[1] else np.sqrt(eigenvalues[1])

    #standard deviation along the other axis
    radius_min=np.sqrt(eigenvalues[1]) if eigenvalues[0]>eigenvalues[1] else np.sqrt(eigenvalues[0]) 

    #if there is no covariance between x and y, the ellipse is aligned with the axes
    if cov_xy == 0:
        pa=0.0
    else:
        #calculate the position angle (in degrees) of the major axis
        pa=np.degrees(np.arctan2(2*cov_xy,cov_xx-cov_yy)/2)

    return radius_maj,radius_min,pa

#############################################################################################################

def _extend(var,target_length):
    """
    Extend an array-like variable to a target length by repetition.

    If the input array is shorter than the target length, its last value is
    repeated until the desired length is reached.

    Parameters
    ----------
    var : array_like
        Input variable to be extended.
    target_length : int
        Desired length of the output array.

    Returns
    -------
    ndarray
        Extended array with length equal to `target_length`.
    """

    var=np.atleast_1d(var)
    if var.size < target_length:
        var=np.concatenate((var,np.repeat(var[-1],target_length-var.size)))
    return var
    
#############################################################################################################

def _extract_reliability_params(sources,
                                *,
                                sign=1.0):
    """
    Extracts logarithmic peak, summed, and mean flux parameters for sources.

    This helper function computes three reliability parameters for each source:
    peak flux, integrated flux, and mean flux, all normalized by sign and
    returned in logarithmic space.

    Parameters
    ----------
    sources : array-like of arrays
        Collection of sources, each represented as an array of channel values.
    sign : float, default=1.0
        Sign convention for the sources (+1 for positive, -1 for negative).

    Returns
    -------
    params : ndarray
        Array of shape (3, N) containing log10(peak), log10(sum), and
        log10(mean) for each of the N sources.
    """

    #initialize the parameter lists
    peak=[]
    summ=[]
    mean=[]

    #for each source
    for src in sources:
        #calculate the peak flux (the max for positive sources and the min for negative sources)
        peak.append(np.log10(sign*np.nanmax(src))) if sign > 0 \
        else peak.append(np.log10(sign*np.nanmin(src)))

        #calculate the total flux
        summ.append(np.log10(sign*np.nansum(src)))

        #calculate the mean flux
        mean.append(np.log10(sign*np.nanmean(src)))

    return np.vstack([peak,summ,mean])

#############################################################################################################

def _gaussianity(spectrum,mask,
                 *,
                 statistic,fluxrange):
    """
    Performs Gaussianity tests on the noise component of a spectrum.

    Two tests are applied: the Anderson–Darling test for normality and the
    Kanekar test based on the expected RMS scaling under smoothing.

    Parameters
    ----------
    spectrum : ndarray
        Input spectrum.
    mask : ndarray
        Source mask identifying detected channels.
    statistic : {'mad', 'std'}
        Statistic used for RMS estimation.
    fluxrange : {'negative', 'positive', 'all'}
        Flux range used for RMS estimation.

    Returns
    -------
    ander : bool
        True if the Anderson–Darling test is consistent with Gaussian noise.
    kanekar : float
        Kanekar parameter measuring RMS scaling under smoothing.
    """

    #do the Andeson-Darling test for gaussianity.
    #The test is done on the noise channels (spec[mask == 0])
    anderson_test=anderson(spectrum[mask == 0])
    
    #compute the p-value depending on the test statistic
    #The p-value gives the probability that the spectrum is gaussian.
    if anderson_test.statistic>=.6:
        p=np.exp(1.2937-5.709*anderson_test.statistic+.0186*anderson_test.statistic**2)
    elif anderson_test.statistic>=.34:
        p=np.exp(0.9177-4.279*anderson_test.statistic-1.38*anderson_test.statistic**2)
    elif anderson_test.statistic>=.2:
        p=1-np.exp(-8.318+42.796*anderson_test.statistic-59.938*anderson_test.statistic**2)
    else:
        p=1-np.exp(-13.436+101.14*anderson_test.statistic-223.73*anderson_test.statistic**2)
    
    #if p>0.05, the spectrum can be considered gaussian
    ander=p > 0.05

    #do the Kanekar test for gaussianity.
    #We expect that the rms of the spectrum is reduced by a factor of sqrt(10)~3.16
    #(see https://www.researchgate.net/publication/3635815_Effect_of_smoothing_window_length_on_RMS_EMG_amplitude_estimates).

    #make a copy of the spectrum
    dummy=spectrum.copy()

    #calculate the rms before smmothing
    rms_0=RMS(dummy,mode=statistic,fluxrange=fluxrange)

    #blank the detections as we want to test the noise
    dummy[mask != 0]=np.nan

    #define the smoothing boxcar kernel
    kernel=Box1DKernel(10)

    #convolve the spectrum
    dummy=convolve(dummy,kernel)

    #calculate the rms after smmothing
    rms_1=RMS(dummy,mode=statistic,fluxrange=fluxrange)

    #calculate the Kanekar parameter
    kanekar=rms_1*np.sqrt(10)/rms_0

    return ander,kanekar

#############################################################################################################

def _kde_density(data,centers,inverse_covariance,
                 *,
                 block_size=512):
    """
    Evaluates a multivariate Gaussian kernel density estimate in blocks.

    This function computes KDE values at specified centers using a provided
    inverse covariance matrix, processing the evaluation in blocks to limit
    memory usage.

    Parameters
    ----------
    data : ndarray
        Array of shape (N, D) containing the data points.
    centers : ndarray
        Array of shape (M, D) specifying KDE evaluation locations.
    inverse_covariance : ndarray
        Inverse covariance matrix of shape (D, D).
    block_size : int, default=512
        Number of evaluation points processed per block.

    Returns
    -------
    kde : ndarray
        KDE density values evaluated at each center.
    """

    #number of points to evaluate
    npoints=(centers.shape[0])

    #initialise the KDE density array
    kde=np.zeros(npoints)

    for start in range(0,npoints,block_size):
        #determine the ending index pf the current block
        end=min(start+block_size,npoints)

        #determine the KDE evaluation points (block centers)
        cen=centers[start:end]

        ##pairwise parameter offsets: data - centers
        diff=data[None,:,:]-cen[:,None,:]

        #compute the quadratic forms (x - mu)^T cov^-1 (x - mu)
        quad=np.einsum('ijk,kl,ijl->ij',diff,inverse_covariance,diff)
        kde[start:end]=np.exp(-0.5*quad).sum(axis=1)

    return kde

#############################################################################################################

def _norm(norm,data,vmin,vmax):
    """
    Construct a normalization object for image display.

    Parameters
    ----------
    norm : {'log', 'linear'} or None
        Normalization mode.
    data : array-like
        Data used to determine scaling limits.
    vmin : float
        Lower scaling factor (relative).
    vmax : float
        Upper scaling factor (relative).

    Returns
    -------
    norm : matplotlib.colors.Normalize
        Normalization instance.
    """

    if norm == 'log':
        norm=cl.PowerNorm(gamma=0.3,vmin=vmin*np.nanmin(data[data!=0]),vmax=vmax*np.nanmax(data))
    else:
        norm=cl.Normalize(vmin=vmin*np.nanmin(data[data!=0]),vmax=vmax*np.nanmax(data))
    return norm

#############################################################################################################

def _normalize(arr,new_min,new_max):
    """
    Linearly normalize an array to a new value range.

    Parameters
    ----------
    arr : array_like
        Input array to be normalized.
    new_min : float
        Minimum value of the normalized range.
    new_max : float
        Maximum value of the normalized range.

    Returns
    -------
    ndarray
        Normalized array scaled to the [new_min, new_max] range.
        Constant arrays are mapped to `new_min`.
    """
    arr=np.asarray(arr)
    old_min=np.nanmin(arr)
    old_max=np.nanmax(arr)

    #handle constant arrays
    if old_max == old_min:
        return np.full_like(arr,new_min)

    scale=(new_max-new_min)/(old_max-old_min)
    return (arr-old_min)*scale+new_min

##############################################################################################################

def _plot_beam(ax,pixelres,bmaj,bmin,bpa,xloc=0.9,yloc=0.1,color='black'):
    """
    Draw a synthesized beam ellipse on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the beam.
    pixelres : float
        Pixel resolution in arcseconds.
    bmaj, bmin : float
        Major and minor beam axes in arcseconds.
    bpa : float
        Beam position angle in degrees.
    xloc, yloc : float, optional
        Beam center location in axes fraction coordinates.
    color : str, optional
        Beam color.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axes.
    """

    #convert from beam units to pixels
    width_px=bmaj/pixelres
    height_px=bmin/pixelres

    #convert from pixels to axis fractions
    xpix=np.diff(ax.get_xlim())[0]
    ypix=np.diff(ax.get_ylim())[0]

    width=width_px/xpix
    height=height_px/ypix

    #define the beam patch
    beam=patch.Ellipse(
        (xloc,yloc),
        width,height,
        angle=bpa,
        color=color,
        transform=ax.transAxes)

    #add the beam
    ax.add_patch(beam)

    return ax

##############################################################################################################

def _plot_kpc_scale(ax,pixelres,asectokpc,
                    *,
                    xloc=0.05,yloc=0.95,kpc=10,color="black",y_offset=0.02):
    """
    Draw a physical scale bar in kiloparsecs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the scale bar.
    pixelres : float
        Pixel resolution in arcseconds.
    asectokpc : float
        Conversion factor from arcseconds to kiloparsecs.
    xloc, yloc : float, optional
        Anchor location in axes fraction coordinates.
    kpc : float, optional
        Physical length of the scale bar.
    color : str, optional
        Color of the scale bar and label.
    y_offset : float, optional
        Vertical offset for the scale bar line.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axes.
    """

    #convert kpc → pixels → axes fraction
    kpc_in_pixels=kpc/(asectokpc*pixelres)
    kpc_in_axes=kpc_in_pixels/np.diff(ax.get_xlim())[0]

    #draw line
    ax.hlines(yloc-y_offset,
              xloc,xloc+kpc_in_axes,
              color=color,linewidth=2,transform=ax.transAxes)

    #draw label
    ax.text(xloc+kpc_in_axes/2,
            yloc,
            f"{kpc} kpc",
            ha="center",va="bottom",color=color,transform=ax.transAxes)

    return ax

##############################################################################################################

def _plot_mom(data,
              *,
              which,
              mult_by=1,add_to=0,
              use_column_density=True, #only used for mom0
              xmin=None,xmax=None,ymin=None,ymax=None,
              wcs=None,
              bmaj=None,bmin=None,bpa=None,
              pixunits='deg',specunits='m/s',fluxunits=None,
              pixelres=None,
              fig=None,ax=None,
              fig_x_size=7, fig_y_size=7,
              cmap=None,
              vmin=0.001,vmax=0.9,norm=None,
              ctr=None,
              ctr_cmap=None,
              ctr_width=1,
              ctr_color_positive='black',ctr_color_negative='gray', #only used for mom1
              left_loc=0.05,right_loc=0.95,lower_loc=0.05,upper_loc=0.9,
              kpc=10,asectokpc=None,vsys=None,
              color="black",y_offset=0.02,
              colorbar=True,colorbar_label=None,
              dpi=300,output=None):
    """
    Plot a moment map (moment 0, 1, or 2) with contours, beam, and scale bar.

    Parameters
    ----------
    data : str or array-like
        Input data or path understood by `load_data`.
    which : {'mom0', 'mom1', 'mom2'}
        Moment type to plot.
    mult_by, add_to : float, optional
        Linear scaling applied to the data.
    use_column_density : bool, optional
        If True, convert moment-0 data to column density when possible.
    xmin, xmax, ymin, ymax : int, optional
        Pixel bounds defining a subregion to plot.
    wcs : astropy.wcs.WCS, optional
        World Coordinate System for the plot.
    bmaj, bmin, bpa : float, optional
        Beam major axis, minor axis, and position angle.
    pixunits : str, optional
        Spatial pixel units ('deg' or arcseconds).
    specunits : str, optional
        Spectral units of the data.
    fluxunits : str, optional
        Flux units for labeling.
    pixelres : float, optional
        Pixel resolution.
    fig, ax : matplotlib objects, optional
        Existing figure and axes to use.
    fig_x_size, fig_y_size : float, optional
        Figure size in inches.
    cmap : str or colormap, optional
        Colormap for the image.
    vmin, vmax : float, optional
        Relative scaling limits.
    norm : str or matplotlib.colors.Normalize, optional
        Normalization specification.
    ctr : array-like or str, optional
        Contour specification.
    ctr_cmap : str or colormap, optional
        Colormap for contours.
    ctr_width : float, optional
        Contour line width.
    ctr_color_positive, ctr_color_negative : str, optional
        Colors for positive/negative contours (moment 1).
    left_loc, right_loc, lower_loc, upper_loc : float, optional
        Axes-fraction locations for annotations.
    kpc : float, optional
        Length of the physical scale bar.
    asectokpc : float, optional
        Arcsecond-to-kiloparsec conversion.
    vsys : float, optional
        Systemic velocity (moment 1).
    color : str, optional
        Color for annotations.
    y_offset : float, optional
        Vertical offset for scale bar.
    colorbar : bool, optional
        Whether to draw a colorbar.
    colorbar_label : str, optional
        Colorbar label.
    dpi : int, optional
        Output resolution.
    output : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created or updated figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    
    #load the data
    data_dict=load_data(data)
    data=data_dict['data']

    #load the metadata
    if wcs is None: wcs=data_dict['wcs']
    if bmaj is None: bmaj=data_dict['bmaj']
    if bmin is None: bmin=data_dict['bmin']
    if bpa  is None: bpa=data_dict['bpa']
    if pixelres is None: pixelres=data_dict['pixelres']
    if pixunits is None: pixunits=data_dict['pixunits']
    if specunits is None: specunits=data_dict['specunits']
    if fluxunits is None: fluxunits=data_dict['fluxunits']

    #scale the data
    data=data*mult_by+add_to

    #convert the spatial units
    pixres_arcsec=pixelres
    bmaj_arcsec=bmaj
    bmin_arcsec=bmin

    if pixunits == "deg":
        if pixres_arcsec is not None: pixres_arcsec*=3600
        if bmaj_arcsec   is not None: bmaj_arcsec  *=3600
        if bmin_arcsec   is not None: bmin_arcsec  *=3600

    
    #moment-specific pre-processing
    if which == "mom0":
        #convert to column density
        if use_column_density and all(x is not None for x in (bmaj_arcsec,bmin_arcsec)):
            data[data<0]=0
    if which == "mom1":
        #subtract the systemic velocity
        if vsys is None: vsys=np.nanmean(data)
        data-=vsys

    #convert the spectral units
    if specunits == "m/s":
        data/=1000
        if which != "mom0":
            fluxunits=r'km s$^{-1}$'

    #define the extraction region
    if xmin is None: xmin=0
    if xmax is None: xmax=data.shape[1]
    if ymin is None: ymin=0
    if ymax is None: ymax=data.shape[0]

    #extract the subregion
    data=data[ymin:ymax,xmin:xmax]
    if wcs is not None:
        wcs=wcs[ymin:ymax,xmin:xmax]

    #plot defaults depending on the moment map
    if norm is None:
        norm={"mom0": "log",
              "mom1": "",
              "mom2": "linear"}[which]
    if cmap is None:
        cmap={"mom0": "Greys",
              "mom1": "mom1_cosmos",
              "mom2": "mom2"}[which]
    if ctr is None:
        ctr={"mom0": "even",
              "mom1": None,
              "mom2": "even"}[which]
    if ctr_cmap is None:
        ctr_cmap={"mom0":"hot",
                  "mom1": "",
                  "mom2": "mom2_ctr"}[which]
    if colorbar_label is None:
        colorbar_label={"mom0": f"Moment 0 [{fluxunits}]",
                        "mom1": f"LoS velocity w.r.t systemic [{fluxunits}]",
                        "mom2": f"Moment 2 [{fluxunits}]"}[which]

    #define the normalization
    if which == "mom1":
        norm_obj= cl.CenteredNorm()
    else:
        norm_obj=_norm(norm,data,vmin,vmax)

    #define the contours
    if which == "mom1":
        if isinstance(ctr,(list,np.ndarray)):
            ctr=np.asarray(ctr)
            ctr_pos=ctr[ctr > 0]
            ctr_neg=ctr[ctr < 0]
        else:
            ctr_res=np.floor(0.85*np.nanmax(np.abs(data))/15)*5
            ctr_pos=np.arange(0,0.85*np.nanmax(np.abs(data)),ctr_res)[1:].astype(int)
            ctr_neg=-ctr_pos[::-1]
    else:
        if isinstance(ctr,(list,np.ndarray)):
            ctr=np.asarray(ctr)
            if which == "mom0" and use_column_density:
                ctr=convert_from_HI(ctr*1e19,
                                    beamarea=1.13*bmaj_arcsec*bmin_arcsec,
                                    from_moment_map=True)
        else:
            ctr=_contours(ctr,data,vmax=vmax,norm=norm_obj)

    #create the figure and axes
    if fig is None:
        fig,ax=make_figure(wcs=wcs,
                           fig_x_size=fig_x_size,
                           fig_y_size=fig_y_size,
                           nrows=1,
                           ncols=1,
                           colorbar=colorbar)
    if ax is None:
        ax=fig.add_subplot(projection=wcs)

    #do the plot
    im=ax.imshow(data,
                 cmap=cmap,
                 norm=norm_obj,
                 origin='lower' if ax.name == 'rectilinear' else None)

    #draw the contours
    if which == "mom1":
        ax.contour(data,levels=ctr_pos,colors=ctr_color_positive,
                   linewidths=ctr_width,linestyles='solid')
        ax.contour(data,levels=ctr_neg,colors=ctr_color_negative,
                   linewidths=ctr_width,linestyles='dashed')
    else:
        ax.contour(data,levels=ctr,cmap=ctr_cmap,
                   linewidths=ctr_width,linestyles='solid')

    #draw the beam
    if all(x is not None for x in (pixres_arcsec,bmaj_arcsec,bmin_arcsec,bpa)):
        ax=_plot_beam(ax,pixres_arcsec,bmaj_arcsec,bmin_arcsec,bpa,
                      xloc=right_loc,yloc=lower_loc,color=color)

    #draw the physical scale line
    if all(x is not None for x in (pixres_arcsec,asectokpc)):
        ax=_plot_kpc_scale(ax,pixres_arcsec,asectokpc,
                           xloc=left_loc, yloc=upper_loc,
                           kpc=kpc,color=color,y_offset=y_offset)

    #add the colorbar
    if colorbar:
        if which == "mom0":
            if use_column_density and all(x is not None for x in (bmaj_arcsec,bmin_arcsec)):
                #convert the contour levels to column density in units of 10^19 cm^-2
                ctr_label=np.around(convert_to_HI(ctr,
                                                  beamarea=1.13*bmaj_arcsec*bmin_arcsec,
                                                  from_moment_map=True)/10.**19,2)
                colorbar_label=r'HI column density [$10^{19}$ cm$^{-2}$]'
            else:
                ctr_label=None
        else:
            ctr_label=None
        if which == "mom1":
            ax=_add_colorbar(fig,ax,im,norm_obj,
                             ctr=np.concatenate((ctr_neg, [0], ctr_pos)),
                             mode='posneg',
                             ctr_pos_color=ctr_color_positive,
                             ctr_neg_color=ctr_color_negative,
                             label=colorbar_label)
        else:
            ax=_add_colorbar(fig,ax,im,norm_obj,ctr,
                             ctr_cmap=ctr_cmap,
                             ctr_label=ctr_label,
                             label=colorbar_label)

    #save the figure
    if output is not None:
        fig.savefig(output,bbox_inches='tight',dpi=dpi)

    return fig,ax

##############################################################################################################

def _plot_reliability(pos_params,
                      neg_params,
                      covariance,
                      reliability,
                      threshold,
                      snr_min,
                      output,
                      *,
                      points_size=0.25,
                      positive_color="royalblue",
                      negative_color="hotpink",
                      reliable_color="black",
                      ellipses_color="black",
                      ellipses_width=1.5,
                      xaxis_scale=0.05,
                      yaxis_scale=0.05,
                      dpi=300,
                      **kwargs):
    """
    Produces diagnostic plots for reliability parameter space.

    The function visualizes positive and negative detections in multiple
    parameter projections, overlays covariance ellipses derived from
    negative detections, and highlights reliable sources.

    Parameters
    ----------
    pos_params : ndarray
        Reliability parameters of positive sources.
    neg_params : ndarray
        Reliability parameters of negative sources.
    covariance : ndarray
        Covariance matrix derived from negative detections.
    reliability : ndarray
        Reliability values of positive sources.
    threshold : float
        Reliability threshold.
    snr_min : float
        Minimum S/N used in the reliability analysis.
    output : str
        Output filename for the plot.
    **kwargs
        Additional plotting configuration options. See make_figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes.Axes
        The last axis used in the plot.
    """

    #define the parameter spaces (0=peak, 1=sum, 2=mean)
    param_space=[#xaxis param  yaxis param    plot title       xlabel             ylabel
                 (     0,           1,      "Peak vs sum" , "Log(peak/rms)", "Log(sum/rms)"),
                 (     0,           2,      "Peak vs mean", "Log(peak/rms)", "Log(mean/rms)"),
                 (     1,           2,      "Sum vs mean" , "Log(sum/rms)" , "Log(mean/rms)")]
    
    #define the scatter point area
    points_area=(points_size*plt.rcParams["lines.markersize"])**2

    #create the figure and the axes
    fig,axes=make_figure(ncols=2,nrows=2,hspace=0.1,wspace=0.1,**kwargs)

    #hide unused axes
    for i in range(3,len(axes)):
        axes[i].set_visible(False)

    for ax,(i,j,title,xlabel,ylabel) in zip(axes,param_space):
        #compute the covariance ellipse
        rmaj,rmin,pa=_covariance_to_error_ellipse(covariance,i,j)

        #determine the distribution center
        center=[np.nanmean(neg_params[i]),np.nanmean(neg_params[j])]

        #add the error ellipses
        for k,ls in zip([1,2,3],["solid","dashed","-."]):
            ellipse=patch.Ellipse(center,
                                  width=k*rmaj,
                                  height=k*rmin,
                                  angle=pa,
                                  fill=False,
                                  ec=ellipses_color,
                                  ls=ls,
                                  lw=ellipses_width,
                                  zorder=100)
            ax.add_patch(ellipse)

        #scatter plot of the negative sources
        ax.scatter(neg_params[i],neg_params[j],
                   s=points_area,c=negative_color,label="Negative")
        ax.scatter(pos_params[i], pos_params[j],
                   s=points_area,c=positive_color,label="Positive")

        #set the title and the labels
        ax.set_title(title,pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        #scale the axis for better visualisation
        xmin,xmax=ax.get_xlim()
        ymin,ymax=ax.get_ylim()
        dx=xaxis_scale*(xmax-xmin)
        dy=yaxis_scale*(ymax-ymin)
        ax.set_xlim(xmin-dx,xmax+dx)
        ax.set_ylim(ymin-dy,ymax+dy)

        #use a square aspect
        ax.set_aspect((xmax-xmin)/(ymax-ymin))

        #draw the S/N line in the sum vs mean panel
        if (i,j) == (1,2):
            x=np.array([pos_params[i].min(),pos_params[i].max()])
            y=2*np.log10(snr_min)-x
            ax.plot(x,y,c="gray",ls="--",lw=1,zorder=0)
            ax.text(0.95,0.05,f"SNR min = {snr_min}",
                    ha="right",transform=ax.transAxes)

    #highlight reliable sources
    reliable= reliability >= threshold

    axes[0].scatter(pos_params[0][reliable],pos_params[1][reliable],
                    s=8*points_area,c=reliable_color,label="Reliable")
    axes[1].scatter(pos_params[0][reliable],pos_params[2][reliable],
                    s=8*points_area,c=reliable_color,label="Reliable")
    axes[2].scatter(pos_params[1][reliable],pos_params[2][reliable],
                    s=8*points_area,c=reliable_color,label="Reliable")

    #add the legend
    for ax in axes:
        ax.legend(loc="best",ncol=2)

    #save the figure
    if output:
        fig.savefig(output,dpi=dpi,bbox_inches="tight")
    return fig,ax

##########################################################################################################

def _plot_skellam(P,N,kernel,output,
                  *,
                  dpi=300,
                  **kwargs):
    """
    Plots the cumulative distribution of the Skellam parameter.

    The Skellam parameter is computed from KDE estimates of positive and
    negative detections and compared against a standard Gaussian cumulative
    distribution.

    Parameters
    ----------
    P : ndarray
        KDE values for positive detections.
    N : ndarray
        KDE values for negative detections.
    kernel : float
        Kernel scaling factor used in the KDE.
    output : str
        Output filename for the plot.
    **kwargs
        Additional plotting options. See make_figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes.Axes
        The axis containing the plot.
    """
        
    #compute the skellam paramenter (P-N)/sqrt(P+N)
    skellam=(P-N)/np.sqrt(P+N)

    #sort the array for the plot.
    #This step is necessary as we are going to normalize it
    skellam=np.sort(skellam)

    #normalize to sigma = 1.
    #This is necessary to ensure that the standard deviation of the Skellam parameter values is 1,
    #such that its distribution can be readily compared to a standard Gaussian
    skellam=(skellam-np.nanmedian(skellam))/np.nanstd(skellam)
    skellam+=np.nanmedian(skellam)

    #compute the cumulative distribution
    cum=np.arange(len(skellam))/(len(skellam)-1)

    #the Gaussian cumulative is given by the erf function
    x=np.linspace(-4,4,401)
    y=0.5-0.5*erf(-x/np.sqrt(2.0))

    #create the figure
    fig,ax=make_figure(ncols=1,nrows=1,**kwargs)

    #plot the skellam distribution
    ax.plot(skellam,cum,c="b",lw=1,
            label=fr"Data: $\mu={np.nanmedian(skellam):.2f}$" "\n"
                  fr"kernel = {kernel}")
    
    #plot the expected gaussian distribution
    ax.plot(x,y,c="r",lw=1,
            label=r"Gaussian: $\mu=0$, $\sigma=1$")
    ax.axvline(np.nanmedian(skellam),c="b",ls="dashed",lw=1)
    ax.axvline(0,c="r",ls="dotted",lw=1)

    #set limits and labels
    ax.set_xlim([-4,4])
    ax.set_ylim([0,1])
    ax.set_xlabel(r"Skellam parameter normalized to $\sigma=1$")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="upper left")

    #save the figure
    if output:
        fig.savefig(output,dpi=dpi,bbox_inches="tight")

    return fig,ax

############################################################################################################

def _plot_spectrum(spectrum,
                   *,
                   v=None,xunits=r'km s$^{-1}$',
                   yunits=r'Jy beam$^{-1}$',
                   color='blue',linewidth=1,
                   mask=None,
                   masked_color='blue',masked_linewidth=1,
                   figx=6,figy=3,
                   **kwargs):
    """
    Plot a 1D spectrum with optional masking.

    This function creates a publication-style plot of a spectrum, optionally
    highlighting masked and unmasked channels differently.

    Parameters
    ----------
    spectrum : array_like
        1D array representing the spectrum.
    v : array_like or None, optional
        X-axis values (e.g., velocity). If None, channel numbers are used.
    xunits : str, optional
        Units for the x-axis label. Default is 'km s$^{-1}$'.
    yunits : str, optional
        Units for the y-axis label. Default is 'Jy beam$^{-1}$'.
    color : str, optional
        Line color for unmasked data. Default is 'blue'.
    linewidth : float, optional
        Line width for unmasked data. Default is 1.
    mask : array_like or None, optional
        Binary mask indicating valid (non-zero) channels.
    masked_color : str, optional
        Line color for masked data. Default is 'blue'.
    masked_linewidth : float, optional
        Line width for masked data. Default is 1.
    figx, figy : float, optional
        Figure size in inches.
    **kwargs
        Additional keyword arguments passed to make_figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """

    figx=kwargs.get('figx',8)
    figx=kwargs.get('fig_x_size',8)
    figy=kwargs.get('figy',4)
    figy=kwargs.get('fig_y_size',4)
    ylabel=kwargs.get('ylabel',f'Flux [{yunits}]')

    #create the figure
    fig,ax=make_figure(ncols=1,nrows=1,figx=figx,figy=figy,
                       ylabel=ylabel,**kwargs)

    #define the x-axes
    if v is None:
        v=np.arange(1,len(spectrum)+1)
        ax.set_xlabel('Channels')
    else:
        ax.set_xlabel(f'Velocity [{xunits}]')

    #if a mask is not given
    if mask is None:
        ax.plot(v,spectrum,color=color,lw=linewidth)
    #if a mask is given
    else:
        #blank the noisy channels
        mask=mask.astype(np.float32) #convert to float to assign nan values
        mask[mask==0]=np.nan

        #ensure mask is binary
        mask[mask>0]=1

        #plot the spectrum
        ax.plot(v,spectrum,color=masked_color,lw=masked_linewidth,ls='dotted',label='Noise')

        #highlight the unmasked channels
        ax.plot(v,spectrum*mask,color=color,lw=linewidth,label='Source')

        #add the legend
        ax.legend(frameon=False)

    #save the axes limits to prevent the ancillary curves to affet them
    xlim=(np.nanmin(v),np.nanmax(v))
    ylim=ax.get_ylim()

    #draw the 0-flux line and 0-velocity
    ax.axhline(y=0,c='black',ls='dashed',lw=1)
    ax.axvline(x=0,c='black',ls='dotted',lw=1)

    #restore the axes limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return fig,ax

############################################################################################################

def _radius(radius,x,y,pa,inc):
    """
    Compute the signed radial offset of a point from an elliptical ring.

    The ellipse is defined by a given radius, position angle, and inclination.
    A positive value indicates the point lies inside the ellipse, while a
    negative value indicates it lies outside.

    Parameters
    ----------
    radius : float
        Radius of the elliptical ring.
    x, y : float
        Cartesian coordinates of the point relative to the center.
    pa : float
        Position angle of the ellipse in radians.
    inc : float
        Inclination angle of the ellipse in radians.

    Returns
    -------
    float
        Signed difference between the ring radius and the elliptical radius
        of the point.
    """
    
    #x-radius is the same of a circle
    xr=(-x*np.sin(pa)+y*np.cos(pa))

    #y-radius is the same of a circle but reduced by the inclination angle
    yr=(-x*np.cos(pa)-y*np.sin(pa))/np.cos(inc)
    
    #return the distance between the radius[i] and the elliptical radius of the point
    return(radius-np.sqrt(xr**2+yr**2))

##############################################################################################################

def _reliability(pos_sources,neg_sources,snr_min,
                 *,
                 kernel,
                 eps,
                 threshold,
                 plot,
                 skellam,
                 output,
                 BLOCK=512, #tune this (256–2048 depending on RAM)
                 **kwargs):
    """
    Computes the reliability of detected sources using KDE statistics.

    Reliability is estimated by comparing the density of positive detections
    to that of negative detections in a multi-parameter space, following the
    methodology adopted in SoFiA-2.

    Parameters
    ----------
    pos_sources : array-like
        Positive detected sources, normalized by RMS.
    neg_sources : array-like
        Negative detected sources, normalized by RMS.
    snr_min : float
        Minimum signal-to-noise ratio required for reliability evaluation.
    kernel : float
        Scaling factor applied to the covariance matrix.
    eps : float
        Regularization term for covariance matrix inversion.
    threshold : float
        Reliability threshold.
    plot : bool
        If True, generate reliability diagnostic plots.
    skellam : bool
        If True, generate a Skellam distribution plot.
    output : str
        Output filename prefix.
    BLOCK : int, default=512
        Block size used for KDE evaluation.
    **kwargs
        Additional plotting options. See _plot_reliability and _plot_skellam.

    Returns
    -------
    reliability : ndarray
        Reliability values for each positive source.
    """

    #get the plot format
    plot_format=kwargs.get('plot_format','pdf')

    #sanity chekcs
    if len(pos_sources) == 0:
        return np.array([])
    if len(neg_sources) < 2:
        raise ValueError("At least two negative detections are required "
                         "to estimate a covariance matrix.")
    
    #compute the S/N of the positive sources.
    #From the SoFiA-2 source code, the S/N is calculated as
    #the summed flux (normalized by the rms) divided by the square root of the number of channels contributing to a source.
    #Here, pos_sources contains already the flux/rms value in each channel
    snr=np.array([np.nansum(source)/np.sqrt(len(source)) for source in pos_sources])

    #determine which sources have enough S/N
    valid=snr > snr_min

    #calculate the peak, sum and mean flux for each positive and negative source.
    #The outcome is a (3,N) array where N is the number of sources and
    #each row contains their peak (1), sum (2) and mean (3) flux
    pos_params=_extract_reliability_params(pos_sources,sign=1.0)
    neg_params=_extract_reliability_params(neg_sources,sign=-1.0)

    #compute the covariance matrix for the negative detections.
    #rowvar=True indicates that each row represents a variable, each column an observations
    #bias=True is needed to match SoFiA-2 determination of the covariance matrix where the normalization is N and not (N-1)
    cov=np.cov(neg_params,rowvar=True,bias=True)

    #scale the covariance matrix
    cov*=kernel**2

    #add diagonal regularization for numerical stability
    cov+=np.eye(cov.shape[0])*eps

    #invert the covariance matrix
    try:
        cov_inv=np.linalg.inv(cov)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Covariance matrix inversion failed. "
                           "Check the distribution of negative detections.") from exc

    #transpose the reliability parameters for the KDE computatiom
    pos_params_T=pos_params.T.astype(np.float32) #shape (Npos,3). Use float 32 to halve memory usage
    neg_params_T=neg_params.T.astype(np.float32) #shape (Nneg,3). Use float 32 to halve memory usage 

    #compute the gaussian KDE of the positive sources
    P=_kde_density(pos_params_T,pos_params_T,cov_inv,block_size=BLOCK)
    N=_kde_density(neg_params_T,pos_params_T,cov_inv,block_size=BLOCK)

    #initialize the reliability array
    reliability=np.zeros(len(pos_sources))

    #valid sources have enough S/N (valid) but also P > N
    mask=valid & (P > N)

    #compute the reliability
    reliability[mask]=(P[mask]-N[mask])/P[mask]

    #plot the parameter space
    if plot:
        outname=output+'_rel.'+plot_format

        #do the reliability plot
        _plot_reliability(pos_params,neg_params,cov,
                          reliability,threshold,snr_min,
                          outname,**kwargs)

    #plot the skellam distribution
    if skellam:

        #we need to recompute the KDE, but now referring to the negative sources
        P=_kde_density(pos_params_T,neg_params_T,cov_inv,block_size=BLOCK)
        N=_kde_density(neg_params_T,neg_params_T,cov_inv,block_size=BLOCK)
      
        #do the skellam plot
        outname=output+'_skellam.'+plot_format  
        _plot_skellam(P,N,
                      kernel,
                      outname,
                      **kwargs)

    return reliability

##############################################################################################################

def _source_finding(spectrum,rms,
                    *,
                    smooth_kernel,
                    finder_threshold,
                    finder_replace,
                    finder_fluxrange,
                    finder_statistic):
    """
    Identifies candidate source channels using thresholding and smoothing.

    The function applies sigma-clipping to the input spectrum and optionally
    repeats the detection after smoothing with one or more kernels.

    Parameters
    ----------
    spectrum : ndarray
        Input spectrum.
    rms : float
        Noise estimate of the spectrum.
    smooth_kernel : array-like
        Smoothing kernel sizes.
    finder_threshold : float
        Detection threshold in units of RMS.
    finder_replace : float
        Replacement value used to mask detections before smoothing.
    finder_fluxrange : {'negative', 'positive', 'all'}
        Flux range used for RMS estimation.
    finder_statistic : {'mad', 'std'}
        Statistic used for RMS estimation.

    Returns
    -------
    mask : ndarray
        Integer mask identifying detected channels.
    """

    #initialize the mask as zeros
    mask=np.zeros(spectrum.shape,dtype=int)

    #set to 1 the channels above the sigma threshold
    mask[np.abs(spectrum) > finder_threshold*rms]=1

    #set the smoothed sigma to the one of the spectrum
    smooth_sigma=np.array(rms)

    #if smoothing kernels are applied
    if smooth_kernel.size > 0:
        #for each smoothing kernel
        for ker in smooth_kernel:
            #smooth only when input kernel is > 1
            if ker > 1:
                #create a copy of the original spectrum
                spec_tmp=spectrum.copy()
                
                #replace the already detected channels to avoid source smearing
                #find the detected channels
                detected_channels=mask > 0

                #replace while retaining the sign
                spec_tmp[detected_channels]=finder_replace*rms*np.sign(spec_tmp[detected_channels])
                
                #define the boxcar kernel
                kernel=Box1DKernel(ker)

                #smooth the spectrum
                smooth_spectrum=convolve(spec_tmp,kernel)

                #calculate the new rms
                smooth_sigma=RMS(smooth_spectrum,mode=finder_statistic,fluxrange=finder_fluxrange)

                #create a tempurary mask
                mask_tmp=np.zeros(smooth_spectrum.shape,dtype=int)

                #apply the sigma clipping
                mask_tmp[np.abs(smooth_spectrum) > finder_threshold*smooth_sigma]=1

                #add to the mask
                mask=np.add(mask,mask_tmp)

    return mask

##############################################################################################################

def _source_linker(data,mask,kernel,min_size):
    """
    Links adjacent detected channels into contiguous sources.

    Detected channels are grouped into sources based on proximity, minimum
    size, and total flux sign, assigning unique positive or negative IDs.

    Parameters
    ----------
    data : ndarray
        Input spectrum.
    mask : ndarray
        Detection mask from source finding.
    kernel : int
        Maximum allowed gap between detections.
    min_size : int
        Minimum number of channels for a valid source.

    Returns
    -------
    linked_mask : ndarray
        Mask with unique integer source IDs.
    """

    #set the number of channels
    n_channels=mask.size

    #initialized the mask
    linked_mask=np.zeros(n_channels,dtype=int)

    #initialize the source indeces
    pos_idx=1
    neg_idx=-1

    #initialize the channel counter
    i=0

    #scan along the channels
    while i < n_channels:
        #skip non-detected channels and keep running
        if mask[i] == 0:
            i+=1
            continue
        
        #store the starting channel
        starting_channel=i
        
        #initialize the zeros counter
        zeros_found=0

        #extend the source until kernel-consecutive zeros are found
        while i < n_channels:
            #if channel i is non-detected
            if mask[i] == 0:
                #increase the zeros counter
                zeros_found+=1

                #if the number of consecutive zeros exceed the linker kernel
                if zeros_found >= kernel:
                    break

            #if channel i is detected
            else:
                #reset the zeros counter
                zeros_found=0

            #increase the channel counter
            i+=1

        #the source ends at the first zero found. +1 accounts for the numpy half-open slicing semantics 
        ending_channel=i-zeros_found+1
        
        #extract the source channels
        source_mask=mask[starting_channel:ending_channel] != 0
        
        #indices of the channels relative to the starting channel of the source
        detected_indices=np.where(source_mask)[0]

        #determine the source size
        source_size=detected_indices.size

        #if of covering enough channels
        if source_size >= min_size:
            #calculate the total flux
            flux_sum=np.nansum(data[starting_channel:ending_channel][detected_indices])

            if flux_sum > 0:
                linked_mask[starting_channel:ending_channel][source_mask]=pos_idx
                pos_idx += 1
            elif flux_sum < 0:
                linked_mask[starting_channel:ending_channel][source_mask]=neg_idx
                neg_idx -= 1
            #zero-flux sources are discarded (remain zero)

        #continue scanning after the source
        i+=1

    return linked_mask 

##############################################################################################################

def _split_sources(spectrum,noise,source_ids):
    """
    Splits detected sources into positive and negative samples.

    The function extracts source channel values, normalizes them by RMS,
    and separates them based on the sign of their source ID.

    Parameters
    ----------
    spectrum : array-like
        Input spectra.
    noise : array-like
        RMS values corresponding to each spectrum.
    source_ids : array-like
        Source ID masks for each spectrum.

    Returns
    -------
    pos : ndarray
        Array of positive sources (object dtype).
    neg : ndarray
        Array of negative sources (object dtype).
    """

    #initialize the containers for results
    pos=[]
    neg=[]

    #loop once over all spectra, noises, and source-ID masks together.
    #enumerate() gives us the spectrum index `i`, which we need to keep
    #track of where each positive source came from.
    for i,(spec,rms,mask) in enumerate(zip(spectrum,noise,source_ids)):

        #identify pixels that belong to any source.
        non_zero_mask = mask != 0

        #if this spectrum has no sources at all, skip it immediately.
        if not np.any(non_zero_mask):
            continue

        #extract the channels belonging to sources
        masked_spec=spec[non_zero_mask]
        masked_ids=mask[non_zero_mask]

        #normalize the spectrum by its noise.
        #This step is needed for the reliability computation
        normalized_spec=masked_spec/rms

        # group channels by source ID.
        #`ids` is the sorted array of unique source IDs present
        # `inverse` gives, for each channel, the index of its source ID in the `ids` array
        ids,inverse=np.unique(masked_ids,return_inverse=True)

        #loop over sources and split them
        for idx,source_id in enumerate(ids):

            #extract all channels belonging to the source idx
            source_values=normalized_spec[inverse == idx]

            #separate positive and negative sources
            if source_id < 0:
                neg.append(source_values)
            else:
                pos.append(source_values)

    #convert lists to NumPy arrays.
    #dtype=object is used because sources can have different lengths.
    pos=np.array(pos,dtype=object)
    neg=np.array(neg,dtype=object)

    return pos,neg

##############################################################################################################

def _write_catalogue(spectrum,source_ids,noise,
                     *,
                     gaussianity,gaussian_for_anderson,gaussian_for_kanekar,
                     reliability,reliability_values):
    """
    Builds a source catalogue from detected spectral features.

    The catalogue contains per-source metadata including channel range,
    flux, S/N, Gaussianity test results, and reliability estimates.

    Parameters
    ----------
    spectrum : array-like
        Input spectra.
    source_ids : array-like
        Source ID masks.
    noise : array-like
        RMS values for each spectrum.
    gaussianity : bool
        Whether Gaussianity test results are included.
    gaussian_for_anderson : array-like
        Anderson–Darling test results per spectrum.
    gaussian_for_kanekar : array-like
        Kanekar test results per spectrum.
    reliability : bool
        Whether reliability information is included.
    reliability_values : array-like
        Reliability values for positive sources.

    Returns
    -------
    catalogue : pandas.DataFrame
        Source catalogue containing one row per detected source.
    """

    #to efficiently create the catalogue, we create each row as a list and append all rows to a list. 
    #Then, we convert the list of list, where list[0] is the row 0, into a dataframe
    
    #initialise the catalogue
    catalogue=[]

    #column names
    colnames=['Spectrum ID','Source ID','Channel min','Channel max',
              'Channels','Flux','Rms','SNR','Mean','Median']

    if gaussianity:
        colnames+=['Anderson test','Kanekar test']

    if reliability:
        colnames+=['Reliability','Log(Sum)','Log(Peak)','Log(Mean)']
        k=0  #reliability index offset

    #loop over spectra
    for i,(spec,src_ids) in enumerate(zip(spectrum,source_ids)):


        if np.all(src_ids == 0):
            continue
        
        #compute the mean
        spec_mean=np.nanmean(spec)

        #compute the median
        spec_median=np.nanmedian(spec)

        #store the noise
        rms=noise[i]

        #unique non-zero source IDs
        unique_sources=np.unique(src_ids[src_ids != 0])

        #for each unique source
        for j in unique_sources:
            #channels belonging to source j
            mask=(src_ids == j)
            idx=np.where(mask)[0]

            #minimum, maximum and total channel
            chanmin=idx[0]
            chanmax=idx[-1]
            nchan=idx.size

            #total flux
            flux=np.nansum(spec[mask])

            #S/N
            snr=flux/(np.sqrt(nchan)*rms)

            #build the row
            row=[i+1,int(j), #spectrum ID and source ID
                 chanmin,chanmax,nchan,
                 flux,rms,snr,
                 spec_mean,
                 spec_median]

            if gaussianity:
                row+=[gaussian_for_anderson[i],gaussian_for_kanekar[i]]

            if reliability:
                #normalise the sources by the rms
                norm_spec=spec[mask]/rms

                #if a positive source
                if j > 0:
                    #get its reliability.
                    #-1 accounts for j being always > 1
                    r=reliability_values[k+int(j)-1]

                    row+=[r,
                          np.log10(np.nansum(norm_spec)),
                          np.log10(np.nanmax(norm_spec)),
                          np.log10(np.nanmean(norm_spec))]
                #if a negative source
                else:
                    #its reliability is 0
                    row+=[0,
                          np.log10(-np.nansum(norm_spec)),
                          np.log10(-np.nanmin(norm_spec)),
                          np.log10(-np.nanmean(norm_spec))]

            #store the row
            catalogue.append(row)

        if reliability:
            #unique positive IDs in the spectrum i
            positive_sources=unique_sources[unique_sources > 0]
            #if there are positive sources
            if positive_sources.size:
                #update k so that the reliability values stay aligned across spectra
                k+=int(np.max(positive_sources))

    return pd.DataFrame(catalogue,columns=colnames)