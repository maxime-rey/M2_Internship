import numpy as np
import poppy as pop
import matplotlib.pyplot as plt
    


############################################################ Plotting
def one_plot(psf_1, shift, logscale=True, normalised=True):
    """
    Plot on graph for focus and one graph for defocus, each with slice and sum. \n
    Input grid of intensities (map). Same x axis for both ! \n
    First input is focused, second is defocused for label purpose. \n
    """

    title = 'Defocused by {} m'.format(shift)

    if normalised:
        norm='peak'
        parttitle = ''
    else:
        norm = ''
        parttitle = 'nope'

    if logscale:
        whichscale = 'log'
    else:
        whichscale = 'linear' 


    pop.display_psf(psf_1, title=title, colorbar=True, colorbar_orientation='vertical', 
                    scale=whichscale, normalize=norm)
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')

    plt.tight_layout()


def foc_VS_defoc_img(psf_1, psf_2, shift, logscale=True, normalised=True):
    """
    Plot on graph for focus and one graph for defocus, each with slice and sum. \n
    Input grid of intensities (map). Same x axis for both ! \n
    First input is focused, second is defocused for label purpose. \n
    """

    title1 = 'In focus'
    title2 = 'Defocused by {} m'.format(shift)

    if normalised:
        norm='peak'
        parttitle = ''
    else:
        norm = ''
        parttitle = 'nope'

    if logscale:
        whichscale = 'log'
    else:
        whichscale = 'linear'


    """
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Scale = '+whichscale + ', normalised ? ' + norm + parttitle)

    display_psf(psf_1, title=title1, colorbar_orientation='horizontal', scale=whichscale, normalize=norm)
    ax1.set_xlabel('x [arcsec]')
    ax1.set_ylabel('y [arcsec]')
    ax1.set_yscale(whichscale)
    ax1.legend()

    display_psf(psf_1, title=title1, colorbar_orientation='horizontal', scale=whichscale, normalize=norm)
    ax2.set_xlabel('x [arcsec]')
    ax2.set_ylabel('y [arcsec]')
    ax2.set_yscale(whichscale)
    ax2.legend()
    
    plt.tight_layout()
    """


    #plt.suptitle('Scale = '+whichscale + ', normalised ? ' + norm + parttitle)

    plt.subplot(1, 2, 1)
    pop.display_psf(psf_1, title=title1, colorbar_orientation='horizontal', 
                    scale=whichscale, normalize=norm, vmin=10**-2)
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')

    plt.subplot(1, 2, 2)
    pop.display_psf(psf_2, title=title2.format(shift), colorbar_orientation='horizontal', 
                    scale=whichscale, normalize=norm, vmin=10**-2)
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')

    plt.tight_layout()

    
def psf2d_and_focdefoc(psf_1, psf_2, shift, map1, map2, xaxis1, xaxis2, logscale=True, normalised=True, wantslice=True):
    
    map_1 = psf_1[0].data
    map_2 = psf_2[0].data
    """
    Plot focused PSF and defocused PSF.  \n
    Input grid of intensities (map). \n
    First input is focused, second is defocused for label purpose. \n
    If wantslice==False, returns sum on columns.
    
    Plot one graph for focus and one graph for defocus, each with slice and sum. \n
    Input grid of intensities (map). Same x axis for both ! \n
    First input is focused, second is defocused for label purpose. \n
    """
        
    if normalised:    
        ylabel = 'Normalised intensity [I]'
        if wantslice:
            slice_or_sum = 'slice'
            psf_1 = slicemap(map1)
            psf_2 = slicemap(map2)
        else:
            slice_or_sum = 'sum'
            psf_1 = sumit(map1)
            psf_2 = sumit(map2)
        
    else:    
        ylabel = 'Intensity [I]'
        if wantslice:
            slice_or_sum = 'slice'
            psf_1 = slicemap(map1, normalised=False)
            psf_2 = slicemap(map2, normalised=False)
        else:
            slice_or_sum = 'sum'
            psf_1 = sumit(map1, normalised=False)
            psf_2 = sumit(map2, normalised=False)
        
    
    if logscale:
        whichscale = 'log'
    else:
        whichscale = 'linear'

    
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    ax1.plot(xaxis1, psf_1, label='focused')
    ax1.plot(xaxis2, psf_2, label='defocused')
    ax1.set_xlabel('Focal plane distance [$\lambda/D$]')
    ax1.set_ylabel(ylabel)
    ax1.set_title(slice_or_sum + ': focused VS defocused')
    ax1.set_yscale(whichscale)
    ax1.legend()
#-------------------------------------------------------------------------

    title1 = 'In focus'
    title2 = 'Defocused by {} m'.format(shift)

    if normalised:
        norm='peak'
        parttitle = ''
    else:
        norm = ''
        parttitle = 'nope'

    if logscale:
        whichscale = 'log'
    else:
        whichscale = 'linear'


    plt.suptitle('Scale = '+whichscale + ', normalised ? ' + norm + parttitle)

    plt.subplot(1, 2, 1)
    pop.display_psf(psf_1, title=title1, colorbar_orientation='horizontal', 
                    scale=whichscale, normalize=norm)
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')

    plt.subplot(1, 2, 2)
    pop.display_psf(psf_2, title=title2.format(shift), colorbar_orientation='horizontal', 
                    scale=whichscale, normalize=norm)
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')

    plt.tight_layout()
    
############################################################ Shift related
def compute_nwaves(focal, wvlgth, rad_tel, shift):
    F_nb = focal/(rad_tel*2)
    InPhase = 2*np.pi / wvlgth
    
    return shift * InPhase / (8*F_nb**2) 


def make_psf(final_ap, focal, wvlgth, rad_tel, pixsize, fov, shift=0, showimg=True, logscale=True, showsteps=False):


    osys = pop.OpticalSystem()              
    osys.add_pupil(final_ap)
    if shift!=0:
        nbwaves = compute_nwaves(focal, wvlgth, rad_tel, shift)                    # Zernike sign convention inversed
        osys.add_pupil(pop.ThinLens(nwaves=nbwaves,                                # Defocus in wvlgth
                                reference_wavelength=wvlgth, radius=rad_tel))
    osys.add_detector(pixelscale=pixsize, fov_arcsec=fov)                      # Add detector to optical system

    if showimg:
        if showsteps:
            psf = osys.calc_psf(wvlgth, display_intermediates=True)
        else:
            psf = osys.calc_psf(wvlgth)
            one_plot(psf, shift, logscale=logscale, normalised=True)
    else:
        psf = osys.calc_psf(wvlgth)
    
    return psf

def make_aperture(rad_obs, nb_spid, spid_width, rad_tel, showimg=False, rotation=45):

    bigpup = pop.CircularAperture(radius=rad_tel)
    anti_ap = pop.SecondaryObscuration(secondary_radius=rad_obs, n_supports=nb_spid,
                                    support_width=spid_width, rotation=45) 

    optics=[bigpup, anti_ap]
    final_ap = pop.CompoundAnalyticOptic(opticslist=optics, name='Auxtel')

    if showimg:
        final_ap.display(npix=1024, colorbar_orientation='vertical')

    return final_ap