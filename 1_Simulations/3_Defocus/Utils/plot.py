import matplotlib.pyplot as plt


def slicemap(map, normalised=True):
    """Cut out the row at size/2 and normalize."""
    
    if normalised==True:
        return map[:,map.shape[0]//2] / map.max()
    else:
        return map[:,map.shape[0]//2]


def sumit(map, normalised=True, slice_middle=True):
    """Sum on columns and normalize."""
    
    if normalised:
        return  map.sum(0) / map.sum(0).max()
    else:
        return  map.sum(0)


def slicum_for_focdefoc(map1, map2, xaxis, xaxis2, logscale=True, normalised=True, ymin=10**-5, xmin=-5, xmax=5):
    """
    Plot on graph for focus and one graph for defocus, each with slice and sum. \n
    Input grid of intensities (map). Same x axis for both ! \n
    First input is focused, second is defocused for label purpose. \n
    """
    
    if normalised:         
        ylabel = 'Normalised intensity [I]'         
        psf_slice1 = slicemap(map1)
        psf_sum1 = sumit(map1)       
        psf_slice2 = slicemap(map2)
        psf_sum2 = sumit(map2)
    else:                  
        ylabel = 'Intensity [I]'
        psf_slice1 = slicemap(map1, normalised=False)
        psf_sum1 = sumit(map1, normalised=False)
        psf_slice2 = slicemap(map2, normalised=False)
        psf_sum2 = sumit(map2, normalised=False)
        
    
    if logscale:
        whichscale = 'log'
    else:
        whichscale = 'linear'
        
    
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Slice VS Sum")

    ax1.plot(xaxis, psf_slice1, label='slice')
    ax1.plot(xaxis, psf_sum1, label='sum')
    ax1.set_xlabel('Focal plane distance [arcsec]') #$\lambda/D$
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(bottom=ymin)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_title('Focused PSF')
    ax1.set_yscale(whichscale)
    ax1.legend()

    ax2.plot(xaxis, psf_slice2, label='slice')
    ax2.plot(xaxis, psf_sum2, label='sum')
    ax2.set_xlabel('Focal plane distance [arcsec]')
    ax2.set_ylabel(ylabel)
    ax2.set_ylim(bottom=ymin)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.set_title('Modified PSF')
    ax2.set_yscale(whichscale)
    ax2.legend()
    
    plt.tight_layout()



def foc_VS_defoc(map1, map2, xaxis1, xaxis2, logscale=True, normalised=True, wantslice=True, ymin=10**-5, xmin=-5, xmax=5):
    """
    Plot focused PSF and defocused PSF.  \n
    Input grid of intensities (map). \n
    First input is focused, second is defocused for label purpose. \n
    If wantslice==False, returns sum on columns.
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
    ax1.set_ylim(bottom=ymin)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.set_title(slice_or_sum + ': focused VS defocused')
    ax1.set_yscale(whichscale)
    ax1.legend()




def HCIPy_vs_POPPY(map1, map2, xaxis1, xaxis2, logscale=True, normalised=True, wantslice=True, focused=True, ymin=10**-5, xmin=-5, xmax=5):
    """
    Plot HCIPy PSF VS POPPY PSF.  \n
    Input grid of intensities (map). \n
    First input is HCIPY, second is POPPY for label purpose. \n
    If wantslice==False, returns sum on columns.
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
    
    if focused:
        focornot = 'focused'
    else:
        focornot = 'not focused'
        

    
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    ax1.plot(xaxis1, psf_1, label=focornot+' HCIPy')
    ax1.plot(xaxis2, psf_2, label=focornot+' POPPY')
    ax1.set_xlabel('Focal plane distance [arcsec]')  #not same units !
    ax1.set_ylabel(ylabel)
    ax1.set_title(slice_or_sum + ' ' + focornot + ': HCIPy VS POPPY')
    ax1.set_yscale(whichscale)
    ax1.set_ylim(bottom=ymin)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.legend()

def HCIPy_vs_POPPY2(map1, map2, xaxis1, xaxis2, logscale=True, normalised=True, focused=True, ymin=10**-5, xmin=-5, xmax=5):
    """
    Plot HCIPy PSF VS POPPY PSF.  \n
    Input grid of intensities (map). \n
    First input is HCIPY, second is POPPY for label purpose. \n
    If wantslice==False, returns sum on columns.
    """
        
    if normalised:    
        ylabel = 'Normalised intensity [I]'
        psf_1_slice = slicemap(map1)
        psf_2_slice = slicemap(map2)
        psf_1_sum = sumit(map1)
        psf_2_sum = sumit(map2)
        
    else:    
        ylabel = 'Intensity [I]'
        slice_or_sum = 'slice'
        psf_1_slice = slicemap(map1, normalised=False)
        psf_2_slice = slicemap(map2, normalised=False)
        psf_1_sum = sumit(map1, normalised=False)
        psf_2_sum = sumit(map2, normalised=False)
        
    
    if logscale:
        whichscale = 'log'
    else:
        whichscale = 'linear'
    
    if focused:
        focornot = 'focused'
    else:
        focornot = 'not focused'
        

    
    fig, [ax1,ax2] = plt.subplots(nrows=1, ncols=2)

    ax1.plot(xaxis1, psf_1_slice, label=focornot+' HCIPy')
    ax1.plot(xaxis2, psf_2_slice, label=focornot+' POPPY')
    ax1.set_xlabel('Focal plane distance [arcsec]')  #not same units !
    ax1.set_ylabel(ylabel)
    ax1.set_title('slice ' + focornot + ': HCIPy VS POPPY')
    ax1.set_yscale(whichscale)
    ax1.set_ylim(bottom=ymin)
    ax1.set_xlim(left=xmin, right=xmax)
    ax1.legend()

    ax2.plot(xaxis1, psf_1_sum, label=focornot+' HCIPy')
    ax2.plot(xaxis2, psf_2_sum, label=focornot+' POPPY')
    ax2.set_xlabel('Focal plane distance [arcsec]')  #not same units !
    ax2.set_ylabel(ylabel)
    ax2.set_title('sum ' + focornot + ': HCIPy VS POPPY')
    ax2.set_yscale(whichscale)
    ax2.set_ylim(bottom=ymin)
    ax2.set_xlim(left=xmin, right=xmax)
    ax2.legend()