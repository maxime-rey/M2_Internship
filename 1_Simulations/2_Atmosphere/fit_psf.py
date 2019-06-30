#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: <2018-06-04 15:34:50 ycopin>

"""
3D-spectrography PSF analysis.
"""

from __future__ import division, print_function
from astropy.table import Table

import numpy as N
import matplotlib.pyplot as P

import iminuit
from iminuit.frontends import console #Frontend
import psfModels

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"


class MetaSlices(object):

    def __init__(self, filename):
        """
        Read metaslices from FITS table into an Astropy Table.
        """

        self.filename = filename          #: Input filename
        self.tab = Table.read(filename)   #: Astropy table
        self.ns = len(self.tab)           #: Nb of rows/stars

        self.waves = self.tab[0]['wave']  #: Wavelengthes [Å]
        self.nw, self.ny, self.nx = inshape = self.tab[0]['data'].shape

        # Sanity checks
        if not N.alltrue([row['data'].shape == inshape for row in self.tab]):
            raise IndexError("Data shape is not constant among rows.")
        if not N.alltrue([row['var'].shape == inshape for row in self.tab]):
            raise IndexError("Variance shape is not constant among rows.")
        if not N.allclose(self.tab[:]['wave'], self.waves):
            raise IndexError("Wavelengths are not constant among rows.")

    def __str__(self):

        s = "{}: {} rows, shape=({}, {}, {}) in {:.0f}--{:.0f} Å".format(
            self.filename, self.ns,
            self.nw, self.ny, self.nx, self.waves[0], self.waves[-1])

        return s

    def extract_metaslice(self, index, lbda):
        """
        Extract flux and error at given index and wavelength.

        .. Warning:: `var` column is actually std error.
        """

        row = self.tab[index]
        iwave = N.argwhere(N.isclose(self.waves, lbda))
        if len(iwave) != 1:
            raise IndexError("Wavelength {} not found.".format(lbda))

        f = N.ma.masked_invalid(row['data'][iwave[0]])  # (1, ny, nx)
        df = N.ma.masked_invalid(row['var'][iwave[0]])

        return f[0], df[0]


def create_coords(shape, starts=0, steps=1, sparse=False):
    """
    Create coordinate arrays.

    Create ...,y,x-coordinate arrays for a given input shape `(...,
    ny, nx)`.  Each of the ndim coordinate arrays will have input
    shape. The output list is squeezed. `starts='auto'` centers *all*
    coordinates on 0.

    >>> create_coords((3,), starts='auto')
    array([-1.,  0.,  1.])
    >>> N.array(create_coords((3, 2),
    ...         starts=(1, 0), steps=(3, 2), sparse=False))
    array([[[ 1.,  1.],
            [ 4.,  4.],
            [ 7.,  7.]],
           [[ 0.,  2.],
            [ 0.,  2.],
            [ 0.,  2.]]])
    >>> create_coords((3, 2), starts=(1, 0), steps=(3, 2), sparse=True)
    [array([[ 1.],
            [ 4.],
            [ 7.]]), array([[ 0.,  2.]])]
    """

    ndim = len(shape)

    # N-dim steps and starts
    steps = N.broadcast_to(steps, (ndim,))        # (ndim,)
    if starts == 'auto':                          # Center coordinates
        starts = [(1 - n) * step / 2.
                  for n, step in zip(shape, steps)]
    else:
        starts = N.broadcast_to(starts, (ndim,))  # (ndim,)

    coords = [N.arange(n, dtype=float) * step + start
              for n, step, start in zip(shape, steps, starts)]

    if ndim == 1:
        return N.meshgrid(coords[0])[0]
    else:
        return N.meshgrid(*coords[::-1], sparse=sparse)[::-1]


class Kolmogorov_PSF(object):

    def __init__(self, shape=(15, 15), scale=0.43, lbda=5e-7):
        """
        :param shape: spatial shape (ny, nx)
        :param float scale: spatial scale [arcsec/spx]
        :param float lbda: reference wavelength [Å]
        """

        self.ny, self.nx = shape
        self.scale = float(scale)  #: Spatial scale [arcsec/spx]
        #: Spatial cartesian coordinates [arcsec]
        self.y, self.x = create_coords(
            shape, starts='auto', steps=self.scale, sparse=True)

        self.lbda = float(lbda)    #: Reference wavelength [Å]

        # PSF parameters
        self.ampl = 1              #: Peak amplitude
        self.x0 = 0                #: PSF x-center [arcsec]
        self.y0 = 0                #: PSF y-center [arcsec]
        self.cy2 = 1               #: Elliptical radius y2-coefficient
        self.cxy = 0               #: Elliptical radius xy-coefficient
        #: Reference Fried radius [m] at ref. wavelength
        self.r0 = 0.1
        self.expo = 5/3            #: Kolmogorov index
        self.bkgnd = 0             #: Background level

        self.flux = None           #: Observed PSF
        self.dflux = None          #: Associated variance

    def get_seeing(self, lbda=None):
        """
        FWHM seeing [arcsec].

        :param lbda: wavelength [Å], default to ref. wavelength
        """

        if lbda is None:
            lbda = self.lbda

        return psfModels.seeing_fwhm(
            lbda * 1e-10, r0ref=self.r0, lref=self.lbda * 1e-10, expo=self.expo)

    def __str__(self):

        s = "PSF({}×{}) @{:.0f} Å:".format(self.ny, self.nx, self.lbda)
        for key, val in self.get_params().iteritems():
            s += "\n  {:>4s} = {}".format(key, val)
            if key == 'r0':
                s += ", seeing = {:.2f}'' = {:.2f}'' @ 5000 Å".format(
                    self.get_seeing(), self.get_seeing(5000.))
            if key == 'expo':
                s += " = {:.2f} / 3".format(self.expo * 3)
        if self.flux is not None:
            s += "\n  data: {} points, max={}".format(
                self.flux.count(), self.flux.max())

        return s

    def estimate_r0(self, seeing, lbda):
        """
        Estimate Fried radius [m] at reference wavelength from FWHM seeing [arcsec]
        at wavelength lbda [Å].

        :param seeing: FWHM seeing [arcsec]
        :param lbda: wavelength [Å]
        """

        r0 = psfModels.r0_from_seeing(
            seeing, lbda * 1e-10, expo=self.expo)  # At lbda

        return psfModels.friedParamater(self.lbda * 1e-10,  # At self.lbda
                                        r0ref=r0, lref=lbda * 1e-10, expo=self.expo)

    def get_params(self):

        from collections import OrderedDict

        return OrderedDict((('ampl', self.ampl),
                            ('x0', self.x0),
                            ('y0', self.y0),
                            ('cy2', self.cy2),
                            ('cxy', self.cxy),
                            ('r0', self.r0),
                            ('expo', self.expo),
                            ('bkgnd', self.bkgnd)))

    def set_params(self, **kwargs):

        for key, val in kwargs.iteritems():
            if not hasattr(self, key):
                raise AttributeError("Unknown parameter '{}'.".format(key))
            setattr(self, key, float(val))

    def get_extent(self):
        """
        Return FoV extent (left, right, bottom, top) [arcsec].
        """

        return (self.x[0, 0], self.x[0, -1],
                self.y[0, 0], self.y[-1, 0])  # L, R, B, T

    def radius_elliptical(self, **kwargs):
        """
        Elliptical radius.
        """

        dx = self.x - kwargs.get('x0', self.x0)
        dy = self.y - kwargs.get('y0', self.y0)
        cy2 = kwargs.get('cy2', self.cy2)
        cxy = kwargs.get('cxy', self.cxy)

        return N.abs(dx**2 + cy2 * dy**2 + cxy * dx * dy) ** 0.5  # [arcsec]

    def radial_profile(self, r, **kwargs):
        """
        PSF profile at radii r [arcsec].

        :param r: (generalized) radii [arcsec]
        """

        r0 = kwargs.get('r0', self.r0)
        expo = kwargs.get('expo', self.expo)

        # Spline interpolator
        spl = psfModels.psf_Kolmogorov_Hankel_interp(
            N.max(r), self.lbda * 1e-10, r0=r0, expo=expo)

        return spl(r)  # Interpolated values

    def psf(self):

        r = self.radius_elliptical()  # Elliptical radius
        p = self.radial_profile(r)    # Peak-normalized PSF

        return self.ampl * p + self.bkgnd

    def set_data(self, flux, dflux, flux_scale=1e-12):

        assert N.shape(flux) == (self.ny, self.nx)

        self.flux = N.ma.MaskedArray(flux) / flux_scale
        self.dflux = N.ma.MaskedArray(dflux) / flux_scale

    def initial_guess(self):

        if self.flux is None:
            raise NotImplementedError()

        bkgnd = N.percentile(self.flux.filled(0), 20)  # 1st quintile
        ampl = self.flux.max() - bkgnd
        weights = (self.flux - bkgnd).filled(0)
        x0 = N.average(self.x.squeeze(), weights=weights.sum(axis=0))
        y0 = N.average(self.y.squeeze(), weights=weights.sum(axis=1))
        expo = 3 / 2 
        r0 = 0.1
        cy2 = 1
        cxy = 0


        self.set_params(ampl=ampl, x0=x0, y0=y0, expo=expo, bkgnd=bkgnd, r0=r0, cy2=cy2, cxy=cxy)

    def chi2(self, ampl=1, x0=0, y0=0, cy2=1, cxy=0, r0=0.1, expo=5/3, bkgnd=0):

        self.set_params(ampl=ampl, x0=x0, y0=y0, cy2=cy2, cxy=cxy,
                        r0=r0, expo=expo, bkgnd=bkgnd)

        res = (self.psf() - self.flux) / self.dflux

        return N.dot(res.ravel(), res.ravel())
    
    def fit(self, **kwargs):
        """
        Chi2-fit using current parameters as initial guess.
        """

        # Initialization
        init = self.get_params()                    # Initial guess
        init.update(error_x0=1, error_y0=1,         # Initial steps
                    error_ampl=1,
                    error_cy2=0.1, error_cxy=0.1,
                    error_r0=0.01, error_expo=0.01,
                    error_bkgnd=1)
        init.update(limit_ampl=(self.ampl / 10, self.ampl * 10),  # Limits
                    limit_cxy=(0,10),
                    limit_cy2=(0.1, 10),
                    limit_expo=(0.1, 2-1e-3), # expo >= 2 doesn't work (PSF takes negative values)
                    limit_r0=(1e-2, 1),
                    limit_bkgnd=(0, self.ampl))
        init.update(kwargs)

        self.minuit = iminuit.Minuit(self.chi2, **init)

        # Fit
        self.minuit.migrad()

        # Results
        fmin = self.minuit.get_fmin()
        if fmin.is_valid:
            # Save best fit in actual params
            self.set_params(**self.minuit.values)
            print("Chi2 = {}, DoF = {}".format(
                self.minuit.fval, self.flux.count() - self.minuit.narg))
            for key in self.minuit.parameters:
                print("{} = {} ± {}".format(
                    key, self.minuit.values[key], self.minuit.errors[key]))

        return self.minuit

    def figure_psf(self, log=False):

        if self.flux is None:
            raise NotImplementedError()

        seeing = self.get_seeing()

        fig = P.figure(tight_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, aspect='equal',
                              xlabel="x [arcsec]", ylabel="y [arcsec]",
                              title=u"Seeing={:.2f}'' @ {:.0f} Å\n"
                              u"({:.2f}'' @ 5000 Å)".format(
                                  seeing, self.lbda, self.get_seeing(5000)))
        ax2 = fig.add_subplot(2, 2, 2,
                              #xlabel="r [arcsec]",
                              yscale='log', ylabel="Flux")
        ax4 = fig.add_subplot(2, 2, 4,
                              xlabel="r [arcsec]",
                              ylabel="Relative error")

        # Contours
        extent = self.get_extent()
        norm = P.matplotlib.colors.LogNorm() if log else None


        import astropy.visualization as AV
        imnorm = AV.ImageNormalize(
            self.flux,
            # interval=AV.ZScaleInterval(),
            stretch=AV.LogStretch())

        ax1.imshow(self.flux, extent=extent,
                        norm=imnorm, cmap='gray_r', label="Observed", origin='lower')
        cnt = ax1.contour(self.flux, extent=extent,
                          norm=norm, cmap='gray_r', label="Observed", origin='lower')
        ax1.contour(self.psf(), levels=cnt.levels, extent=extent,
                    norm=norm, cmap='magma', label="Adjusted", origin='lower')

        # cbar = fig.colorbar(im, ax=ax1, orientation='horizontal')
        # cbar.add_lines(cnt)

        # Elliptical radius
        r = self.radius_elliptical()
        ax1.plot([self.x0], [self.y0], marker='*', color='g', mec='0.2')
        cnt = ax1.contour(r, levels=[2*seeing], extent=extent,
                          colors='g', linewidths=0.5, label="r=2''")
        # cnt.clabel(inline=1)
        # ax1.legend(loc='best', fontsize='small')

        # Radial plot
        mr = N.ma.MaskedArray(data=r, mask=self.flux.mask)
        ax2.plot(mr, self.flux, color='k', marker='.', ls='none')

        r = N.logspace(-2, N.log10(mr.max()), max(50, int(mr.count() ** 0.5)))
        p = self.radial_profile(r) * self. ampl + self.bkgnd

        l, = ax2.plot(r, p, label="n={:.2f}".format(self.expo))
        ax2.set_autoscaley_on(False)
        ax2.axhline(self.bkgnd, ls='--')
        ax2.legend(loc='best', fontsize='small')

        # Residuals
        mp = N.ma.MaskedArray(data=self.psf(), mask=self.flux.mask)
        ax4.plot(mr, (self.flux - mp)/mp, color='k', marker='.', ls='none')
        ax4.axhline(0, color=l.get_color())

        return fig

class vonKarman_PSF(object):

    def __init__(self, shape=(15, 15), scale=0.43, lbda=5e-7):
        """
        :param shape: spatial shape (ny, nx)
        :param float scale: spatial scale [arcsec/spx]
        :param float lbda: reference wavelength [Å]
        """

        self.ny, self.nx = shape
        self.scale = float(scale)  #: Spatial scale [arcsec/spx]
        #: Spatial cartesian coordinates [arcsec]
        self.y, self.x = create_coords(
            shape, starts='auto', steps=self.scale, sparse=True)

        self.lbda = float(lbda)    #: Reference wavelength [Å]

        # PSF parameters
        self.ampl = 1              #: Peak amplitude
        self.x0 = 0                #: PSF x-center [arcsec]
        self.y0 = 0                #: PSF y-center [arcsec]
        self.cy2 = 1               #: Elliptical radius y2-coefficient
        self.cxy = 0               #: Elliptical radius xy-coefficient
        #: Reference Fried radius [m] at ref. wavelength
        self.r0 = 0.1
        self.L0 = 10               #: External length of coherence
        self.bkgnd = 0             #: Background level

        self.flux = None           #: Observed PSF
        self.dflux = None          #: Associated variance

    def get_seeing(self, lbda=None):
        """
        FWHM seeing [arcsec].

        :param lbda: wavelength [Å], default to ref. wavelength
        """

        if lbda is None:
            lbda = self.lbda

        return psfModels.seeing_fwhm_vK(
            lbda * 1e-10, r0ref=self.r0, lref=self.lbda * 1e-10, L0=self.L0)

    def __str__(self):

        s = "PSF({}×{}) @{:.0f} Å:".format(self.ny, self.nx, self.lbda)
        for key, val in self.get_params().iteritems():
            s += "\n  {:>4s} = {}".format(key, val)
        if self.flux is not None:
            s += "\n  data: {} points, max={}".format(
                self.flux.count(), self.flux.max())

        return s

    def estimate_r0(self, seeing, lbda):
        """
        Estimate Fried radius [m] at reference wavelength from FWHM seeing [arcsec]
        at wavelength lbda [Å].

        :param seeing: FWHM seeing [arcsec]
        :param lbda: wavelength [Å]
        """

        r0 = psfModels.r0_from_seeing_vK(
            seeing, lbda * 1e-10, L0=self.L0)  # At lbda

        return psfModels.friedParamater(self.lbda * 1e-10,  # At self.lbda
                                        r0ref=r0, lref=lbda * 1e-10, L0=self.L0)

    def get_params(self):

        from collections import OrderedDict

        return OrderedDict((('ampl', self.ampl),
                            ('x0', self.x0),
                            ('y0', self.y0),
                            ('cy2', self.cy2),
                            ('cxy', self.cxy),
                            ('r0', self.r0),
                            ('L0', self.L0),
                            ('bkgnd', self.bkgnd)))

    def set_params(self, **kwargs):

        for key, val in kwargs.iteritems():
            if not hasattr(self, key):
                raise AttributeError("Unknown parameter '{}'.".format(key))
            setattr(self, key, float(val))

    def get_extent(self):
        """
        Return FoV extent (left, right, bottom, top) [arcsec].
        """

        return (self.x[0, 0], self.x[0, -1],
                self.y[0, 0], self.y[-1, 0])  # L, R, B, T

    def radius_elliptical(self, **kwargs):
        """
        Elliptical radius.
        """

        dx = self.x - kwargs.get('x0', self.x0)
        dy = self.y - kwargs.get('y0', self.y0)
        cy2 = kwargs.get('cy2', self.cy2)
        cxy = kwargs.get('cxy', self.cxy)

        return N.abs(dx**2 + cy2 * dy**2 + cxy * dx * dy) ** 0.5  # [arcsec]

    def radial_profile(self, r, **kwargs):
        """
        PSF profile at radii r [arcsec].

        :param r: (generalized) radii [arcsec]
        """

        r0 = kwargs.get('r0', self.r0)
        L0 = kwargs.get('L0', self.L0)

        # Spline interpolator
        spl = psfModels.psf_vonKarman_Hankel_interp(
            N.max(r), self.lbda * 1e-10, r0=r0, L0=L0)

        return spl(r)  # Interpolated values

    def psf(self):

        r = self.radius_elliptical()  # Elliptical radius
        p = self.radial_profile(r)    # Peak-normalized PSF

        return self.ampl * p + self.bkgnd

    def set_data(self, flux, dflux, flux_scale=1e-12):

        assert N.shape(flux) == (self.ny, self.nx)

        self.flux = N.ma.MaskedArray(flux) / flux_scale
        self.dflux = N.ma.MaskedArray(dflux) / flux_scale

    def initial_guess(self):

        if self.flux is None:
            raise NotImplementedError()

        bkgnd = N.percentile(self.flux.filled(0), 20)  # 1st quintile
        ampl = self.flux.max() - bkgnd
        weights = (self.flux - bkgnd).filled(0)
        x0 = N.average(self.x.squeeze(), weights=weights.sum(axis=0))
        y0 = N.average(self.y.squeeze(), weights=weights.sum(axis=1))
        L0 = 20
        r0 = 0.1
        cy2 = 1
        cxy = 0


        self.set_params(ampl=ampl, x0=x0, y0=y0, L0=L0, bkgnd=bkgnd, r0=r0, cy2=cy2, cxy=cxy)

    def chi2(self, ampl=1, x0=0, y0=0, cy2=1, cxy=0, r0=0.1, L0=10, bkgnd=0):

        self.set_params(ampl=ampl, x0=x0, y0=y0, cy2=cy2, cxy=cxy,
                        r0=r0, L0=L0, bkgnd=bkgnd)

        res = (self.psf() - self.flux) / self.dflux
         
        return N.dot(res.ravel(), res.ravel())
    
    def fit(self, **kwargs):
        """
        Chi2-fit using current parameters as initial guess.
        """

        # Initialization
        init = self.get_params()                    # Initial guess
        init.update(error_x0=1, error_y0=1,         # Initial steps
                    error_ampl=1,
                    error_cy2=0.1, error_cxy=0.1,
                    error_r0=0.01, error_L0=1,
                    error_bkgnd=1)
        init.update(limit_ampl=(self.ampl / 10, self.ampl * 10),  # Limits
                    limit_cxy=(0,10),
                    limit_cy2=(0.1, 10),
                    limit_L0=(1, 100),
                    limit_r0=(1e-2, 1),
                    limit_bkgnd=(0, self.ampl))
        init.update(kwargs)

        self.minuit = iminuit.Minuit(self.chi2, frontend=ConsoleFrontend(), **init)

        # Fit
        self.minuit.migrad()

        # Results
        fmin = self.minuit.get_fmin()
        if fmin.is_valid:
            # Save best fit in actual params
            self.set_params(**self.minuit.values)
            print("Chi2 = {}, DoF = {}".format(
                self.minuit.fval, self.flux.count() - self.minuit.narg))
            for key in self.minuit.parameters:
                print("{} = {} ± {}".format(
                    key, self.minuit.values[key], self.minuit.errors[key]))

        return self.minuit

    def figure_psf(self, log=False):

        if self.flux is None:
            raise NotImplementedError()

        seeing = self.get_seeing()

        fig = P.figure(tight_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, aspect='equal',
                              xlabel="x [arcsec]", ylabel="y [arcsec]",
                              title=u"Seeing={:.2f}'' @ {:.0f} Å\n"
                              u"({:.2f}'' @ 5000 Å)".format(
                                  seeing, self.lbda, self.get_seeing(5000)))

        ax2 = fig.add_subplot(2, 2, 2,
                              #xlabel="r [arcsec]",
                              yscale='log', ylabel="Flux")
        ax4 = fig.add_subplot(2, 2, 4,
                              xlabel="r [arcsec]",
                              ylabel="Relative error")

        # Contours
        extent = self.get_extent()
        norm = P.matplotlib.colors.LogNorm() if log else None


        import astropy.visualization as AV
        imnorm = AV.ImageNormalize(
            self.flux,
            # interval=AV.ZScaleInterval(),
            stretch=AV.LogStretch())

        ax1.imshow(self.flux, extent=extent,
                        norm=imnorm, cmap='gray_r', label="Observed", origin='lower')
        cnt = ax1.contour(self.flux, extent=extent,
                          norm=norm, cmap='gray_r', label="Observed", origin='lower')
        ax1.contour(self.psf(), levels=cnt.levels, extent=extent,
                    norm=norm, cmap='magma', label="Adjusted", origin='lower')

        
        
        # cbar = fig.colorbar(im, ax=ax1, orientation='horizontal')
        # cbar.add_lines(cnt)

        # Elliptical radius
        r = self.radius_elliptical()
        ax1.plot([self.x0], [self.y0], marker='*', color='g', mec='0.2')
        cnt = ax1.contour(r, levels=[2*seeing], extent=extent,
                          colors='g', linewidths=0.5, label="r=2''")
        # cnt.clabel(inline=1)
        # ax1.legend(loc='best', fontsize='small')

        # Radial plot
        mr = N.ma.MaskedArray(data=r, mask=self.flux.mask)
        ax2.plot(mr, self.flux, color='k', marker='.', ls='none')

        r = N.logspace(-2, N.log10(mr.max()), max(50, int(mr.count() ** 0.5)))
        p = self.radial_profile(r, L0=self.L0) * self. ampl + self.bkgnd

        l, = ax2.plot(r, p, label="L0={:.2f}".format(self.L0))
        ax2.set_autoscaley_on(False)
        ax2.axhline(self.bkgnd, ls='--')
        ax2.legend(loc='best', fontsize='small')

        # Residuals
        mp = N.ma.MaskedArray(data=self.psf(), mask=self.flux.mask)
        ax4.plot(mr, (self.flux - mp)/mp, color='k', marker='.', ls='none')
        ax4.axhline(0, color=l.get_color())

        return fig

if __name__ == '__main__':    
    ms = MetaSlices('Data/meta_slices_B_full.fits') #Read the table to treat
    table = ms.tab
    nstars = ms.ns     # Number of rows/stars
    nwaves = ms.nw  # Number of metaslices/waves
    

    N.random.seed(0) #Fix random for reproductability
    irows = N.arange(ms.ns)  # Row indices
    N.random.shuffle(irows)  # Shuffled indices
    
    t = Table(table[irows])
    
    kpsf = Kolmogorov_PSF(shape=(ms.ny, ms.nx), lbda=ms.waves[0])
    pnames = kpsf.get_params().keys()
    
    for name in pnames:
        t[name] = N.zeros((len(t), nwaves, 2))

    t['chi2'] = N.zeros((len(t), nwaves))
    t['DoF'] = N.zeros((len(t), nwaves))        
    t['cov_mat'] = N.zeros((len(t), nwaves, len(pnames), len(pnames)))
    
    counter = 0
    count = 20 #number of stars to treat before writing file
    
    for i in range(len(t)):
        for j in range(nwaves):
            print('etoile {0}, jd {1},metaslice {2}, wave {3}'.format(
                i, t[i]['jd'], j, t['wave'][i, j]))
    
    
            kpsf.lbda = ms.waves[j]
            kpsf.set_data(*ms.extract_metaslice(irows[i], kpsf.lbda))
            if j == 0 or kpsf.get_params()['ampl'] == 0:
            	kpsf.initial_guess()
            
            try:
                kpsf.fit()
                t['cov_mat'][i, j] = N.array(kpsf.minuit.matrix(correlation=True))
                t['chi2'][i, j] = kpsf.minuit.fcn()
                
                for name in pnames:
                    t[name][i, j, 0] = kpsf.minuit.values[name]
                    t[name][i, j, 1] = kpsf.minuit.errors[name]
                    
            except ValueError as err:
                print(err)
            except RuntimeError as err:
                print(err)
                
        counter += 1
        if counter == count:
            t.write('Data/meta_slices_BvK_treated.fits',overwrite=True) #File to write into
            counter = 0