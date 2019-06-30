#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: <2018-05-28 18:54:44 ycopin>

"""
Simulation of seeing PSF in large telescopes (D>>r0), tested against
SNfactory ad-hoc PSF models.

References:

* Tokovinin 2002PASP..114.1156T
* Fried 1966JOSA...56.1372F
* Racine 1996PASP..108..699R

Author: Yannick Copin <y.copin@ipnl.in2p3.fr>
"""

from __future__ import division, print_function

import numpy as N

from scipy import special as S
from scipy import fftpack as F
from scipy.optimize import fsolve
from scipy import interpolate as I

RAD2ARCS = 206264.80624709636           # From radians to arcsecs


"""
Kolmogorov Model
"""


def otf_Kolmogorov(r, r0=0.15, expo=5/3):
    """
    Kolmogorov Optical Transfer Function.
 
    :param r: [m/rad]
    :param float r0: Fried parameter [m]
    :param float expo: Kolmogorov exponent
 
    .. Note:: lower exponent (e.g. 3/2=1.5 instead of 5/3=1.67) means more
       wings.  1.55 corresponds roughly to long exposure ad-hoc SNfactory PSFs,
       short SNfactory PSFs are closer to 1.5.
    """
 
    # *Half* phase structure function
    hdphi = (r / r0)**expo
    hdphi *= ( 8/expo * S.gamma(2/expo) )**(expo/2)
 
    # otf = exp(-0.5*phase), the 0.5 factor has already been included
    return N.exp(-hdphi)

def psf_Kolmogorov_FFT_interp(rmax, lbda, pxscale, r0=0.15, expo=5/3):
    """
    Return interpolator (up to rmax) for (max-normalized) Kolmogorov
    (FFT-based) PSF given free parameters r0 and expo for radius array r
    [arcsec] at wavelength lbda [m].  (pxscale in arcsec/px.)

    .. Note:: use rmax >> seeing.
    """

    # 2D-array construction
    imax = int(rmax / pxscale + 1)                # Image half width [px]
    i = N.arange(-imax, imax + 1)                  # px-coordinate (n,) [px]
    rpx = N.hypot(*N.meshgrid(i, i, sparse=True))  # Radius 2D-array (n, n) [px]
    px2f = RAD2ARCS / pxscale / len(i)   # [1/rad]
    f = rpx * px2f                       # Spatial freq. 2D-array (n, n) [1/rad]

    # FFT-based PSF computation
    psf = psf2D_FFT(otf_Kolmogorov(lbda * f, r0=r0, expo=expo))  # (n, n)

    r = rpx * pxscale                    # Radius (n, n) [arcsec]
    j = N.argsort(r.ravel())             # Such that a.ravel()[j] is sorted
    i = j[:N.searchsorted(r.ravel()[j], rmax)]

    rr = r.ravel()[i]           # Radii up to rmax
    pp = psf.ravel()[i]         # Profile up to rmax

    # Spline interpolation in lin-log
    n, = N.nonzero(N.diff(rr))  # Discard duplicated radii (and last point)
    assert (pp[n] > 0).all()
    spl = I.UnivariateSpline(rr[n], N.log(pp[n]), k=3, s=0)

    return lambda r: N.exp(spl(r))

    
def psf_Kolmogorov_Hankel(r, lbda, r0=0.15, expo=5/3, normed=True):
    """
    Compute (max-normalized) Kolmogorov (Hankel-based) PSF given free
    parameters r0 and expo for radius array r [arcsec] at wavelength lbda [m].
    """
 
    from hankelTransform import hankelTransform
 
    # One should compute psf(r) = HT.hankelTransform(lambda r:
    # otf_Kolmogorov(lbda*r/(2*N.pi)), r/RAD2ARCS), but for some numerical
    # reasons, one computes lbda**2/(2*pi) * psf(r) = HT.hankelTransform(lambda
    # r: otf_Kolmogorov(r, r0=r0, expo=expo), 2*pi/lambda*r/RAD2ARCS)
    tmp = (6.2831853071795862 / RAD2ARCS / lbda) * N.atleast_1d(r)
    psf = hankelTransform(lambda r: otf_Kolmogorov(r, r0=r0, expo=expo), tmp)
 
    # psf(0) = 2*pi/expo * fO**2 * Gamma(2/expo)
    # where f0 = r0/lbda * beta**(-1/expo)
    # and beta = (8/expo*Gamma(2/expo))**(expo/2) = 3.44... for expo=5/3
    # Therefore f0**2 = (r0/lbda)**2 / ( 8/expo*Gamma(2/expo) )
    # and psf(0) = pi/4 * (r0/lbda)**2.  Given we actually compute
    # lbda**2/(2*pi)*psf(r), this gives psf(0) = r0**2 / 8
    pmax = r0**2 / 8
    psf[tmp == 0] = pmax                # Remove NaNs at r=0
    if normed:
        psf /= pmax                     # Max normalisation
 
    return psf.reshape(N.shape(r))      # PSF should have the same shape as r
 
 
def psf_Kolmogorov_Hankel_interp(rmax, lbda, r0=0.15, expo=5/3, npts=50):
    """
    Return interpolator (up to rmax) for (max-normalized) Kolmogorov
    (Hankel-based) PSF given free parameters r0 and expo for radius array r
    [arcsec] at wavelength lbda [m].
    """
 
    r = N.linspace(0, rmax**0.5, npts)**2  # linear sampling in sqrt(r)
    p = psf_Kolmogorov_Hankel(r, lbda, r0=r0, expo=expo, normed=True)
    # Spline interpolation in lin-log
    assert (p > 0).all()
    spl = I.UnivariateSpline(r, N.log(p), k=3, s=0)
 
    return lambda r: N.exp(spl(r))

    
def friedParamater(lbda, r0ref=0.10, lref=5e-7, expo=5/3):
    """Implement r0 chromaticity."""
 
    return r0ref * (lbda/lref)**(2/expo)

 
def seeing_fwhm(lbda, r0ref=0.10, lref=5e-7, expo=5/3, verbose=False):
    """
    Seeing FWHM [arcsec] at wavelength lbda [m] for Fried parameter r0ref [m]
    at reference wavelength lref [m].
 
    .. Note:: the *traditional* seeing is dependent on exponent 5/3 used in
       :func:`otf_Kolmogorov` through constant 0.976 (actually 0.97586...).
 
    >>> '%f' % seeing_fwhm(5e-7, r0ref=0.1, lref=5e-7, expo=5/3)
    '1.006431'
    >>> '%f' % seeing_fwhm(3e-7, r0ref=0.1, lref=5e-7, expo=5/3)
    '1.114690'
    >>> '%f' % seeing_fwhm(5e-7, r0ref=0.2, lref=5e-7, expo=5/3)
    '0.503216'
    >>> '%f' % seeing_fwhm(5e-7, r0ref=0.1, lref=5e-7, expo=3/2)
    '0.951169'
    """
 
    r0 = friedParamater(lbda, r0ref, lref)
    s = 0.975863432579 * lbda/r0 * RAD2ARCS        # [arcsec] Valid for expo=5/3
    if expo != (5/3):                              # Solve explicitely for expo != 5/3
        s = 2*fsolve(
            lambda r: psf_Kolmogorov_Hankel(r, lbda, r0=r0, expo=expo) - 0.5,
            s/2).reshape(N.shape(lbda))
 
    if verbose and N.ndim(lbda) == 0:
        if N.isclose(lbda, lref):
            print("r0=%.0f cm @%.0f Å: "
                  "seeing=%.2f\" (FWHM) (expo=%.2f)" %
                  (r0ref*100, lref*1e10, s, expo))
        else:
            print("r0=%.0f cm @%.0f Å =%.0f cm @%.0f Å: "
                  "seeing=%.2f\" (FWHM) (expo=%.2f)" %
                  (r0ref*100, lref*1e10, r0*100, lbda*1e10, s, expo))
 
    return s
 
 
def r0_from_seeing(fwhm, lbda, expo=5/3, verbose=False):
    """
    Fried parameter r0 [m] at wavelength lbda [m] for FWHM seeing [arcsec] at
    same wavelength.
 
    >>> '%f' % r0_from_seeing(1.006431409222669, 5e-7, expo=5/3)
    '0.100000'
    >>> '%f' % r0_from_seeing(0.9511686315271922, 5e-7, expo=3/2)
    '0.100000'
    >>> '%f' % friedParamater(5e-7,
    ...     r0_from_seeing(1.1146895556429748, 3e-7, expo=5/3),
    ...     lref=3e-7, expo=5/3)
    '0.100000'
    """
 
    r0 = 0.975863432579 * lbda/fwhm * RAD2ARCS  # [m] Valid for expo=5/3
    if expo != (5/3):
        r0 = fsolve(
            lambda r0: psf_Kolmogorov_Hankel(fwhm/2, lbda, r0=r0, expo=expo) - 0.5,
            r0).reshape(N.shape(fwhm))
 
    if verbose and N.ndim(fwhm) == 0:
        print("Seeing=%.2f\" (FWHM) @%.0f Å: "
              "r0=%.0f cm  (expo=%.2f)" %
              (fwhm, lbda*1e10, r0*100, expo))
 
    return r0
    
               
"""
von Karman Model
"""

def otf_vonKarman(r, r0=0.15, L0=10, desaturate=False):
    """
    von Karman optical transfer function.

    .. Note:: Tokovinin's formula has a typo -- 2**(11/6) should be 2**(5/6)
       --, see Conan (2000).
    """

    #raise NotImplementedError("Von Karman phase does not work properly...")
    #print("WARNING: Von Karman phase does not work properly...")

    # (Half) Phase structure function
    def hphase(r):
        x = (2 * N.pi * r) / L0
        # gamma(11/6)/(2**(5/6)*pi**(8/3)) * ( 24/5*gamma(6/5) )**(5/6) =
        # 0.08583068106228546...
        # gamma(5/6)/2**(1/6) = 1.0056349179985902...
        return 0.08583068106228546 * (L0/r0)**(5/3) * (1.0056349179985902 - x**(5/6) * S.kv(5/6, x))

        
    hdphi = hphase(r)
    hdphi[r == 0] = 0.                    # Extension to 0 (NaN otherwise)
    otf = N.exp(-hdphi)                 # Optical transfert function

    if desaturate:                      # See Tokovinin (2002)
        print("Desaturation:", hphase(N.max(r)), N.exp(-hphase(N.max(r))))
        otf -= N.exp(-hphase(N.max(r)))  # Remove constant level
        otf /= otf.max()                # Renormalization

    return otf

def psf_vonKarman_Hankel(r, lbda, r0=0.15, L0=10, normed=True):
    """
    Compute (max-normalized) Von Karman (Hankel-based) PSF given free
    parameters r0 and L0 for radius array r [arcsec] at wavelength lbda [m].
    """

    from hankelTransform import hankelTransform

    # One should compute psf(r) = HT.hankelTransform(lambda r:
    # otf_vonKarman(lbda*r/(2*N.pi)), r/RAD2ARCS), but for some numerical
    # reasons, one computes lbda**2/(2*pi) * psf(r) = HT.hankelTransform(lambda
    # r: otf_vonKarman(r, r0=r0, L0=L0), 2*pi/lambda*r/RAD2ARCS)
    tmp = (6.2831853071795862 / RAD2ARCS / lbda) * N.atleast_1d(r)
    psf = hankelTransform(lambda r: otf_vonKarman(r, r0=r0, L0=L0), tmp)
    
    #k = N.array([tmp[0], tmp[-1]])
    #h, Ftmp, n = H.get_h(lambda r: otf_vonKarman(r, r0=r0, L0=L0), nu=0, K=k)
    #ht = HankelTransform(nu=0, N=n, h=h)
    #psf = ht.transform(lambda r: otf_vonKarman(r, r0=r0, L0=L0), tmp, ret_err=False)
    
    tmp0 = (6.2831853071795862 / RAD2ARCS / lbda) * N.atleast_1d(1e-9)
    pmax = hankelTransform(lambda r: otf_vonKarman(r, r0=r0, L0=L0), tmp0)[0]     
    psf[tmp == 0] = pmax                # Remove NaNs at r=0
    if normed:
        psf /= pmax                     # Max normalisation

    return psf.reshape(N.shape(r))      # PSF should have the same shape as r

def psf_vonKarman_Hankel_interp(rmax, lbda, r0=0.15, L0=20, npts=50):
    """
    Return interpolator (up to rmax) for (max-normalized) von Karman
    (Hankel-based) PSF given free parameters r0 and L0 for radius array r
    [arcsec] at wavelength lbda [m].
    """

    r = N.linspace(0, rmax**0.5, npts)**2  # linear sampling in sqrt(r)
    p = psf_vonKarman_Hankel(r, lbda, r0=r0, L0=L0, normed=True)
    # Spline interpolation in lin-log
    assert (p > 0).all()
    spl = I.UnivariateSpline(r, N.log(p), k=3, s=0)

    return lambda r: N.exp(spl(r))


def seeing_fwhm_vK(lbda, r0ref=0.10, lref=5e-7, L0=20, verbose=False):
    """
    Seeing FWHM [arcsec] at wavelength lbda [m] for Fried parameter r0ref [m]
    at reference wavelength lref [m].

    """

    r0 = friedParamater(lbda, r0ref, lref)
    s = 0.975863432579 * lbda/r0 * RAD2ARCS * N.sqrt(1 - 2.183 * (r0/L0)**0.356)        # [arcsec] Valid for all L0
    s = 2*fsolve(
        lambda r: psf_vonKarman_Hankel(r, lbda, r0=r0, L0=L0) - 0.5,
        s/2).reshape(N.shape(lbda))

    if verbose and N.ndim(lbda) == 0:
        if N.isclose(lbda, lref):
            print("r0=%.0f cm @%.0f Å: "
                  "seeing=%.2f\" (FWHM) (L0=%.2f)" %
                  (r0ref*100, lref*1e10, s, L0))
        else:
            print("r0=%.0f cm @%.0f Å =%.0f cm @%.0f Å: "
                  "seeing=%.2f\" (FWHM) (L0=%.2f)" %
                  (r0ref*100, lref*1e10, r0*100, lbda*1e10, s, L0))

    return s


def r0_from_seeing_vK(fwhm, lbda, L0=20, verbose=False):
    """
    Fried parameter r0 [m] at wavelength lbda [m] for FWHM seeing [arcsec] at
    same wavelength.

    """

    r0 = 0.975863432579 * lbda/fwhm * RAD2ARCS
    r0 = fsolve(
        lambda r0: 0.975863432579 * lbda/fwhm * RAD2ARCS * N.sqrt(1 - 2.183 * (r0/L0)**0.356) -fwhm, r0).reshape(N.shape(fwhm))
        
    if verbose and N.ndim(fwhm) == 0:
        print("Seeing=%.2f\" (FWHM) @%.0f Å: "
              "r0=%.0f cm  (L0=%.2f)" %
              (fwhm, lbda*1e10, r0*100, L0))

    return r0
   
    
    
    
def psf2D_FFT(otf, normed=True):
    """
    FFT-computed 2D-PSF from input OTF 2D-array (e.g. from otf_Kolmogorov).
    """

    assert otf.ndim == 2, "Input OTF array is not 2D"

    psf = N.absolute(F.fftshift(F.fft2(otf)))  # PSF=FT(OTF), otf.shape
    if normed:
        psf /= psf.max()

    return psf
    

def integ_radial_laguerre(f, accuracy=1e-3, imin=1, imax=40, verbose=False):
    """
    Compute s = 2*pi int_0^infinity g(r) dr with g(r) = r*f(r), using
    Gauss-Laguerre quadrature: int_0^infinity exp(-r)*exp(r)*g(r) dr = sum_i
    w_i exp(r_i) g(r_i) where r_i and w_i are the roots and weights of
    ith-order Laguerre polynomial.

    .. WARNING: :func:`integ_radial_quad` provides a better and safer estimate.
    """

    def g(r): return r * f(r)

    sprev = 0
    errprev = N.inf
    for i in xrange(imin, imax):
        w = S.laguerre(i + 1).weights   # Laguerre polynomial roots & (total) weights
        s = N.dot(w[:, 2], g(w[:, 0]))  # sum_i (w_i exp(x_i)) g(x_i)

        err = abs(1 - s / sprev)        # Relative error wrt previous iteration
        if verbose:
            print("Order %d (%f-%f): s=%f err=%f%s" %
                  (i + 1, min(w[:, 0]), max(w[:, 0]), 2*N.pi*s, err,
                   ' !!!' if (err > errprev) else ' DONE' if (err < accuracy) else ''))
        if err < accuracy:
            break
        elif err > errprev:
            pass

        sprev = s               # Prepare next iteration
        errprev = err

    if i == imax - 1:
        raise RuntimeError("Gauss-Laguerre quadrature did not converge.")

    s *= 6.2831853071795862              # 2*pi

    return s, s*err                      # Integral and absolute error estimate


def integ_radial_quad(f, epsabs=1e-6, epsrel=1e-6):
    """Compute s = 2*pi int_0^infinity g(r) dr with g(r) = r*f(r)."""

    from scipy.integrate import quad

    s, ds = quad(lambda r: r*f(r), 0, N.inf, epsabs=epsabs, epsrel=epsrel)

    s *= 6.2831853071795862
    ds *= 6.2831853071795862

    return s, ds


def psf_ES(r, alpha, coeffs, decompose=False):
    """
    Compute (max-normalized) extract_star PSF given free parameter
    alpha and set of correlation coefficients coeffs for radius array r
    [arcsec]. If decompose=True, return individual components of PSF
    (gaussian + moffat).
    """

    spx = 0.43                          # Spaxel size [arcsec]

    name, s1, s0, b1, b0, e1, e0 = coeffs
    sigma = s0 + s1 * alpha
    beta = b0 + b1 * alpha
    eta = e0 + e1 * alpha
    # print("PSF %s: sigma,beta,eta =" % name, sigma, beta, eta)
    r2 = (r / spx)**2                     # (Radius [spx])**2
    gaussian = N.exp(-r2 / 2 / sigma**2)
    moffat = ( 1 + r2 / alpha**2 )**(-beta)

    s = eta * gaussian + moffat
    if decompose:
        return eta * gaussian / s.max(), moffat / s.max()
    else:
        return s / s.max()


def psf_ES_long(r, alpha, decompose=False):

    return psf_ES(r, alpha,
                  ['long', 0.215, 0.545, 0.345, 1.685, 0.0, 1.04],
                  decompose=decompose)


def psf_ES_short(r, alpha, decompose=False):

    return psf_ES(r, alpha,
                  ['short', 0.2, 0.56, 0.415, 1.395, 0.16, 0.6],
                  decompose=decompose)


if __name__ == '__main__':

    import fractions
    import matplotlib.pyplot as P

    # Color names
    RED, GREEN, BLUE = 'C3', 'C2', 'C0'

    ftel = 22.5                          # Telescope focal length [m]
    pixsize = 15e-6                      # Pixel size [m]
    px2arcs = pixsize / ftel * RAD2ARCS  # Pixel scale [arcsec/px]
    print("Focal length=%.2f m, pixel size=%.0f microns, scale=%.3f\"/px" %
          (ftel, pixsize*1e6, px2arcs))

    amax = 10.                          # Image half width [arcsec]
    xmax = int(amax / px2arcs)          # Image half width [px]
    amax = xmax * px2arcs               # Image half width [arcsec]
    print("Image half width: %d px = %.2f\"" % (xmax, amax))

    x = N.arange(-xmax, xmax+0.1, dtype='d')  # (n,) [px]
    r = N.hypot(*N.meshgrid(x, x))      # Radius 2D-array (n, n) [px]
    a = r * px2arcs                     # Radius (n, n) [arcsec]
    px2f = RAD2ARCS / px2arcs / len(x)  # Minimal spatial freq. [1/rad]
    f = r * px2f                        # Spatial freq. 2D-array (n, n) [1/rad]

    # All 2D-radii up to 5 arcsec
    ar = a.ravel()                      # (n**2,)
    j = N.argsort(ar)
    i = j[:N.searchsorted(ar[j], 5)]    # ar[i]: r up to 5 arcsec

    # Reference medium seeing
    lref = 5e-7                         # Reference wavelength [m]
    r0ref = 0.10                        # Fried parameter at ref. lbda [m]
    expo = 5 / 3                        # Kolmogorov index (standard is 5/3)

    # Medium seeing at 5000A
    lbda = 5e-7                         # Current wavelength [m]
    r0m = friedParamater(lbda, r0ref, lref, expo)
    s = seeing_fwhm(lbda, r0ref, lref, expo,
                    verbose=True)       # Seeing FWHM [arcsec]

    a1 = N.logspace(-1, N.log10(5))     # Radius (n,) [arcsec]
    a1 = N.concatenate(([0], a1))

    # Radial profile: comparison of methods
    if True:
        psf0 = psf2D_FFT(otf_Kolmogorov(lbda*f, r0=r0m, expo=expo))
        psf1 = psf_Kolmogorov_FFT_interp(4 * a1.max(), lbda,
                                         pxscale=px2arcs, r0=r0m, expo=expo)(a1)
        psf2 = psf_Kolmogorov_Hankel(a1, lbda, r0=r0m, expo=expo)
        psf3 = psf_Kolmogorov_Hankel_interp(a1.max(), lbda, r0=r0m, expo=expo)(a1)

        fig = P.figure()
        ax = fig.add_subplot(1, 1, 1,
                             title=u"Kolmogorov PSF (r₀=%.0f cm@%.0f Å)" %
                             (r0m*100, lbda*1e10),
                             xscale='log', xlabel="r [arcsec]",
                             yscale='log', ylabel="Normalized flux")
        ax.plot(ar[i], psf0.ravel()[i],
                ls='None', marker='.', ms=5, label='2D-FFT')
        l2, = ax.plot(a1, psf1, label='2D-FFT (interp.)')
        l1, = ax.plot(a1, psf2, label='Hankel')
        l3, = ax.plot(a1, psf3, label='Hankel (interp.)')
        ax.axhline(0.5, c='0.8')  # FWHM
        ax.axvline(s/2, c='0.8', label=u'HWHM=%.2f"' % (s/2))
        ax.set_ylim(psf1.min()/1.5, psf1.max()*1.5)
        ax.legend(loc='best', fontsize='small')

    # Plots ==============================

    psf = psf_Kolmogorov_Hankel_interp(a.max(), lbda, r0=r0m, expo=expo)(a)

    print("Total flux (2D-sum):", psf.sum() * px2arcs**2)
    print("Total flux (quad): ",
          integ_radial_quad(
              lambda r: psf_Kolmogorov_Hankel(r, lbda, r0=r0m, expo=expo)))
    # PSF flux
    # print("Total flux (Gauss-Laguerre+Hankel):",
    #       integ_radial_laguerre(
    #           lambda r: psf_Kolmogorov_Hankel(r, lbda, r0=r0m, expo=expo),
    #           verbose=True))

    # Spatial profile
    if True:
        fig = P.figure()
        axI = fig.add_subplot(1, 1, 1,
                              title=u"Kolmogorov PSF (r₀=%.0f cm@%.0f Å)" %
                              (r0m*100, lbda*1e10),
                              xlabel='x [arcsec]',
                              ylabel='y [arcsec]',
                              aspect='equal')
        im = axI.contourf(psf,
                          norm=P.matplotlib.colors.LogNorm(vmin=(psf[psf > 0]).min()),
                          interpolation='nearest',
                          extent=[-amax, amax, -amax, amax])
        fig.colorbar(im, ax=axI)

    # Radial profile ------------------------------

    if True:
        fig = P.figure()
        axR = fig.add_subplot(1, 1, 1,
                              title=u"Kolmogorov seeing radial profile "
                              u"(r₀=%.0f cm@%.0f Å)" %
                              (r0ref*1e2, lref*1e10),
                              xlabel='r [arcsec]',  # xscale='log',
                              ylabel='Normalized flux', yscale='log')

        lbda = 5e-7                         # Medium seeing @5000A
        r0 = friedParamater(lbda, r0ref, lref)
        s = seeing_fwhm(lbda, r0ref, lref)
        axR.plot(a1, psf_Kolmogorov_Hankel(a1, lbda, r0=r0),
                 color=GREEN, ls='-', marker='None',
                 label=u'%.0f Å: seeing=%.2f"' % (lbda*1e10, s))

        lbda = 3e-7                         # Medium seeing @3000A
        r0 = friedParamater(lbda, r0ref, lref)
        s = seeing_fwhm(lbda, r0ref, lref)
        axR.plot(a1, psf_Kolmogorov_Hankel(a1, lbda, r0=r0),
                 color=BLUE, ls='-', marker='None',
                 label=u'%.0f Å: seeing=%.2f"' % (lbda*1e10, s))

        lbda = 10e-7                        # Medium seeing @10000A
        r0 = friedParamater(lbda, r0ref, lref)
        s = seeing_fwhm(lbda, r0ref, lref)
        axR.plot(a1, psf_Kolmogorov_Hankel(a1, lbda, r0=r0),
                 color=RED, ls='-', marker='None',
                 label=u'%.0f Å: seeing=%.2f"' % (lbda*1e10, s))

        lbda = 5e-7
        r0ref = 0.15                        # Good seeing @5000A
        r0 = friedParamater(lbda, r0ref, lref)
        s = seeing_fwhm(lbda, r0ref, lref)
        axR.plot(a1, psf_Kolmogorov_Hankel(a1, lbda, r0=r0, expo=expo),
                 color=GREEN, ls=':', marker='None',
                 label=u'r₀=%.0f cm: seeing=%.2f"' % (r0ref*100, s))

        r0ref = 0.07                        # Bad seeing @5000A
        r0 = friedParamater(lbda, r0ref, lref)
        s = seeing_fwhm(lbda, r0ref, lref)
        axR.plot(a1, psf_Kolmogorov_Hankel(a1, lbda, r0=r0, expo=expo),
                 color=GREEN, ls='--', marker='None',
                 label=u'r₀=%.0f cm: seeing=%.2f"' % (r0ref*100, s))

        axR.set_ylim(1e-4, 1.5)
        axR.legend(loc='best', fontsize='small')

    # Comparison w/ extract_star PSF ------------------------------

    if True:
        lbda = 5e-7
        r0ref = 0.10
        s = seeing_fwhm(lbda, r0ref, lref)

        fig = P.figure()
        ax = fig.add_subplot(1, 1, 1,
                             title=u"Kolmogorov (r₀=%.0f cm@%.0f Å) "
                             "vs. ad-hoc seeing PSF" % (r0ref*1e2, lref*1e10),
                             xlabel='r [arcsec]',  # xscale='log',
                             ylabel='Normalized flux', yscale='log')

        # Long exposures
        gl, ml = psf_ES_long(a1, 2.03, decompose=True)
        ax.plot(a1, gl + ml, color=BLUE, ls='-', marker='None',
                label="Ad-hoc PSF (long)")
        ax.set_autoscale_on(False)
        ax.plot(a1, gl, color=BLUE, ls='--', marker='None')
        ax.plot(a1, ml, color=BLUE, ls='--', marker='None')

        # Short exposures
        gs, ms = psf_ES_short(a1, 2, decompose=True)
        ax.plot(a1, gs + ms, color=RED, ls='-', marker='None',
                label="Ad-hoc PSF (short)")
        ax.plot(a1, gs, color=RED, ls='--', marker='None')  # [Individual comp.]
        ax.plot(a1, ms, color=RED, ls='--', marker='None')

        # Kolmogorov
        ax.plot(a1, psf_Kolmogorov_Hankel(a1, lbda, r0=r0m, expo=5/3),
                color=GREEN, ls='-', lw=2, marker='None',
                label=u'%.0f Å: seeing=%.2f"' % (lbda*1e10, s))

        for n in range(8, 12):
            y = psf_Kolmogorov_Hankel(a1, lbda, r0=r0m, expo=n/6)
            if n != 10:         # non-standard exponent
                sel = a1 > 1.5
                ax.plot(a1[sel], y[sel], color=GREEN, ls='-', marker='None')
            sel = a1 > 2
            ax.annotate('n=%s' % (fractions.Fraction(n, 6)), (a1[sel][0], y[sel][0]),
                        fontsize='small', rotation=-30)

        ax.set_ylim(1e-4, 1.5)
        ax.legend(loc='best', fontsize='small')

    # Ad-hoc comparison w/ observations ------------------------------

    if False:
        # Long exposure @4330A: GD71_07_253_090_B.txt
        Rl, Fl = N.loadtxt("../PSF/GD71_07_253_090_B.txt", unpack=True)
        Fl /= Fl.max() / 0.8                  # /0.85
        b = 0.00353                           # Background
        ll = 4.33e-7                          # Wavelength [m]
        r0l = 0.038                           # Fried paramater

        # Short exposure @7600: HR7950_07_253_041_R.txt
        Rs, Fs = N.loadtxt("../PSF/HR7950_07_253_041_R.txt", unpack=True)
        Fs /= Fs.max() / 0.9
        ls = 7.6e-7                           # Wavelength [m]
        r0s = 0.075                           # Fried parameter

        fig = P.figure(figsize=(10, 5))
        axl = fig.add_subplot(1, 2, 1,
                              title=u"Long exposure @%.0f Å" % (ll*1e10),
                              xlabel="r [arcsec]",
                              ylabel="Normalized flux", yscale='log')
        axs = fig.add_subplot(1, 2, 2,
                              title=u"Short exposure @%.0f Å" % (ls*1e10),
                              xlabel="r [arcsec]",
                              yscale='log')

        # Long
        a = N.linspace(0, Rl.max())          # Radius (n,) [arcsec]
        axl.plot(Rl, Fl,
                 color=BLUE, ls='None', marker='.', ms=5,
                 label='Observed')
        axl.plot(a, psf_Kolmogorov_Hankel(a, ll, r0=r0l, expo=5/3) + b,
                 color=GREEN, ls='-', lw=2, marker='None',
                 label='Kolmogorov (n=5/3)', zorder=0)
        axl.plot(a, psf_Kolmogorov_Hankel(a, ll, r0=r0l, expo=3/2) + b,
                 color=GREEN, ls='-', marker='None',
                 label='Kolmogorov (n=3/2)', zorder=0)

        # Short
        a = N.linspace(0, Rs.max())          # Radius (n,) [arcsec]
        axs.plot(Rs, Fs,
                 color=RED, ls='None', marker='.', ms=5,
                 label='Observed')
        axs.plot(a, psf_Kolmogorov_Hankel(a, ls, r0=r0s, expo=5/3),
                 color=GREEN, ls='-', lw=2, marker='None',
                 label='Kolmogorov (n=5/3)', zorder=0)
        axs.plot(a, psf_Kolmogorov_Hankel(a, ls, r0=r0s, expo=3/2),
                 color=GREEN, ls='-', marker='None',
                 label='Kolmogorov (n=3/2)', zorder=0)

        axl.set_ylim(1e-4, 1.5)
        axl.legend(loc='best', fontsize='small')

        axs.set_ylim(1e-4, 1.5)
        axs.legend(loc='best', fontsize='small')

    if False:
        for i in P.get_fignums():
            fig = P.figure(i)
            fig.savefig("seeing_%d.png" % i)

    P.show()
