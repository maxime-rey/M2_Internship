"""
Functions to:
  - get adr from several headers contained in a json file
  - get adr from a dict containing a header
  - get avg, median, std, max and min of two lists
  - plot the adr along both axis and the biggest one against the wavelength
"""

import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as AC
from astropy import units as u

from pyifu import adr as ADR

__author__ = "Maxime Rey"


# ================================= #
#    Main functions (call these)    #
# ================================= #


def get_adr_from_hdrs(fitsfile, lbda_ref, lbdas, telescope='CTIO'):
  """ 
  Returns a list of the shifts in x and y due to ADR in arcsec. 
  lbdas is a list of wavelengths in angstrom.
  """

  geod_lat, _ = get_lat_tel(telescope)

  with open(fitsfile) as json_file:  
      headers = json.load(json_file) 

  x_shift = []
  y_shift = []

  for obs_name in headers: 
    a, b = get_adr_from_hdr(headers[obs_name], lbda_ref, lbdas, geod_lat)
    x_shift.append(a)
    y_shift.append(b)

  return np.array(x_shift), np.array(y_shift)


def print_stats(x_shift, y_shift, me_min=True, me_max=True, me_avg=True, me_median=False, me_std=False):
  """
  Returns the statistics that were input as True.
  x_shift and y_shift should each be a list of lists (containing adr).
  """

  list_xshift = abs(x_shift[:,-1]-x_shift[:,0])
  list_yshift = abs(y_shift[:,-1]-y_shift[:,0])

  if me_min:
    print_val_stat('minimum', np.min(list_xshift), np.min(list_yshift))
  if me_max:
    print_val_stat('maximum', np.max(list_xshift), np.max(list_yshift))
  if me_avg:
    print_val_stat('average', np.mean(list_xshift), np.mean(list_yshift))
  if me_median:
    print_val_stat('median', np.median(list_xshift), np.median(list_yshift))
  if me_std:
    print_val_stat('standard deviation', np.std(list_xshift), np.std(list_yshift))


def plot_adr(lbdas, x_shift, y_shift):
  """
  Plot adr in x against y (arcsec) and in wavelength against the strongest shift (x or y).
  Of course lbdas has to be a list with the same size as the x and y shift.
  """

  test_size_2list(x_shift, y_shift)
  try:
    test_size_2list(x_shift, lbdas)
  except ValueError:   ############################################# Autre moyen de faire ça (remplacer texte de l'erreur) ? M'a pas l'air intelligent mais ai pas trouvé mieux
    raise ValueError('You used an array of array, didn\'t you ? \n Please select an array of values')             # as e and e.args += machin mais si pas +, ça split par lettre

  xshift = abs(x_shift[-1]-x_shift[0])
  yshift = abs(y_shift[-1]-y_shift[0])
  shift_xminusy = xshift - yshift

  if shift_xminusy>0:                        # returns axis with biggest shift
      bigshiftaxis = x_shift
      title2 = 'Shift in lambdas against x'
  else:
      bigshiftaxis = y_shift
      title2 = 'Shift in lambdas against y'

  _, [ax1, ax2] = plt.subplots(nrows=1, ncols=2) #################################### Why often see fig, axes et pas _, axes ?

  ax1.plot(x_shift, y_shift)
  ax1.set_xlabel('centered x shift in arcsec')
  ax1.set_ylabel('centered y shift in arcsec')
  ax1.set_title('Shift in x against y')

  ax2.plot(lbdas, bigshiftaxis)
  ax2.set_xlabel('wavelength (A)')
  ax2.set_ylabel('biggest shift in arcsec')
  ax2.set_title(title2)

  plt.tight_layout()
  plt.show()




# ================================= #
#        Utilitary functions        #
# ================================= #


def get_adr_from_hdr(header_dict, lbda_ref, lbdas, latitude):
  """
  Returns shift in x and y as arrays.
  header_dict should be a dict composed of the keynames of the header and their associated value and the latitude should be geodetic.
  By default, the latitude is from CTIO, otherwise replace with astropy.coordinates.Latitude object.
  """ 

  if not isinstance(header_dict, dict):
    raise TypeError('The format of the header given is not a dict')

  test_key_in_dict(header_dict, 'DEC')
  test_key_in_dict(header_dict, 'HA')
  test_key_in_dict(header_dict, 'OUTTEMP')
  test_key_in_dict(header_dict, 'OUTPRESS')
  test_key_in_dict(header_dict, 'OUTHUM')
  test_key_in_dict(header_dict, 'AIRMASS')


  dec = AC.Angle(header_dict['DEC'], unit=u.deg)
  hour_angle = AC.Angle(header_dict['HA'], unit=u.hourangle)

  temperature = header_dict['OUTTEMP']                          # outside temp (C) 
  pressure = header_dict['OUTPRESS']                            # outside pressure (mbar)
  humidity = header_dict['OUTHUM']                              # outside humidity (%)
  airmass = header_dict['AIRMASS']                              # airmass    

  _, parangle = ADR.hadec2zdpar(hour_angle.degree, dec.degree, latitude.degree, deg=True)
  adr = ADR.ADR(airmass=airmass, parangle=parangle, temperature=temperature,
                pressure=pressure, lbdaref=lbda_ref, relathumidity=humidity)

  arcsecshift = adr.refract(0, 0, lbdas)
  
  x_shift = (arcsecshift[0])
  y_shift = (arcsecshift[1])


  return x_shift, y_shift


def get_lat_tel(telescope):
  """
  Returns the latitude of the telescope given as a string.
  """

  if telescope=='CTIO':
    geod_lat = AC.Latitude('-30:10:07.90', unit=u.deg)     #From CTIO doc: http://www.ctio.noao.edu/noao/node/2085
    geod_lon = AC.Angle('-70:48:23.86', unit=u.deg)
    return geod_lat, geod_lon
  elif telescope=='Auxtel':
    raise ValueError('not documented yet (you can add the value in the code')


def print_val_stat(string, stat_value1, stat_value2): ######### est ce que super mauvaise idée ? Pour pas recopier
  """
  Prints the statistics wanted explicitely.
  """
  
  print(str(string) + ' adr in x = {:.3f} arcsec'.format(stat_value1))
  print(str(string) + ' adr in y = {:.3f} arcsec \n'.format(stat_value2))




# ================================= #
#          Test functions           #
# ================================= #


def test_size_2list(list1, list2): ############# How do I return the name of the list ?
  """
  Check that 2 lists have the same size.
  """

  if len(list1)!=len(list2):
    raise ValueError("{} and {} don't have the same length".format(str(list1), str(list2)))


def test_key_in_dict(dict,key):
  """
  Check that keys are indeed in a dictionnary.
  """

  try:
    dict[key]
  except KeyError:
    raise KeyError('{} is not contained in the fits files and is necessary'.format(key))
