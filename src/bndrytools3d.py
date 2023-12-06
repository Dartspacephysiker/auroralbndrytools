"""
This module contains classes and stuff for working with auroral boundaries. It is an implementation of the equations derived by SMH in Appendix A of Ohma et al (2023, https://essopenarchive.org/doi/full/10.22541/essoar.169447428.84472457/v1)

MIT License

Copyright (c) 2023 Spencer Mark Hatch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.misc import derivative
from datetime import datetime
from apexpy import Apex
from .mlt_utils import mlt_to_mlon #mlt_utils from K. Laundal's pyAMPS package
from .utils import subsol
import warnings

y2frac = lambda dt: dt.year+dt.timetuple().tm_yday/365.25

lt2rad = 15*np.pi/180
from pysymmetry.constants import RE
from pysymmetry.utils.spherical import geo_dist
# SMALL = np.finfo(float).eps
SMALL = 1e-15

avg = lambda x: (x[1:]+x[0:-1])/2

VALID_COORD_SYS = ['Apex','geodetic']
VALIDINTERPMETHODS = ['savgol','polyfit','npinterp']
class ABoundary(object):

    def __init__(self,
                 mltp,
                 mlatp,
                 t=datetime(2000,10,28,11,50,0),
                 n_interp_pts=201,
                 interpmethod='savgol',
                 savgol_npts=31,
                 fit_order=2,
                 coordsys='geodetic',
                 dmlt=0.05,
                 apex_refh=110,
                 enable_mlt_safety_check=True,
                 discontinuity_MLT=24,
                 verbose=True):
        """
        INPUTS
        =======
        mltp          : MLT  points along auroral boundary
        mlatp         : MLat points along auroral boundary

        KEYWORDS
        ========
        dmlt          : The angle difference (in HOURS MLT) to be used for 
                        calculating the derivative of the boundary latitude
                        wrt boundary MLT/longitude.
                        (NOTE: The functions created by make_aur_bdry_funcs
                         —tauf, dtaudphif, normvecf, tanvecf—all take RADIANS
                         as input)

        TODO
        ======
        Ensure that mlatp is a monotonic function of mltp!


        AUTHOR
        =====
        S. M. Hatch
        Mar 2022
        """

        assert coordsys in VALID_COORD_SYS,"'coordsys' must be one of ['Apex','geodetic']"
        assert isValidLT(mltp),"All mltp must be between 0 and 24!"
        assert isValidLat(mlatp),"All mlatp must be between 0 and 90 (haven't tested SH)!"
        

        self._coordinate_sys = ''  # Let self.set_coordinate_system set this variable 
        self.t = t

        self.dropbadMLTs = True

        # Apex stuff
        self.apex_refh      = apex_refh
        self.apexobj        = Apex(date=self.t,refh=self.apex_refh)
        self.apexlatmlt2geo = lambda mlats,mlts, t=self.t, refh=self.apex_refh, a=self.apexobj: apexlatmlt2geo(mlats,mlts,t,refh=refh,a=a)

        self.discontinuity_MLT     = discontinuity_MLT
        self.do_convert_anglerange = not any([np.isclose(discontinuity_MLT,candidate) \
                                              for candidate in [0,24]])

        self.doCheckMLTSafety = enable_mlt_safety_check

        self.interpmethod  = interpmethod
        self.n_interp_pts  = n_interp_pts
        self.savgol_npts   = savgol_npts
        self.fit_order     = fit_order
        self.dphi          = dmlt*lt2rad

        gdlatp, glonp = self.apexlatmlt2geo(mlatp,mltp)
        # and local time
        gltp          = get_localtime([self.t]*len(mltp), glonp,
                                      return_dts_too=False,
                                      verbose=False)
            
        assert isValidLT(gltp),"All gltp must be between 0 and 24!"

        # Get discontinuity LT
        if self.do_convert_anglerange:
            gdlat_d, glon_d = self.apexlatmlt2geo([70],[self.discontinuity_MLT])
            glt_d = get_localtime([self.t], glon_d,
                                 return_dts_too=False,
                                 verbose=False)
            self.discontinuity_LT = glt_d[0]

        mltp = self._convert_mlt_range(mltp)
        gltp = self._convert_lt_range(gltp)

        self._mltp = mltp
        self._mlatp = mlatp
        self._gdlatp = gdlatp
        self._glonp = glonp
        self._gltp = gltp

        magdict = get_smooth_func_and_info(self._mltp,self._mlatp,
                                           n_interp_pts=self.n_interp_pts,
                                           interpmethod=self.interpmethod,
                                           savgol_npts=self.savgol_npts,
                                           fit_order=self.fit_order,
                                           verbose=verbose)
        # assert 1<0,"YOU NEED TO CLEAN UP CODE HERE; GET RID OF SMOOTHING IN GEO COORDINATES PERMANENTLY"
        import copy
        geodict = copy.deepcopy(magdict)
        
        gdlatsmooth, glonsmooth = self.apexlatmlt2geo(magdict['latsmooth'],magdict['ltinterp'])
        geodict['latsmooth'] =gdlatsmooth
        geodict['ltinterp'] = glonsmooth/15

        geodict['ltinterp'] = self._convert_lt_range(geodict['ltinterp'])

        # geodict = get_smooth_func_and_info(self._gltp,self._gdlatp,
        #                                    n_interp_pts=self.n_interp_pts,
        #                                    interpmethod=self.interpmethod,
        #                                    savgol_npts=self.savgol_npts,
        #                                    fit_order=self.fit_order,
        #                                    verbose=verbose)

        self._magdict = magdict
        self._geodict = geodict

        mtauf, mdtaudphif, mnormvecf, mtanvecf = make_aur_bdry_funcs(self._magdict['ltinterp'],
                                                                     self._magdict['latsmooth'],
                                                                     dphi=self.dphi)
        gtauf, gdtaudphif, gnormvecf, gtanvecf = make_aur_bdry_funcs(self._geodict['ltinterp'],
                                                                     self._geodict['latsmooth'],
                                                                     dphi=self.dphi)
        self._magdict['tauf']      = mtauf
        self._magdict['dtaudphif'] = mdtaudphif
        self._magdict['normvecf']  = mnormvecf
        self._magdict['tanvecf']   = mtanvecf

        self._geodict['tauf']       = gtauf
        self._geodict['dtaudphif']  = gdtaudphif
        self._geodict['normvecf']   = gnormvecf
        self._geodict['tanvecf']    = gtanvecf

        self.set_coordinate_system(coordsys)

    def _convert_mlt_range(self,mlts):
        if self.do_convert_anglerange:
            mlts = np.where(mlts > self.discontinuity_MLT,mlts-24,mlts)
        return mlts

    def _convert_lt_range(self,lts):
        if self.do_convert_anglerange:
            lts = np.where(lts > self.discontinuity_LT,lts-24,lts)
        return lts

    def _convert_mlts_to_lts_if_geodetic(self,mlts):

        # Don't need to do convert_mlt_range and check_valid_mlts in this method.
        # They should be done BEFORE calling this method!!!
        # mlts = self._convert_mlt_range(mlts)
        # self._check_valid_mlts(mlts)

        if self._coordinate_sys == 'geodetic':

            mlats = self.get_boundary_mlat(mlts)
            gdlats, glons = self.apexlatmlt2geo(mlats,mlts)
            lts = get_localtime([self.t]*len(mlts),glons)

            return lts
        else:
            return mlts

    def _get_dl(self,mlt0,mlt1,mlat0,mlat1):
        """
        Get distance between two points in Apex coordinates
        by converting MLTs and MLats to glon and gdlat        
        """
        return get_dl(mlt0,mlt1,mlat0,mlat1,self.t,self.t,
                      refh=self.apex_refh)

    def _check_valid_mlts(self,mlts):

        if self.doCheckMLTSafety:
            if self.dropbadMLTs:
                mlts = mlts[in_angle_interval(mlts,
                                              self._magdict['ltmin'],
                                              self._magdict['ltmax'],
                                              degree=False,
                                              mlt=True)]
            else:
                assert np.all(in_angle_interval(mlts,
                                                self._magdict['ltmin'],
                                                self._magdict['ltmax'],
                                                degree=False,
                                                mlt=True)), \
                                                f"Some LTs are outside range ltmin={self._magdict['ltmin']:.2f} "+\
                                                f"and ltmax={self._magdict['ltmax']:.2f} "+\
                                                f"(coordsys={self._coordinate_sys})! This isn't going to work ..."

    def set_coordinate_system(self,coordsys):
        """
        Set coordinate system for velocity calculation
        """
        assert coordsys in VALID_COORD_SYS,\
        "'coordsys' must be one of ['Apex','geodetic']"

        if self._coordinate_sys == coordsys:
            return

        self._coordinate_sys = coordsys

        if coordsys == 'Apex':
            self.lt          = self._mltp
            self.lat         = self._mlatp
            self.ltmin       = self._magdict['ltmin']    
            self.ltmax       = self._magdict['ltmax']    
            self._ltinterp   = self._magdict['ltinterp'] 
            self._latinterp  = self._magdict['latinterp']
            self._latsmooth  = self._magdict['latsmooth']
            self.tauf        = self._magdict['tauf']
            self.dtaudphif   = self._magdict['dtaudphif']
            self.normvecf    = self._magdict['normvecf']
            self.tanvecf     = self._magdict['tanvecf']

        elif coordsys == 'geodetic':
            self.lt          = self._gltp
            self.lat         = self._gdlatp
            self.ltmin       = self._geodict['ltmin']    
            self.ltmax       = self._geodict['ltmax']    
            self._ltinterp   = self._geodict['ltinterp'] 
            self._latinterp  = self._geodict['latinterp']
            self._latsmooth  = self._geodict['latsmooth']
            self.tauf        = self._geodict['tauf']
            self.dtaudphif   = self._geodict['dtaudphif']
            self.normvecf    = self._geodict['normvecf']
            self.tanvecf     = self._geodict['tanvecf']

    def get_dl_list(self,mlts):
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        mlats = self.get_boundary_mlat(mlts)
        return self._get_dl(mlts[:-1],mlts[1:],
                            mlats[:-1],mlats[1:])

    def toggle_mlt_safety_check(self):
        stri = 'OFF' if self.doCheckMLTSafety else 'ON'
        print(f"Turning MLT safety check {stri}")
        self.doCheckMLTSafety = ~self.doCheckMLTSafety

    ########################################
    # Getters for quantities in Apex coordinates

    def get_boundary_mlat(self,mlts):
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        return 90-np.rad2deg(self._magdict['tauf'](mlts*lt2rad))

    def get_normvec_apex(self,mlts):
        """
        Get normal vector in Apex "ENU" coordinates
        Shape is [N,3], where N is the number of mlts provided by user
        Component 0 is "East"  (parallel to lines of constant MLat)
        Component 1 is "North" (parallel to lines of constant MLT)
        Component 2 is "Up"    (parallel to field lines?)
        """
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        return sphvec_to_ENU(np.vstack(self._magdict['normvecf'](mlts*lt2rad)).T)

    def get_tanvec_apex(self,mlts):
        """
        Get tangent vector in Apex "ENU" coordinates
        Shape is [N,3], where N is the number of mlts provided by user

        For a description of Apex "ENU" coordinates, see description of 
        ABoundary.get_normvec_apex.
        """
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        return sphvec_to_ENU(np.vstack(self._magdict['tanvecf'](mlts*lt2rad)).T)
        
    def get_dl_vector_apex(self,mlts,loc='center'):
        """
        Get tangent vectors pointing from one point to another along the auroral
        boundary in Apex "ENU" coordinates, with vector magnitude the 
        distance between each point.

        For a description of Apex "ENU" coordinates, see description of 
        ABoundary.get_normvec_apex.

        NOTE: This means that this function returns ONE FEWER vectors
        than the number of MLTs provided by the user!
        """                

        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        dls = self.get_dl_list(mlts)
        tanvecs = self.get_tanvec_apex(get_shifted_mlt(mlts,loc=loc))
        return dls[:,np.newaxis]*tanvecs

    ########################################
    # Getters for quantities in geographic(-ish?) coordinates

    def get_boundary_gdlatlt(self,mlts):
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)

        mlats = self.get_boundary_mlat(mlts)
        gdlats, glons = self.apexlatmlt2geo(mlats,mlts)
        lts = get_localtime([self.t]*len(mlts),glons)

        return gdlats,lts

    def get_boundary_gdlatlon(self,mlts):
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        gdlats, glons = self.apexlatmlt2geo(self.get_boundary_mlat(mlts),mlts)
        return gdlats, glons

    def get_fake_apex_east_north_basis_vectors(self,mlts):
        gdlats, glons = self.get_boundary_gdlatlon(mlts)

        # Get apex basis vectors
        # These basis vectors have only two components: geodetic east and geodetic north
        f1, f2 = self.apexobj.basevectors_qd(gdlats,glons,self.apex_refh,coords='geo')

        # normalize f1
        f1 = f1/np.sqrt(np.sum(f1**2,axis=0))
        f1 = np.r_[f1,[np.zeros(f1.shape[1])]].T  # Add row of zeros to f1 for "up" dimension

        # Make vector that is perp to f1
        crossme = np.zeros(f1.shape)
        crossme[:,2] = 1
        f2new = np.cross(f1,crossme)
        assert np.all(np.isclose(np.sum(f1*f2new,axis=1),0))  # See? They ARE orthogonal!

        return f1, f2new

    def get_normvec_geo_fakeapex(self,mlts):

        nvecs = self.get_normvec_apex(mlts)

        f1,f2 = self.get_fake_apex_east_north_basis_vectors(mlts)

        # norm_geo = f1*normal_eastward_component_apex + (k X f1)*normal_northward_component_apex
        nvecs_geo = (f1.T*nvecs[:,0] + f2.T*nvecs[:,1]).T

        return nvecs_geo

    def get_tanvec_geo_fakeapex(self,mlts):

        tvecs = self.get_tanvec_apex(mlts)

        f1,f2 = self.get_fake_apex_east_north_basis_vectors(mlts)

        # tan_geo = f1*tangent_eastward_component_apex + (k X f1)*tangent_northward_component_apex
        tvecs_geo = (f1.T*tvecs[:,0] + f2.T*tvecs[:,1]).T

        return tvecs_geo

    def get_dl_vector_geo_fakeapex(self,mlts,loc='center'):
        """
        Get tangent vectors pointing from one point to another along the auroral
        boundary in geographic ENU coordinates, with vector magnitude the 
        distance between each point.

        NOTE: This means that this function returns ONE FEWER vectors
        than the number of MLTs provided by the user!
        """
        
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        dls = self.get_dl_list(mlts)
        
        tanvecs = self.get_tanvec_geo_fakeapex(get_shifted_mlt(mlts,loc=loc))
        return dls[:,np.newaxis]*tanvecs

    def get_normvec_geo(self,mlts):
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)

        gdlats,lts = self.get_boundary_gdlatlt(mlts)
        lts = self._convert_lt_range(lts)

        return sphvec_to_ENU(np.vstack(self._geodict['normvecf'](lts*lt2rad)).T)

    def get_tanvec_geo(self,mlts):
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)

        gdlats,lts = self.get_boundary_gdlatlt(mlts)
        lts = self._convert_lt_range(lts)

        return sphvec_to_ENU(np.vstack(self._geodict['tanvecf'](lts*lt2rad)).T)

    def get_dl_vector_geo(self,mlts,loc='center'):
        """
        Get tangent vectors pointing from one point to another along the auroral
        boundary in geographic ENU coordinates, with vector magnitude the 
        distance between each point.

        NOTE: This means that this function returns ONE FEWER vectors
        than the number of MLTs provided by the user!
        """
        
        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)
        dls = self.get_dl_list(mlts)
        
        tanvecs = self.get_tanvec_geo(get_shifted_mlt(mlts,loc=loc))
        return dls[:,np.newaxis]*tanvecs

    def get_velocity(self, *args,
                     refh=110,
                     coordsys='geodetic',
                     DEBUG=False,
                     return_ENU=True,
                     uncertainty_theta_deg=None,
                     uncertainty_R_km=None,
                     unc_corr_t=None,
                     unc_corr_phi=None):
        """
        Taking this ABoundary object as t0 and the other objs as the other times, calculate 
        the boundary velocity (if possible) using an analytic expression for each lt in lts
        
        return_ENU : return the boundary velocity in ENU coordinates
        
        NOTE: If the calculation is carried out with abobj.coordinate_sys == 'Apex',
        ENU coordinates means MAGNETIC East, MAGNETIC North, and whatever completes these
        right-handed coordinate system.

        Velocity uncertainty calculation
        ====================================
        If any of the keywords beginning with "uncertainty" or "unc" are not None, then get_velocity returns a tuple.
        The first element of the tuple is the velocity components, and the second is the uncertainty of each velocity component.

        uncertainty_theta_deg: Uncertainty in the location of the boundary in degrees
        uncertainty_R_km     : Uncertainty in the altitude of the boundary in kilometers
        unc_corr_t           : The correlation coefficient between boundary location θ(phi, t=t0) and θ(phi, t=t0+dt)
        unc_corr_phi         : The correlation coefficient between a θ(phi=phi0, t) and θ(phi=phi0+dphi, t)

        AUTHOR
        =====
        S. M. Hatch
        Mar 2022

        CHANGELOG
        =========
        Jun 2023: Added velocity uncertainty calculation to get_velocity_general_analytic, included via uncertainty keyword here

        """
        
        nargs = len(args)
        if nargs < 2:
            print("Please provide at least \n"+\
                  "1. A list of MLT values at which to get the auroral boundary velocity;\n"+\
                  "2. A second auroral boundary object to use as reference for calculating the boundary velocity\n"+\
                  "Usage: get_velocity(lts, abndryobj1[, abndryobj2])")
            return dict()

        mlts = args[0]
        assert isValidLT(mlts),"All mltp must be between 0 and 24!"

        mlts = self._convert_mlt_range(mlts)
        self._check_valid_mlts(mlts)

        abobj_t0 = self

        # Set coordinate system for all abobjs
        abobj_t0.set_coordinate_system(coordsys)
        for abobj in args[1:]:
            abobj.set_coordinate_system(coordsys)
        # assert all([abobj_t0.coordinate_sys == arg.coordinate_sys for arg in args[1:]])

        # Convert to geographic LT if need be
        lts = self._convert_mlts_to_lts_if_geodetic(mlts)
        lts = self._convert_lt_range(lts)

        #print("TIME HACK!!!")
        if self.dropbadMLTs:
            lts = lts[in_angle_interval(lts,
                                        self.ltmin-1e-2,
                                        self.ltmax+1e-2,
                                        degree=False,
                                        mlt=True)]
        else:
            assert np.all(in_angle_interval(lts,self.ltmin-1e-2,self.ltmax+1e-2,degree=False,mlt=True)), \
                f"Some LTs are outside range ltmin={self.ltmin:.2f} and ltmax={self.ltmax:.2f} "+\
                f"(coordsys={self._coordinate_sys})! This isn't going to work ..."

        if nargs == 2:
            abobj1 = args[1]

            # This works for forward or backward differences, bc sign of dt changes accordingly
            dt = (abobj1.t-abobj_t0.t).total_seconds()
            abobjs = [abobj_t0,abobj1]
            coeffs = [-1,1]

        elif nargs == 3:

            abobj1 = args[1]
            abobj2 = args[2]

            if abobj1.t < abobj_t0.t:
                if abobj2.t > abobj_t0.t:
                    abobj_tm1 = args[1]
                    abobj_tp1 = args[2]

                    dt     = (abobj_tp1.t-abobj_t0.t).total_seconds()
                    abobjs = [abobj_t0, abobj_tm1, abobj_tp1]
                    coeffs = [0,-1/2,1/2]

                elif abobj2.t < abobj_t0.t:
                    print("2nd-order back diff not tested yet! Could be lots of sign errors!")

                    assert abobj1.t < abobj2.t < abobj_t0.t

                    abobj_tm2 = abobj1
                    abobj_tm1 = abobj2 
            
                    dt     = (abobj_t0.t-abobj_tm1.t).total_seconds()
                    abobjs = [abobj_t0, abobj_tm1, abobj_tm2]
                    coeffs = [3/2,-2,1/2]

            elif abobj1.t > abobj_t0.t:
                if abobj2.t > abobj_t0.t:
                    if DEBUG:
                        print("2nd-order fwd diff")
                    
                    assert abobj_t0.t < abobj1.t < abobj2.t

                    abobj_tp1 = abobj1 
                    abobj_tp2 = abobj2
            
                    assert np.isclose((abobj_tp1.t-abobj_t0.t).total_seconds(),
                                      (abobj_tp2.t-abobj_tp1.t).total_seconds())

                    dt     = (abobj_tp1.t-abobj_t0.t).total_seconds()
                    abobjs = [abobj_t0, abobj_tp1, abobj_tp2]
                    coeffs = [-3/2,2,-1/2]

        elif nargs == 4:
            assert 2<0,"Don't know what to do with only four arguments!"

        elif nargs == 5:

            if DEBUG:
                print("Fourth-order centered difference (stencil [1/12,-2/3,0,2/3,-1/12])")

            abobj_tm2 = args[1]
            abobj_tm1 = args[2]
            abobj_tp1 = args[3]
            abobj_tp2 = args[4]

            assert abobj_tm2.t < abobj_tm1.t < abobj_t0.t < abobj_tp1.t < abobj_tp2.t

            assert np.isclose((abobj_tp2.t-abobj_tp1.t).total_seconds(),
                              (abobj_tp1.t-abobj_t0.t).total_seconds()) & \
                    np.isclose((abobj_t0.t-abobj_tm1.t).total_seconds(),
                               (abobj_tm1.t-abobj_tm2.t).total_seconds())

            dt     = (abobj_tp1.t-abobj_t0.t).total_seconds()
            abobjs = [abobj_t0, abobj_tm2, abobj_tm1, abobj_tp1, abobj_tp2]
            coeffs = [0, 1/12, -2/3, 2/3, -1/12]

        # Are we doing uncertainties?
        uncertainty_info = []

        uncertainty_info.append(uncertainty_theta_deg if uncertainty_theta_deg is not None else 0.)
        uncertainty_info.append(uncertainty_R_km if uncertainty_R_km is not None else 0.)
        uncertainty_info.append(unc_corr_t if unc_corr_t is not None else 0.)
        uncertainty_info.append(unc_corr_phi if unc_corr_phi is not None else 0.)

        # Use given ABoundary objects and finite difference coefficients
        # to calculate boundary velocity
        est = get_velocity_general_analytic(lts, abobjs, coeffs, dt,
                                            refh=refh,
                                            DEBUG=DEBUG,
                                            return_ENU=return_ENU,
                                            uncertainty_info=uncertainty_info)

        return est


def get_shifted_mlt(mlts,loc='left'):
    assert loc in ['left','center','right']
    
    if loc == 'left':
        mlts = mlts[:-1]
    elif loc == 'center':
        mlts = avg(mlts)
    elif loc == 'right':
        mlts = mlts[1:]

    return mlts

def sphvec_to_ENU(vec):
    """
    Vector shape should be [N,3], where N is the number of vectors
    This assumes 
    Component 0 (vec[:,0]) is the radial component
    Component 1 (vec[:,1]) is the colatitudinal component
    Component 2 (vec[:,2]) is the azimuthal component
    """
    
    return np.vstack((vec[:,2],-vec[:,1],vec[:,0])).T


def isValidLT(lt):
    lt = np.asarray(lt)
    return np.all(lt >= 0) & np.all(lt <= 24)


def isValidLat(lat):
    lat = np.asarray(lat)
    return np.all(lat >= 0) & np.all(lat <= 90)


def in_angle_interval(x, a, b,degree=True,mlt=False):
    assert not (degree and mlt),"Select either degree==True or mlt==True!"
    if degree:
        maxAngle = 360
    elif mlt:
        maxAngle = 24
    else:
        maxAngle = 2*np.pi

    return (x - a) % maxAngle <= (b - a) % maxAngle


def get_smooth_func_and_info(lt,lat,
                             n_interp_pts=201,
                             interpmethod='savgol',
                             savgol_npts=31,
                             fit_order=2,
                             verbose=False):

    # CHECKS
    assert interpmethod in VALIDINTERPMETHODS,"'interpmethod' must be one of ['savgol','polyfit','npinterp']!"
    assert n_interp_pts < 5000 #For the sake of memory
    assert len(lt) <= n_interp_pts/4

    if interpmethod == 'savgol':
        assert savgol_npts/n_interp_pts <= 0.1,"savgol_npts is too high relative to n_interp_pts. " + \
            "You want many interpolation points for each savgol point, so either increase n_interp_pts " +\
            "or decrease savgol_npts (probably the former!)  " 


    ltmin,ltmax = lt.min(),lt.max()

    # Now we need to permute lt so that the smallest lt is at the beginning of the array
    ltargmin = lt.argmin()
    lt = np.roll(lt,-ltargmin)
    lat = np.roll(lat,-ltargmin)

    # Now ensure that things are monotonic so that we don't screw up interpolation
    if not np.all(np.diff(lt) > 0):
        warnings.warn("Couldn't get LT array into a monotonic form! This is probably trouble")
    # assert np.all(np.diff(lt) > 0),"No! Couldn't get LT array into a monotonic form!"

    # Make array of local times at which to interpolate
    ltinterp = np.linspace(ltmin,ltmax,n_interp_pts)

    if interpmethod == 'savgol':
        if verbose:
            print(f"Smoothing boundaries using Savitzky-Golay filter of order {fit_order} with {savgol_npts}-point window")

            latinterp = np.interp(ltinterp, lt, lat)
            latsmooth = savgol_filter(latinterp,savgol_npts,fit_order)

    elif interpmethod == 'polyfit':
        if verbose:
            print(f"Smoothing boundaries using polyfit of order {fit_order}")

        # poly = np.polyfit(ltinterp, latinterp, fit_order)
        poly = np.polyfit(lt, lat, fit_order)
        latsmooth = np.poly1d(poly)(ltinterp)
    elif interpmethod == 'npinterp':
        if verbose:
            print(f"Interpolating boundaries using np.interp (NO SMOOTHING)")

            latinterp = np.interp(ltinterp, lt, lat)
            latsmooth = latinterp

    return dict(ltmin=ltmin,ltmax=ltmax,ltinterp=ltinterp,
                latinterp=latinterp,latsmooth=latsmooth)


def make_aur_bdry_funcs(ltp,latp,
                        dphi=0.1*lt2rad,
                        kind='cubic',
                        fill_value='extrapolate'):
    """
    Based on a set of lt and lat points, makes functions describing 
    ->The colatitude of the auroral boundary as a function of phi(=lt*lt2rad), and 
    ->The derivative of the colatitude of the auroral boundary with respect to phi.
    ->The unit vector normal  to the boundary in the form <n_r , n_theta, n_phi>
    ->The unit vector tangent to the boundary in the form <t_r , t_theta, t_phi>


    Parameters
    ==========
    ltp       : Magnetic local times from which to create the auroral boundary function (array-like)
    latp      : Magnetic latitudes   from which to create the auroral boundary function (array-like)

    Keywords
    ==========

    dphi       : Spacing (in radians) to use when calculating derivative of tau(=90-lat) wrt phi(=lt*15)
    fill_value : What value to return when user requests lat at lts outside bounds of ltp
    """

    # Convert lat and lt to theta and phi, respectively, with theta and phi in radians
    thetap,phip = latlt_to_thetaphi(latp,ltp,degree=False)

    # Make functions
    taufunc = interp1d(phip,thetap,
                       kind=kind,
                       fill_value=fill_value)

    dtaudphifunc = lambda phi, taufunc=taufunc, dphi=dphi: derivative(taufunc,
                                                                      phi,
                                                                      dx=dphi)

    normvecfunc = lambda phi, taufunc=taufunc, dtaudphifunc=dtaudphifunc: normal_vector_spherical(phi,
                                                                                              taufunc,
                                                                                              dtaudphifunc)

    tanvecfunc = lambda phi, taufunc=taufunc, dtaudphifunc=dtaudphifunc: tangent_vector_spherical(phi,
                                                                                                  taufunc,
                                                                                                  dtaudphifunc)

    return taufunc,dtaudphifunc,normvecfunc,tanvecfunc


def normal_vector_spherical(phi,tau,dtaudphi):
    """
    This function returns the radial (or colatitudinal) and azimuthal components
    of the normal vector for an implicitly defined surface f(θ,φ) = θ-τ(φ) == 0 in spherical coordinates.

    The normal vector is given by the gradient of f: 
    ∇f/|∇f| = <  0, ∂f/∂θ, 1/(rsinθ) *∂f/∂φ  > / |∇f| 
            = <  0, 1/r  , -1/(rsinθ)* ∂τ/∂φ > /  [(1/r) * sqrt(1+csc²(φ)(∂τ/∂φ)²)
            = <  0, 1    , -1/ sinθ  * ∂τ/∂φ > /           sqrt(1+csc²(φ)(∂τ/∂φ)²

    # 

    phi       : Azimuthal angle(s) (in RADIANS) where we evaluate normal vector
    tau       : Vector-compatible function of phi of form   f(phi), return value must be in RADIANS
    dtaudphi  : Vector-compatible function of phi of form   g(phi), return value must be in RADIANS

    NOTE: 
    The theta component is the colatitude (or southward) component and 
    The phi   component is the azimuthal  (or eastward ) component.
    We use the convection that SOUTHWARD (the colatitudinal direction) is positive; that is, 
    we insist that the colatitudinal component always be positive (if it is not zero).
    

    WARNING:
    dtaudphi must be the derivative of the tau function with phi in RADIANS, NOT degrees! 
    Otherwise you'll get the wrong answer here (i.e., I think the variable "der" would need 
    to be multiplied by a factor of 180/pi or something like that)

    RETURNS
    ======
    (r component, tau component, phi component)

    AUTHOR
    =====
    S. M. Hatch
    Mar 2022
    """
    
    theta = tau(phi)               # Get colatitudes
    der = -dtaudphi(phi)/np.sin(theta) # Get deriv. of f wrt φ

    vnorm = 1/np.sqrt(1+der**2)# Get norm of gradient

    rc = vnorm*0
    tc = vnorm                 # t/colatitude/southward component
    pc = der*vnorm             # phi/eastward component

    rc = np.asarray(rc)
    tc = np.asarray(tc)
    pc = np.asarray(pc)

    # Ensure colatitudinal(/southward) component is positive
    flips = tc < 0
    rc[flips] = -1*rc[flips]
    tc[flips] = -1*tc[flips]
    pc[flips] = -1*pc[flips]

    return rc,tc,pc


def tangent_vector_spherical(phi,tau,dtaudphi):
    """
    This function returns the colatitudinal and azimuthal components
    of the tangent vector for an implicitly defined surface f(θ,φ) = θ-τ(φ) == 0 in spherical coordinates.

    The tangent vector is given by T = R'/|R'|, where R = r < sin(τ)cos(φ),sin(τ)sin(φ),cos(τ) > in Cartesian coordinates.
    Carrying out the calculation for R'/|R'| and converting to spherical coordinates, we get

    T = < 0, dτ/dφ, sin(τ) > / sqrt(sin²(τ)+(dτ/dφ)²)

    with the second and third positions being the colatitudinal and azimuthal components, respectively.

    Here we insist that the azimuthal component always be positive (if it is not zero)
    # 

    phi       : Azimuthal angle(s) (in RADIANS) where we evaluate normal vector
    tau       : Vector-compatible function of phi of form   f(phi), return value must be in RADIANS
    dtaudphi  : Vector-compatible function of phi of form   g(phi), return value must be in RADIANS

    NOTE: 
    The theta component is the colatitude (or southward) component and 
    The phi   component is the azimuthal  (or eastward ) component.
    We use the convection that EASTWARD (the azimuthal direction) is positive

    WARNING:
    dtaudphi must be the derivative of the tau function with phi in RADIANS, NOT degrees! 
    Otherwise you'll get the wrong answer here (i.e., I think the variable "der" would need 
    to be multiplied by a factor of 180/pi or something like that)

    RETURNS
    ======
    (r component, tau component, phi component)

    AUTHOR
    =====
    S. M. Hatch
    Mar 2022
    """

    theta = tau(phi)               # Get colatitudes
    der = dtaudphi(phi)            # Get deriv. of τ wrt φ

    vnorm = 1/np.sqrt(np.sin(theta)**2+der**2)# Get vector norm 

    rc = vnorm*0             # radial component
    tc = der*vnorm           # colatitude/southward component
    pc = np.sin(theta)*vnorm # phi/eastward component

    rc = np.asarray(rc)
    tc = np.asarray(tc)
    pc = np.asarray(pc)

    # Ensure azimuthal(/eastward) component is positive
    flips = pc < 0
    rc[flips] = -1*rc[flips]
    tc[flips] = -1*tc[flips]
    pc[flips] = -1*pc[flips]

    return rc,tc,pc


def sphericalvec_to_cartesian(phi,theta,v_r,v_theta,v_phi,degree=False):
    """
    Conversion between spherical and Cartesian coords requires specification of angles, så klart.

    Input:
    phi     : Angle (in RADIANS by default) at which the vector v finds itself
    theta   : Angle (in RADIANS by default) at which the vector v finds itself
    v_r     : Component of vector v in radial direction
    v_theta : Component of vector v in colatitudinal direction
    v_phi   : Component of vector v in azimuthal direction

    Returns the vector v in Cartesian coordinates = <x component, y component, z component>
    """

    if degree:
        phir = np.deg2rad(phi)
        thetar = np.deg2rad(theta)
    else:
        phir = phi
        thetar = theta

    xr = np.sin(thetar)*np.cos(phir)
    yr = np.sin(thetar)*np.sin(phir)
    zr = np.cos(thetar)

    xt = np.cos(thetar)*np.cos(phir)
    yt = np.cos(thetar)*np.sin(phir)
    zt = -np.sin(thetar)

    xp = -np.sin(phir)
    yp =  np.cos(phir)
    zp = 0

    xcomp = v_r * xr + v_theta * xt + v_phi * xp
    ycomp = v_r * yr + v_theta * yt + v_phi * yp
    zcomp = v_r * zr + v_theta * zt + v_phi * zp

    return (xcomp, ycomp, zcomp)


def thetaphi_to_latlt(theta,phi,degree=False,DEBUG=False):
    if degree:
        lat = 90-theta
        lt = phi/15
    else:
        lat = 90-np.rad2deg(theta)
        lt = phi/lt2rad

    swaps = lat > 90

    if DEBUG:
        olds = lat[swaps].copy(),lt[swaps].copy()

    lat[swaps] = 180-lat[swaps]
    lt[swaps]  = (lt[swaps]+12)%24

    # Make sure it worked
    if DEBUG:
        for i in range(len(olds[0])):
            print(f'{olds[0][i]:.2f},{lat[np.where(swaps)[0][i]]:.2f}')

    return lat,lt


def latlt_to_thetaphi(lat,lt,degree=False):

    theta = np.deg2rad(90-lat)
    phi = lt*lt2rad
    return theta,phi


def get_dl(mlt0,mlt1,mlat0,mlat1,t0,t1,
           refh=110):
    """
    Get distance between two points in Apex coordinates
    by converting MLTs and MLats to glon and gdlat
    """

    a = Apex(date=t0,refh=refh)
    
    mlon0 = mlt_to_mlon(mlt0,[t0]*len(mlt0),y2frac(t0))
    mlon1 = mlt_to_mlon(mlt1,[t1]*len(mlt1),y2frac(t1))
        
    gdlat0, glon0, _ = a.apex2geo(mlat0,mlon0,refh)
    gdlat1, glon1, _ = a.apex2geo(mlat1,mlon1,refh)
        
    # Distance between points in m
    dls = geo_dist(glon0, gdlat0,
                   glon1, gdlat1,
                   deg=True,
                   rearth=6.370949e3,
                   altitude=refh,
                   haversine=True)*1000

    return dls


def get_dl_geo(glon0,glon1,gdlat0,gdlat1,
               refh=110):
    """
    Get distance between two points in geodetic coordinates
    """

    # Distance between points in m
    dls = geo_dist(glon0, gdlat0,
                   glon1, gdlat1,
                   deg=True,
                   rearth=6.370949e3,
                   altitude=refh,
                   haversine=True)*1000

    return dls


def apexlatmlt2geo(mlats,mlts,t,refh=110,a=None):
    """
    Given set of mlts, mlats, return geodetic lats and lons
    """
    if a is None:
        a = Apex(date=t,refh=refh)

    mlons = mlt_to_mlon(mlts,[t]*len(mlts),y2frac(t))
    gdlats, glons, _ = a.apex2geo(mlats,mlons,refh)

    return gdlats, glons


# def mlatmlt_to_gdlatlt(mlats,mlts,t,refh=110):

#     gdlats, glons = apexlatmlt2geo(mlats,mlts,t,refh=refh)

#     lts = get_localtime([t]*len(mlts),glons)
    
#     return gdlats,lts


def get_localtime(dts, glons,
                  return_dts_too=False,
                  verbose=False):
    """
    #TEST:
    glons = np.r_[0:361:1]
    dts = np.array([datetime(2000,1,1,12)]*len(glons))
    lts = get_localtime(dts,glons)
    """
    if not hasattr(dts,'__iter__'):
        dts = pd.DatetimeIndex([dts])

    elif not hasattr(dts,'hour'):
        dts = pd.DatetimeIndex(dts)

    if not hasattr(glons,'__iter__'):
        glons = np.array(glons)

    sslat, sslon = map(np.ravel, subsol(dts))
    # LTs = ((glons - sslon + 180) % 360 - 180) / 15
    LTs = ((glons - sslon + 180) % 360) / 15

    midnightlongitude = (sslon - 180.) % 360
    if return_dts_too:
        return LTs, dts, midnightlongitude, glons
    else:
        return LTs

def get_glon_from_lt(dts, lts):
    """

    """

    if not hasattr(dts,'__iter__'):
        dts = pd.DatetimeIndex([dts])

    elif not hasattr(dts,'hour'):
        dts = pd.DatetimeIndex(dts)

    if not hasattr(lts,'__iter__'):
        glons = np.array(lts)

    sslat, sslon = map(np.ravel, subsol(dts))
    
    try:
        #code that generates warning
        glons = ((lts * 15) +sslon-180) % 360
    except:
        #put a breakpoint here
        print("DAD")
        breakpoint()

    return glons


def get_velocity_general_analytic(lts, abobjs, coeffs, dt, refh=110,
                                  DEBUG=False,
                                  return_ENU=True,
                                  uncertainty_info=[],
):
    """
    Get boundary normal velocity in spherical (<r, theta, phi>) coordinates

    INPUTS
    =========

    lts              : List/array of local times at which to perform calculation
    abobjs           : List of ABoundary objects
    coeffs           : Finite-difference stencil coefficients, list of same length as abobjs

    uncertainty_info : List of at most two elements, with first and second element respectively Dtheta_deg and DR_km.
            Dtheta_deg - the uncertainty of the boundary in degrees. Can be a scalar or an array with same dimensions as lts
            DR_km      - altitude uncertainty in km (default: 0)
            rho_t      - Correlation coefficient between boundary at t=t_0 and t=t_1
            rho_lon    - Correlation coefficient between boundary at φ=φ_0 and φ=φ_1

        
    For velocity calculation, see 'Analytic boundary velocity in spherical coordinates.pdf'

    For velocity uncertainty calculation, see '[DOCUMENT THAT I HAVEN'T PREPARED].pdf'

    AUTHOR
    =========
    S. M. Hatch
    Mar 2022

    CHANGELOG
    =========
    Jun 2023: Added velocity uncertainty calculation

    """

    assert len(abobjs) == len(coeffs) & len(abobjs) > 1
    assert ~np.isclose(dt,0),"dt is approx. zero! This won't work ..."

    assert len(uncertainty_info) <= 4,"'uncertainty_info' list should be no more than four elements long: uncertainty_info = [Dtheta_deg, DR_km, rho_t, rho_lon]"

    print("UNCERTAINTY INFO:")
    print(uncertainty_info)

    R = (RE+refh)*1000          # Ref height in meters
    nAngle = len(lts)
    phis = lts*lt2rad           # Convert local time to radians

    abobj_t0  = abobjs[0]
    tauf_t0   = abobj_t0.tauf
    dtdpf     = abobj_t0.dtaudphif

    taus      = tauf_t0(phis)
    dtaudphis = dtdpf(phis)

    # Use all coeffs
    dtaudt = sum([abobj.tauf(phis)*coeff for abobj,coeff in zip(abobjs,coeffs)])/dt
    # dtaudt = (tauf_tp1(phis)-tauf_tm1(phis))/dt

    sintaus = np.sin(taus)

    # Make sure nothing blows up
    dtaudphis[np.isclose(dtaudphis,0)] = SMALL
    dtaudt[np.isclose(dtaudt,0)] = SMALL
    sintaus[np.isclose(sintaus,0)] = SMALL

    # azimuthal velocity
    dphidts = -dtaudt * dtaudphis / (sintaus**2+dtaudphis**2)
    vphi    = R*sintaus*dphidts

    # colatitudinal velocity
    dthetadt = dtaudt * sintaus**2/(sintaus**2+dtaudphis**2)
    vtheta = R * dthetadt

    # rcn,tcn,pcn = abobj_t0.normvecf(phis,tauf_t0,dtdpf)
    # vtheta = -sintaus/dtaudphis * vphi

    if np.any(~np.isfinite(vtheta) | ~np.isfinite(vphi)):
        breakpoint()
    # if DEBUG:
    #     vtheta2 = tcn/pcn * vphi
    #     print("Making sure all theta velocity components are close")
    #     assert np.all(np.isclose(vtheta,vtheta2))

    # Uncertainty calculation?
    if (len(uncertainty_info) > 0) and (any([np.abs(info) > 0. for info in uncertainty_info])):
            
        Dtheta_deg = uncertainty_info[0]
        DR         = uncertainty_info[1]*1000 if (len(uncertainty_info) >= 2) else 0.
        rho_t      = uncertainty_info[2] if (len(uncertainty_info) >= 3) else 0.
        rho_lon    = uncertainty_info[3] if (len(uncertainty_info) == 4) else 0.

        if DEBUG:
            print("DEBUG INFO")
            print(Dtheta_deg,DR,rho_t,rho_lon)
            print("")

        # Get dphi (space between lts), which must be the same everywhere for this routine to work
        dphis      = np.diff(phis)
        # dphi      = np.median(dphis)
        # assert np.isclose(dphis,dphi).sum() >= int(len(dphis)*0.9)
        assert (dphis > 0).all(),"dphis is not monotonically increasing! What to do?"
        dphis = np.insert(dphis,0, dphis[0])
        for i in range(1,len(dphis)-1):  # probably not necessary, but here we average dphi between two adjacent cells for all cells that are not the first or last
            dphis[i] = (dphis[i]+dphis[i+1])/2
    
        Dtheta    = np.deg2rad(Dtheta_deg)
        Dtaudt    = np.sqrt(2*(1-rho_t  )) * Dtheta / dt     # Uncertainty in time derivative of tau
        Dtaudphi  = np.sqrt(2*(1-rho_lon)) * Dtheta / dphis # Uncertainty in derivative of tau wrt phi
    
        cottaus = 1/np.tan(taus)

        # partial derivatives of vphi
        dvphidR = vphi/R
        dvphidtau = -vphi * cottaus * (1 + 2 * dtaudphis/R/dtaudt/sintaus * vphi)
        dvphiddtaudt = vphi/dtaudt
        dvphiddtaudphi = vphi/dtaudphis + 2/R/dtaudt/sintaus * vphi**2

        # partial derivatives of vtheta
        dvthetadR = vtheta/R
        dvthetadtau = 2*vtheta*cottaus*(1-vtheta/R/dtaudt)
        dvthetaddtaudt = vtheta/dtaudt
        dvthetaddtaudphi = -2*vtheta*dtaudphis/(sintaus**2+dtaudphis**2)

        Dvphi = np.sqrt( (dvphidR*DR)**2 + (dvphidtau*Dtheta)**2 + (dvphiddtaudt*Dtaudt)**2 + (dvphiddtaudphi*Dtaudphi)**2)
        Dvtheta = np.sqrt( (dvthetadR*DR)**2 + (dvthetadtau*Dtheta)**2 + (dvthetaddtaudt*Dtaudt)**2 + (dvthetaddtaudphi*Dtaudphi)**2)

        if return_ENU:
            return np.vstack((vphi,-vtheta,0*vphi)).T, np.vstack((Dvphi,Dvtheta,0*Dvphi)).T
        else:
            return np.vstack((0*vphi,vtheta,vphi)).T, np.vstack((0*Dvphi,Dvtheta,Dvphi)).T

    else:

        if return_ENU:
            return np.vstack((vphi,-vtheta,0*vphi)).T
        else:
            return np.vstack((0*vphi,vtheta,vphi)).T
    

def get_velocities_from_abobj_list__everyscheme(mlts,abobjs,
                                                return_ENU=True,
                                                coordsys='geodetic',
                                                uncertainty_theta_deg=None,
                                                uncertainty_R_km=None,
                                                unc_corr_t=None,
                                                unc_corr_phi=None,
                                                DEBUG=False):
    
    nabobj = len(abobjs)
    canschemes = []             # Keep track of which schemes are possible
    schemedict = dict()

    canschemeslist,schemedictlist = get_schemenames_from_abobj_list(mlts,abobjs,
                                                                    DEBUG=DEBUG)

    outs = []

    get_vel_kws = dict(return_ENU=return_ENU,
                       coordsys=coordsys,
                       DEBUG=DEBUG,
                       uncertainty_theta_deg=uncertainty_theta_deg,
                       uncertainty_R_km=uncertainty_R_km,
                       unc_corr_t=unc_corr_t,
                       unc_corr_phi=unc_corr_phi)

    for refidx in range(nabobj):
    
        #print(f"refidx = {refidx:02d}")
        out = dict()
        ab_t0 = abobjs[refidx]
        
        canschemes = canschemeslist[refidx]
        schemedict = schemedictlist[refidx]

        for scheme in canschemes:
            out[scheme] = ab_t0.get_velocity(mlts,*schemedict[scheme],
                                          **get_vel_kws)

        outs.append(out)
        
    return outs


def get_velocities_from_abobj_list(mlts,abobjs,
                                   coordsys='geodetic',
                                   uncertainty_theta_deg=None,
                                   uncertainty_R_km=None,
                                   unc_corr_t=None,
                                   unc_corr_phi=None,
                                   preferred_scheme1='ctr',
                                   preferred_scheme2='fwd',
                                   preferred_scheme3='bkwd',
                                   max_order_fwd=1,
                                   max_order_bkwd=1,
                                   max_order_ctr=2,
                                   return_ENU=True,
                                   return_usescheme=False,
                                   DEBUG=False):
    
    ALLOWED_GEN_SCHEMES = ['fwd','bkwd','ctr']
    ALLOWED_SCHEMES = ['fwd1','bkwd1','ctr2','fwd2','bkwd2','ctr4']
    prefscheme = preferred_scheme1 
    assert preferred_scheme1 in ALLOWED_GEN_SCHEMES,"'preferred_scheme1' must be one of ['fwd','bkwd','ctr']!"

    def get_maxorder(prefscheme,mof,mob,moc):
        if prefscheme == 'ctr':
            return moc
        elif prefscheme == 'fwd':
            return mof
        elif prefscheme == 'bkwd':
            return mob
    def get_prefscheme_str(prefscheme,mof,mob,moc):
        return prefscheme+str(get_maxorder(prefscheme,mof,mob,moc))


    pref1 = get_prefscheme_str(preferred_scheme1,
                               max_order_fwd,
                               max_order_bkwd,
                               max_order_ctr)
    pref2 = get_prefscheme_str(preferred_scheme2,
                               max_order_fwd,
                               max_order_bkwd,
                               max_order_ctr)
    pref3 = get_prefscheme_str(preferred_scheme3,
                               max_order_fwd,
                               max_order_bkwd,
                               max_order_ctr)

    canschemeslist,schemedictlist = get_schemenames_from_abobj_list(mlts,abobjs,
                                                                    DEBUG=DEBUG)

    nabobj = len(abobjs)
    canschemes = []             # Keep track of which schemes are possible
    schemedict = dict()

    outs = []
    useschemes = []

    get_vel_kws = dict(return_ENU=return_ENU,
                       coordsys=coordsys,
                       DEBUG=DEBUG,
                       uncertainty_theta_deg=uncertainty_theta_deg,
                       uncertainty_R_km=uncertainty_R_km,
                       unc_corr_t=unc_corr_t,
                       unc_corr_phi=unc_corr_phi)

    for refidx in range(nabobj):
    
        print(f"refidx = {refidx:02d}")
        out = dict()
        ab_t0 = abobjs[refidx]
        
        canschemes = canschemeslist[refidx]
        schemedict = schemedictlist[refidx]

        if pref1 in canschemes:
            usepref = pref1
            prefnum = 1
        elif pref2 in canschemes:
            usepref = pref2
            prefnum = 2
        elif pref3 in canschemes:
            usepref = pref3
            prefnum = 3
        else:
            prefnum = 4
            # continue
            
        if DEBUG:
            print(f"refidx={refidx}, using prefnum{prefnum}='{usepref}'")

        if prefnum <= 3:
            out = ab_t0.get_velocity(mlts,*schemedict[usepref],**get_vel_kws)
            useschemes.append(usepref)
        else:
            out = np.zeros((len(mlts),3))*np.nan

        outs.append(out)
        
    if return_usescheme:
        return outs,useschemes
    else:
        return outs


def get_schemenames_from_abobj_list(mlts,abobjs,
                                    DEBUG=False):
    
    nabobj = len(abobjs)
    canschemeslist = []             # Keep track of which schemes are possible
    schemedictlist = []             # List of abobjs to feed to abobj.get_velocity

    for refidx in range(nabobj):
    
        canschemes = []             # Keep track of which schemes are possible
        schemedict = dict()
        print(f"refidx = {refidx:02d}")
        ab_t0 = abobjs[refidx]
        
        # aur boundary at t = t_0-dt ("t minus one")
        if refidx > 0:
            ab_tm1 = abobjs[refidx-1]
            canschemes.append('bkwd1')
            schemedict['bkwd1'] = (ab_tm1,)
        else:
            ab_tm1 = None

        # aur boundary at t = t_0-2*dt ("t minus two")
        if refidx > 1:
            ab_tm2 = abobjs[refidx-2]
            doback2 = True
            canschemes.append('bkwd2')
            schemedict['bkwd2'] = (ab_tm2,ab_tm1)
        else:
            ab_tm2 = None
    
        # aur boundary at t = t_0+dt ("t plus one")
        if refidx < (len(abobjs)-1):
            ab_tp1 = abobjs[refidx+1] 
            canschemes.append('fwd1')
            schemedict['fwd1'] = (ab_tp1,)
        else:
            ab_tp1 = None
        
        # aur boundary at t = t_0+2*dt ("t plus two")
        if refidx < (len(abobjs)-2):
            ab_tp2 = abobjs[refidx+2] 
            canschemes.append('fwd2')
            schemedict['fwd2'] = (ab_tp1,ab_tp2)
        else:
            ab_tp2 = None
    
        doctr2 = ('bkwd1' in canschemes) and ('fwd1' in canschemes)
        doctr4 = ('bkwd2' in canschemes) and ('fwd2' in canschemes)
        
        if doctr2:
            canschemes.append('ctr2')
            schemedict['ctr2'] = (ab_tm1,ab_tp1)
        if doctr4:
            canschemes.append('ctr4')
            schemedict['ctr4'] = (ab_tm2,ab_tm1,ab_tp1,ab_tp2)

        canschemeslist.append(canschemes)
        schemedictlist.append(schemedict)
        
    return canschemeslist,schemedictlist
