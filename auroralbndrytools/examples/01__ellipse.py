########################################
import numpy as np
from datetime import datetime,timedelta

from auroralbndrytools import ABoundary,get_dl,get_velocities_from_abobj_list__everyscheme,get_velocities_from_abobj_list

showplots = True
savefig = False

verbose = True

if showplots:
    
    try:
        import matplotlib as mpl
    except:
        raise ValueError("matplotlib must be installed and importable!")

    # MPL opts
    mpl.rcParams.update({'figure.figsize': [10.0, 8.0]})
    plotdir = './'
    mpl.rcParams.update({'savefig.directory': plotdir})
    
    import matplotlib.pyplot as plt
    plt.ion()
    
    try:
        from polplot import Polarplot #https://github.com/klaundal/polplot
    except:
        import warnings
        warnings.warn("polplot must be installed and importable if you wish to see plots!")
    
mlt2rad = 15*np.pi/180

## Get boundary velocities

coordsys = 'geodetic'
# coordsys = 'Apex'

# Select how many times
imindices = [0,1,2,3,4,5]
t0 = datetime(2000,10,28,11,50,0)
ts = [t0 + timedelta(seconds=dt) for dt in [imindex*60 for imindex in imindices]]

saveplotpref = 'bndryvelocity'

bndries = []
t = np.deg2rad(np.linspace(0,359,360))  # NOTE: This is not phi; it is a parameterization parameter for an ellipse!
ra = 20                                 # Radius along noon/midnight meridian
rb = 20                                 # Radius along dawn/dusk meridian
anglespeed = 0.5                          # How much should the boundary expand for each timestep?

if ra == rb:
    saveplotpref += f'_CIRCLE_TEST_r{ra}_dthetadt{anglespeed}'
    shapestr = 'CIRCLE'
    anglestr = f'radius={ra}'
else:
    shapestr = 'ELLIPSE'
    anglestr = f'radius_a={ra},radius_b={rb}'
    saveplotpref += f'_ELLIPSE_TEST_ra{ra}_rb{rb}_dthetadt{anglespeed}'

print(f"GENERATING {shapestr} ({anglestr}) IN MLT/MLAT COORDINATES")

for refidx in imindices:

    print(f"i={refidx}, t = {ts[refidx]} …",end='')
    
    x = (ra+refidx*anglespeed)*np.cos(t)
    y = (rb+refidx*anglespeed)*np.sin(t)
    mlatp = 90-np.sqrt(x**2+y**2)
    mltp = (np.arctan2(y,x)/mlt2rad) % 24
    
    xnew = (90-mlatp)*np.cos(mltp*mlt2rad)
    ynew = (90-mlatp)*np.sin(mltp*mlt2rad)
        
    n_interp_pts = len(mltp)*5+1
    savgol_npts  = int((n_interp_pts*0.04//2)*2+1)  # odd number

    # print(f"Got {len(mltp)} points")

    bndry = ABoundary(mltp,mlatp,t=ts[refidx],
                      n_interp_pts=n_interp_pts,
                      savgol_npts=savgol_npts,
                      coordsys=coordsys,
                      discontinuity_MLT=12,
                      verbose=verbose)

    bndries.append(bndry)

##
#Plot boundaries

if showplots:
    print("EXAMPLE BOUNDARY PLOTS")
    
    mltmin = 2
    mltmax = 22
    
    dmlt = 0.2
    showmlts = np.arange(mltmin,mltmax+dmlt/2,dmlt)
    
    showbndries = [0,1,2,3,4,5]
    
    
    fig = plt.figure(10)
    for refidx in showbndries:
        b = bndries[refidx]
    
        savename = f'{saveplotpref}__refidx{refidx:02d}_boundaryplot.png'
        
        print("")
        print(b.t)
            
    
        figtitle = f'refidx={refidx}\n{b.t}'
        fig.suptitle(figtitle)
    
        ax = plt.subplot(111)
        # ax.set_title("GEO")
        geocolor,magcolor = 'C0','C1'
        pax = Polarplot(ax,minlat=60,sector='all')
        pax.plot(b._gdlatp,b._gltp,marker='^',color=geocolor)
        pax.plot(b._geodict['latsmooth'],b._geodict['ltinterp'],color='black',linestyle='--')
        pax.plot(b._mlatp,b._mltp,marker='o',color=magcolor)
        pax.plot(b._magdict['latsmooth'],b._magdict['ltinterp'],color='black',linestyle=':')
        
        pax.write(61,12,"GEO",color=geocolor)
        pax.write(61,11,"--GEOinterp",color='black')
        pax.write(64,12,"MAG",color=magcolor)
        pax.write(64,11,":MAGinterp",color='black')
            
        if savefig:
            print(f"Saving to {savename}")
            plt.savefig(plotdir+savename,dpi=150)
            plt.clf()
        else:
            break
    
## dl and velocity plot

# select MLTs where we will calculate the differential length vector dl and the boundary normal velocity
calcmlt = np.arange(2,8,0.1)
    
# Select image
refidx = 1
b = bndries[refidx]
    
if showplots:
    print("EXAMPLE DL AND VELOCITY PLOT")
    
    # Get velocities
    uncertainty_theta_deg = 0.5
    uncertainty_R_km = None
    unc_corr_t = None
    unc_corr_phi = 0.5

    get_vel_kws = dict(
        preferred_scheme1='ctr',
        preferred_scheme2='fwd',
        preferred_scheme3='bkwd',
        max_order_fwd=1,
        max_order_bkwd=1,
        max_order_ctr=2,
        return_usescheme=True,
        return_ENU=True,
        uncertainty_theta_deg=uncertainty_theta_deg,
        uncertainty_R_km=uncertainty_R_km,
        unc_corr_t=unc_corr_t,
        unc_corr_phi=unc_corr_phi,
        verbose=verbose,
        DEBUG=False)
    velsg,usescheme = get_velocities_from_abobj_list(calcmlt,bndries,
                                                    coordsys='geodetic',
                                                    **get_vel_kws)
    velsA,usescheme = get_velocities_from_abobj_list(calcmlt,bndries,
                                                     coordsys='Apex',
                                                     **get_vel_kws)
    
    velg = velsg[refidx]
    velA = velsA[refidx]
    
    if (uncertainty_theta_deg is not None):
        velg, dvelg = velg
        velA, dvelA = velA
    
    calcmlat = b.get_boundary_mlat(calcmlt)
    dlvec_geo = b.get_dl_vector_geo(calcmlt)/1000  # in km
    dlvec_apex = b.get_dl_vector_apex(calcmlt)/1000  # in km
    
    gdlats,lts = b.get_boundary_gdlatlt(calcmlt)
    
    figtitle = f'refidx={refidx}\n{bndries[refidx].t}\nu calculated in {coordsys} coords'
    savename = f'{saveplotpref}__refidx{refidx:02d}_ANALYTIC_{coordsys}_dl_and_boundaryvel.png'
    
    plotpinvel_kw = dict(SCALE=500,
                         unit='m/s',marker='o',color='blue')
    plotpindl_kw = dict(SCALE=500,
                        unit='km',marker='^',color='orange')
    
    show_mag_coord_version = False
    showdl = False
    fig,axes = plt.subplots(1,1+show_mag_coord_version,figsize=(17,9),num=21)
    if show_mag_coord_version:
        ax0,ax1 = axes
        pax0 = Polarplot(ax0,minlat=60,sector='dawn')
        pax1 = Polarplot(ax1,minlat=60,sector='dawn')
    else:
        ax0 = axes
        pax0 = Polarplot(ax0,minlat=60,sector='dawn')
        
    
    # Selveste boundaries
    pax0.plot(b._gdlatp,b._gltp,color='C0')
    pax0.plot(b._geodict['latsmooth'],b._geodict['ltinterp'],color='black',linestyle='--')
    if show_mag_coord_version:
        pax1.plot(b._mlatp,b._mltp,color='C0')
        pax1.plot(b._magdict['latsmooth'],b._magdict['ltinterp'],color='black',linestyle=':')
    
    # velocity vectors
    pax0.plotpins(gdlats,lts,velg[:,1],velg[:,0],**plotpinvel_kw)
    if show_mag_coord_version:
        pax1.plotpins(calcmlat,calcmlt,velA[:,1],velA[:,0],**plotpinvel_kw)
    
    # dl vectors
    if showdl:
        avger = lambda x: (x[1:]+x[0:-1])/2
        
        pax0.plotpins(avger(gdlats),avger(lts),dlvec_geo[:,1],dlvec_geo[:,0],**plotpindl_kw)
        if show_mag_coord_version:
            pax1.plotpins(avger(calcmlat),avger(calcmlt),dlvec_apex[:,1],dlvec_apex[:,0],**plotpindl_kw)
    
    if savefig:
        print(f"Saving to {savename}")
        plt.savefig(plotdir+savename,dpi=150)
    

##############################
# Example calculation of boundary normal velocity vectors and their uncertainties

b = bndries[refidx]
calcmlt = np.arange(2,8.1,0.4)
gdlats,lts = b.get_boundary_gdlatlt(calcmlt)

uncertainty_theta_degs  = [1.0, 0.5, 0.2]
unc_corr_ts             = [0.0, 0.0, 0.0]
unc_corr_phis           = [0.0, 0.0, 0.0]

uncertainty_theta_degs  = [1.0, 1.0, 1.0]
# unc_corr_ts             = [0.0, 0.0, 0.0]
unc_corr_ts             = [0.9]*3
unc_corr_phis           = [0.0, 0.3, 0.7]

uncertainty_theta_deg   = 0.5
uncertainty_R_km        = None
unc_corr_t              = None
unc_corr_phi            = 0.5

get_vel_kws = dict(preferred_scheme1='ctr',
                   preferred_scheme2='fwd',
                   preferred_scheme3='bkwd',
                   max_order_fwd=1,
                   max_order_bkwd=1,
                   max_order_ctr=2,
                   return_usescheme=True,
                   return_ENU=True,
                   uncertainty_theta_deg=uncertainty_theta_deg,
                   uncertainty_R_km=uncertainty_R_km,
                   unc_corr_t=unc_corr_t,
                   unc_corr_phi=unc_corr_phi,
                   verbose=verbose,
                   DEBUG=False)

velsgs = []
useschemes = []
if verbose:
    print("uncertainty_theta_deg, unc_corr_ts, unc_corr_phis[i]")
for i,uncertainty_theta_deg in enumerate(uncertainty_theta_degs):
    get_vel_kws['uncertainty_theta_deg'] = uncertainty_theta_deg
    get_vel_kws['unc_corr_t'] = unc_corr_ts[i]
    get_vel_kws['unc_corr_phi'] = unc_corr_phis[i]

    if verbose:
        print(uncertainty_theta_deg,", ",unc_corr_ts[i],", ",unc_corr_phis[i])
    velsg,usescheme = get_velocities_from_abobj_list(calcmlt,bndries,
                                                     coordsys='geodetic',
                                                     **get_vel_kws)

    velsgs.append(velsg)
    useschemes.append(usescheme)


if showplots:
    linewidths = [5,3,1]
    alphas = [0.3,0.3,1.0]
    colors = ['black','black','black']
    
    fig = plt.figure(13)
    plt.clf()
    ax0 = plt.subplot(1,2,1)
    ax1 = plt.subplot(1,2,2)

    legend_unctheta = not all([x == uncertainty_theta_degs[0] for x in uncertainty_theta_degs])
    legend_rho_t = not all([x == unc_corr_ts[0] for x in unc_corr_ts])
    legend_rho_phi = not all([x == unc_corr_phis[0] for x in unc_corr_phis])
    figtitle = ''
    if not legend_unctheta:
        figtitle += f"$\Delta \\theta =${uncertainty_theta_degs[0]}$^\circ$"
    if not legend_rho_t:
        figtitle += (", " if len(figtitle) > 0 else "") + f"$\\rho_t =${unc_corr_ts[0]}"
    if not legend_rho_phi:
        figtitle += (", " if len(figtitle) > 0 else "") + f"$\\rho_\\phi =${unc_corr_phis[0]}"
    
    if len(figtitle) > 0:
        _ = fig.suptitle(figtitle)


    for i,uncertainty_theta_deg in enumerate(uncertainty_theta_degs):
        
        rho_t = unc_corr_ts[i]
        rho_phi = unc_corr_phis[i]

        velsg, usescheme = velsgs[i], useschemes[i]
    
        velg,dvelg = velsg[refidx]
    
        # _ = fig.suptitle(f"$\Delta \\theta =${uncertainty_theta_deg}$^\circ$")
    
        nrows = len(uncertainty_theta_degs)
    
        label = ''
        if legend_unctheta:
            label += f"$\Delta \\theta =${uncertainty_theta_deg}$^\circ$"
        if legend_rho_t:
            label += (", " if len(label) > 0 else "") + f"$\\rho_t =${rho_t}"
        if legend_rho_phi:
            label += (", " if len(label) > 0 else "") + f"$\\rho_\\phi =${rho_phi}"

        # labeler = f"$\Delta \\theta =${uncertainty_theta_deg}$^\circ$"
        # if np.abs(rho_t) > 0.:
        #     labeler += f","

        plt.sca(ax0)
        plt.title("East")
        plt.errorbar(lts,velg[:,0],dvelg[:,0],label=label,
                     linewidth=linewidths[i],
                     alpha=alphas[i],
                     color=colors[i])
        plt.xlabel("MLT")
        plt.ylabel("m/s")
    
        plt.sca(ax1)
        plt.title("North")
        plt.errorbar(lts,velg[:,1],dvelg[:,1],
                     linewidth=linewidths[i],
                     alpha=alphas[i],
                     color=colors[i])
        plt.xlabel("MLT")
        plt.ylabel("m/s")
    
    plt.sca(ax0)
    plt.legend()
    # breakpoint()

