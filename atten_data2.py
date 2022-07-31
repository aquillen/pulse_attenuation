
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

nchannels=7 # number of accelerometers

# import max's calibration values for 7 accelerometers
from CalibrationValues import *
#print(zCal)  # only 7 channels
#print(xCal)  # V/m s^-2

# we make a data structure to store data!
class data_struct():
    def __init__(self,tarr,asetx,asety,vsetx,vsety):
        self.vset_R =vsetx
        self.vset_z =vsety
        self.aset_R =asetx
        self.aset_z =asety
        self.tarr_d =tarr  #shifted by tshift also
        self.tshift=0.0
        self.tpeak=0.0 # peak of first channel
        self.amean_x = np.zeros(nchannels)
        self.amean_y = np.zeros(nchannels)
        self.tarr_d0 = tarr  # shifted so peak is at zero in first channel
        self.Rcoords = np.zeros(nchannels)
        self.zcoords = np.zeros(nchannels)
        self.rcoords = np.zeros(nchannels)
        self.aset_r  = self.aset_R*0.0
        self.vset_r  = self.vset_R*0.0
        
    def set_coords(self,Rcoords,zcoords):
        self.Rcoords = Rcoords
        self.zcoords = zcoords
        self.rcoords = np.sqrt(self.Rcoords**2 + self.zcoords**2)
        self.mk_rcomps()
     
    # compute r components of acceleration and velocity
    def mk_rcomps(self):
        for k in range(nchannels):
            self.aset_r[:,k] = (self.aset_R[:,k]*self.Rcoords[k] + self.aset_z[:,k]*self.zcoords[k])/self.rcoords[k]
            self.vset_r[:,k] = (self.vset_R[:,k]*self.Rcoords[k] + self.vset_z[:,k]*self.zcoords[k])/self.rcoords[k]
            
 
# think about correcting for orientation problems using DC voltages
def cor_acc(ds):
    for k in range(nchannels):
        diff =ds.amean_y[k] - y_rest_Cal[k]
        diffmks = diff/yCal[k]  # now in m/s^2
        diffg = diffmks/9.8  # diff now in g
        angle = np.arcsin(diffg)  # in radians
        #print('y mean {:.3f} rest val {:.3f} (V) diff = {:.3f} V diff_g = {:.3f} g'.format(\
        #    ds.amean_y[k],y_rest_Cal[k],diff,diffg))
        print('y angle {:.1f} deg'.format(angle*180/np.pi))
        
    for k in range(nchannels):
        diff =ds.amean_x[k] - x_rest_Cal[k]
        diffmks = diff/xCal[k]  # now in m/s^2
        diffg = diffmks/9.8  # diff now in g
        angle = np.arcsin(diffg)  # in radians
        #print('x mean {:.3f} rest val {:.3f} (V) diff = {:.3f} V diff_g = {:.3f} g'.format(\
        #   ds.amean_y[k],y_rest_Cal[k],diff,diffg))
        print('x angle {:.1f} deg'.format(angle*180/np.pi))

        
# structure for storing measurements
class measured_struct():
    def set_vpeaks(self,dt_arr,top_arr,t_top_arr,theta_arr):
        self.v_fwhm = dt_arr
        self.v_pk = top_arr
        self.v_tprop = t_top_arr
        self.v_theta = theta_arr
        
    def set_apeaks(self,dt_arr,top_arr,t_top_arr,theta_arr):
        self.a_fwhm = dt_arr
        self.a_pk = top_arr
        self.a_tprop = t_top_arr
        self.a_theta = theta_arr
        
    def __init__(self):
        self.junk=0
        self.v_fwhm =np.zeros(nchannels)
        self.a_fwhm =np.zeros(nchannels)
        self.a_pk =np.zeros(nchannels)
        self.v_pk =np.zeros(nchannels)
        self.a_tprop =np.zeros(nchannels)
        self.v_tprop =np.zeros(nchannels)
        self.v_theta =np.zeros(nchannels)
        self.a_theta =np.zeros(nchannels)
        self.EJ_arr = np.zeros(nchannels)
        
    def set_EJ(self,EJ_arr):
        self.EJ_arr = EJ_arr
        

# read in 2 csv files and apply accelerometer calibrations which are Globals xCal,yCal
# returns:
#    aset_z_ret, aset_y_ret:  2x2 arrays that contains z,y accelerations in each channel
#    vset_z,vset-y:        2x2 arrays that contains integrated velocities in z,y in each channel
#    tarr:  a time array shifted so zero is at a peak
# time in csv file now seems to be in s
# arguments:
#    csvfile_root : string giving name of csv file including directory info
#      we assume that we add + 'z.csv' to get the z channels file
#       and we add 'y.csv' to get the y channels file
#    tmin,tmax:   look for peak within [tmin,tmax] in milliseconds in first z channel
#    tleft,tright:
#       truncate to tleft,tright times where time has been shifted to be zero in peak
#    swin is width of a savgol filter applied to data in all channels
def read_data_xy_struct(csvfile_root,tmin,tmax,tleft,tright,swin):
    csvfile_x = csvfile_root + 'x.csv'
    csvfile_y = csvfile_root + 'y.csv'
    myfile = open(csvfile_x, "r")
    myline = myfile.readline()
    myline = myfile.readline()
    if (myline[1:3] == 'ms'):
        print('ms')
        tfac = 1e3  # data was in ms
    else:
        tfac=1  # data in s
    aset_x = np.loadtxt(csvfile_x,skiprows=3,delimiter=',')
    aset_y = np.loadtxt(csvfile_y,skiprows=3,delimiter=',')
    tarr_x = np.squeeze(aset_x[:,0])  #time array from trigger
    tarr_y = np.squeeze(aset_y[:,0])  #time array from trigger
    nl = min(len(tarr_x),len(tarr_y))  #make sure lengths are the same!
    aset_x = aset_x[0:nl,1:nchannels+1]  # and get rid of time array
    aset_y = aset_y[0:nl,1:nchannels+1]

    tarr = tarr_x[0:nl]/tfac  # put in seconds
    
    # clean the data if data is cut off!
    for j in range(nchannels):
        ii_x = np.isinf(aset_x[:,j]) #
        ii_y = np.isinf(aset_y[:,j])
        aset_x[ii_x,j] = 5.0 # +5 V
        aset_y[ii_y,j] = 5.0
        
    # store medians at beginning of dataset in Volts
    amean_x = np.zeros(nchannels)
    amean_y = np.zeros(nchannels)
    for j in range(nchannels):
        amean_x[j]=np.median(aset_x[:200,j])
        amean_y[j]=np.median(aset_y[:200,j])
    
    # calibrate with xCal and yCal values
    for j in range(nchannels):  # nothing in last channel so only to 7
        aset_x[:,j] /= xCal[j] # Volts to m s^-2 z-accels
        aset_y[:,j] /= yCal[j] # Volts to m s^-2 y-accels
            
    # do some filtering of data
    aset_x_filt = np.copy(aset_x)
    aset_y_filt = np.copy(aset_y)
    
    for j in range(nchannels):
        if (swin > 2):
            aset_x_filt[:,j] = savgol_filter(aset_x[:,j],swin,2)  # poly order is 2
            aset_y_filt[:,j] = savgol_filter(aset_y[:,j],swin,2)
        else:
            aset_x_filt[:,j] =aset_x[:,j]
            aset_y_filt[:,j] =aset_y[:,j]
     
    
    # subtract a median value in the range [tleft,0] in all channels
    ii = (tarr > tleft) & (tarr< 0)
    for j in range(nchannels):
        aset_x_filt[:,j] -=np.median(aset_x_filt[ii,j])
        aset_y_filt[:,j] -=np.median(aset_y_filt[ii,j])
    
    # find peak in abs of first channel
    a_chan = aset_x_filt[:,0]
    ipeak = findpeak_i(tarr,tmin,tmax,a_chan) #find peak in first channel
    tpeak = tarr[ipeak]
    tarr -= tpeak
    # shifting time via peak in the first channel of z accels, t=0 is now where peak is
    
    print('tpeak {:.4f}'.format(tpeak))
    
    # also chop data so that is between times tleft and tright
    ii = (tarr > tleft) & (tarr < tright)
    tarr = tarr[ii]
    aset_x_ret = aset_x_filt[ii,:]
    aset_y_ret = aset_y_filt[ii,:]
    
    # make velocity vectors
    vset_x = np.copy(aset_x_ret)*0
    vset_y = np.copy(aset_x_ret)*0
    for j in range(nchannels):
        apulse = np.squeeze(aset_x_ret[:,j])
        vset_x[:,j] = integrate_to_vel(tarr,apulse,tleft)  # integrate velocity
        apulse = np.squeeze(aset_y_ret[:,j])
        vset_y[:,j] = integrate_to_vel(tarr,apulse,tleft)  # integrate velocity
    
    ds =data_struct(tarr,aset_x_ret,aset_y_ret,vset_x,vset_y)
    ds.tpeak = tpeak
    ds.amean_x = amean_x
    ds.amean_y = amean_y
    return ds
    
# fix the peak of a chopped signal with a quadratic function
# here ichan is which channel, rz=1 if in r otherwise in z component
# ddi lets you set range of poly fit for replaceing peak
def fix_chop(ds,tleft,tright,ichan,rz,ddi,frac):
    plt.figure()
    jj = (ds.tarr_d > tleft) & (ds.tarr_d < tright)
    if (rz == 1):
        apulse = np.squeeze(ds.aset_R[jj,ichan])
    else:
        apulse = np.squeeze(ds.aset_z[jj,ichan])
    amax = np.max(apulse)
    tshort = ds.tarr_d[jj]
    ii = (apulse>frac*amax)  # where to replace data
    #print(amax)
    tl = np.min(tshort[ii])  # chopped region in time
    tr = np.max(tshort[ii])
    ileft = np.argmin(np.abs(tshort - tl))  # chop region
    iright = np.argmin(np.abs(tshort - tr))
    #print(amax,ileft,iright)
    
    weights =tshort*0 + 1.0
    weights[ileft-1:iright+1] = 0  #chop with weights
    plt.plot(tshort,apulse,':',lw=1,color='black',label='orig')
    #plt.plot(tshort,weights)
    porder = 4
    pp = np.polyfit(tshort[ileft-ddi:iright+ddi],apulse[ileft-ddi:iright+ddi],porder,w=weights[ileft-ddi:iright+ddi])
    pfun = np.poly1d(pp)
    plt.plot(tshort[ileft-ddi:iright+ddi],pfun(tshort[ileft-ddi:iright+ddi]),lw=3,color='orange',label='pfun',alpha=0.5)
    ileft_replace = np.argmin(np.abs(ds.tarr_d - tl))
    iright_replace = np.argmin(np.abs(ds.tarr_d - tr))+1
    jj2 = (ds.tarr_d >= tl) & (ds.tarr_d <= tr)
    pvals = pfun(ds.tarr_d[ileft_replace:iright_replace])
    plt.plot(ds.tarr_d[ileft_replace:iright_replace],pvals,color='red',\
        label='replace',lw=5,alpha=0.5)
    if (rz == 1):
        #apulse_new =np.squeeze(ds.aset_r[:,ichan])
        #apulse_new[jj2] = pvals
        #plt.plot(ds.tarr_d[jj],apulse_new[jj])
        ds.aset_R[ileft_replace:iright_replace,ichan] = pvals
        plt.plot(ds.tarr_d[jj],ds.aset_R[jj,ichan],label='full')
        vpulse = integrate_to_vel(ds.tarr_d,np.squeeze(ds.aset_R[:,ichan]),tleft)
        ds.vset_R[:,ichan] = vpulse
    else:
        ds.aset_z[ileft_replace:iright_replace,ichan] = pvals
        plt.plot(ds.tarr_d[jj],ds.aset_z[jj,ichan],label='full')
        vpulse = integrate_to_vel(ds.tarr_d,np.squeeze(ds.aset_z[:,ichan]),tleft)
        ds.vset_z[:,ichan] = vpulse
        
    ipeak = findpeak_i(ds.tarr_d,tleft,tright,np.squeeze(ds.aset_R[:,0]))
    #find peak in first channel
    tpeak = ds.tarr_d[ipeak]
    ds.tarr_d -= tpeak
    #print(tpeak)
    ds.tpeak += tpeak
    plt.legend()
    
    
# shift the time in a data structure, being consistent about tpeak storing the shift
def shift_time(ds,tshift):
    ds.tarr_d += tshift
    ds.tpeak  -= tshift
    print('{:.2f} ms'.format(ds.tpeak*1e3))

# replace tpeak with a new one consistently shifting time array in a data structure
def shift_time_tpeak(ds, new_tpeak):
    ds.tarr_d += ds.tpeak  # restore time array of ds to original values
    ds.tpeak  = new_tpeak  # a new tpeak
    ds.tarr_d -= ds.tpeak  # shift time array to be consistent with it
    print('{:.2f} ms'.format(ds.tpeak*1e3))

# find a peak in abs(zarr) that is between tmin and tmax
# the time of the array is tarr
# return the index of the peak
def findpeak_i(tarr,tmin,tmax,zarr):
    itmin = np.argmin(np.fabs(tarr-tmin))
    itmax = np.argmin(np.fabs(tarr-tmax))
    ipeak = np.argmax(np.fabs(zarr[itmin:itmax]))  # we may not want abs here
    return ipeak+itmin

# integrate acceleration, return velocity
def integrate_to_vel(tarr_d,apulse,tleft):
    vpulse = np.cumsum(apulse)*(tarr_d[1] - tarr_d[0])  # integrate
    ii = (tarr_d > tleft) & (tarr_d < 0)  # find a median in [tleft:0]
    vpulse -= np.median(vpulse[ii]) # subtract left side
    return vpulse

# tpts is an array of points which should be zero
def fix_vel(tarr_d,vpulse,tpts,swin):
    plt.figure()
    jj = (tarr_d >= min(tpts)) & (tarr_d  <= max(tpts))
    plt.plot(tarr_d[jj],vpulse[jj],label='orig')
    mzeros = tpts*0.0
    vpulse_out = np.copy(vpulse)
    iarr = np.arange(len(tpts))
    n_tpts =len(tpts)
    for k in range(n_tpts):
        iarr[k] = np.argmin(abs(tarr_d-tpts[k]))
    iarr[-1] +=1
        
    for k in range(n_tpts-1):
        fix_slope(vpulse_out[iarr[k]:iarr[k+1]],10)
    #fixpulse = savgol_filter(vpulse,swin,2)
    plt.plot(tarr_d[jj],vpulse_out[jj],label='fixed')
    #plt.plot(tarr_d[jj],vpulse[jj]-fixpulse[jj],label='try')
    
    plt.legend()
    return vpulse_out
    
        
# plt R z accelerations and velocities
# within [tleft,tright] with vertical offsets dy_az, dy_ay
def plt_av_fig_ds(ds,dy_vec,\
            tleft,tright,ofile,label_x,label_y,label_s,\
                        label_x2,label_y2,label_s2):

    do_fix_slope = 1
    tshift=0.0
    colorlist = ['red','orange','gold','green','cyan','blue','purple']
    fig,axarr = plt.subplots(2,2,figsize=(6.0,3.0),\
            dpi=200,sharex=True,sharey=False)
    plt.subplots_adjust(hspace=0,wspace=0,top=0.96, bottom=0.20, left=0.15,right=0.82)
    
    tarr = ds.tarr_d
    tfac = 1e3;  # to ms
    tt = (np.copy(tarr) + tshift)*tfac  # shift in time, in ms
    #rlabel = r'$\hat R$'
    #zlabel = r'$\hat z$'
    
    ii = (tt>tleft*tfac) & (tt < tright*tfac)
    asetz = np.copy(ds.aset_z[ii,:])
    asetr = np.copy(ds.aset_R[ii,:])
    vsetz = np.copy(ds.vset_z[ii,:])
    vsetr = np.copy(ds.vset_R[ii,:])
    tt = tt[ii]
    ddi=20  # for slope fixing
    
    #axarr[1][1].set_yticklabels([])
    #axarr[0][1].set_yticklabels([])
    axarr[1][1].yaxis.set_label_position("right")
    axarr[1][1].yaxis.tick_right()
    axarr[0][1].yaxis.set_label_position("right")
    axarr[0][1].yaxis.tick_right()
    
    min1=0.; min2=0.; min3=0.;min4=0.;
    max1=0.; max2=0.; max3=0.;max4=0.;

    for k in range(nchannels):
        #clabel='{:d}'.format(k)
        apulsez = np.copy(np.squeeze(asetz[:,k]))
        apulser = np.copy(np.squeeze(asetr[:,k]))
        vpulsez = np.copy(np.squeeze(vsetz[:,k]))
        vpulser = np.copy(np.squeeze(vsetr[:,k]))
        if (do_fix_slope==1):
            fix_slope(apulser,ddi)
            fix_slope(apulsez,ddi)
            fix_slope(vpulser,ddi)
            fix_slope(vpulsez,ddi)
        toplot1 = apulser - k*dy_vec[0]
        toplot2 = apulsez - k*dy_vec[1]
        toplot3 = vpulser - k*dy_vec[2]
        toplot4 = vpulsez - k*dy_vec[3]
        min1= min(np.min(toplot1),min1)
        min2= min(np.min(toplot2),min2)
        min3= min(np.min(toplot3),min3)
        min4= min(np.min(toplot4),min4)
        max1= max(np.max(toplot1),max1)
        max2= max(np.max(toplot2),max2)
        max3= max(np.max(toplot3),max3)
        max4= max(np.max(toplot4),max4)
        
        axarr[0][0].plot(tt,toplot1,lw=1,color=colorlist[k])
        axarr[1][0].plot(tt,toplot2,lw=1,color=colorlist[k])
        axarr[0][1].plot(tt,toplot3,lw=1,color=colorlist[k])
        axarr[1][1].plot(tt,toplot4,lw=1,color=colorlist[k])
     
     
    #axarr[0][0].set_ylim([min1*1.05,max1*1.1])  # top left
    #axarr[1][0].set_ylim([min2*1.05,max2*1.1])  # bottom left
    #axarr[0][1].set_ylim([min3*1.05,max3*1.1])  # top right
    #axarr[1][1].set_ylim([min4*1.05,max4*1.1])  # bottom right
    
    axarr[0][0].set_ylabel(r'$\hat \mathbf{R}$' + r' accel (m/s$^2$)' + '\n + offset')  #top left
    axarr[1][0].set_ylabel(r'$\hat \mathbf{z}$' + r' accel (m/s$^2$)' + '\n + offset')  #bottom left
    
    axarr[0][1].set_ylabel(r'$\hat \mathbf{R}$' + r' velocity (m/s)' + '\n + offset')  #top left
    axarr[1][1].set_ylabel(r'$\hat \mathbf{z}$' + r' velocity (m/s)' + '\n + offset')  #bottom left
    
    if (tfac >1):
        axarr[1][1].set_xlabel(r'time (ms)')  # bottom right
        axarr[1][0].set_xlabel(r'time (ms)')  # bottom left
    else:
        axarr[1][1].set_xlabel(r'time (s)')
        axarr[1][0].set_xlabel(r'time (s)')
    axarr[0][0].text(label_x,label_y,label_s)
    axarr[0][0].text(label_x2,label_y2,label_s2)
 
    if (len(ofile)>3):
        plt.savefig(ofile)
    return fig,axarr


# subtract off a linear function that has slope set by values at the
# ends of the array
def fix_slope(pulse,ddi):
    nl = len(pulse)
    vl = np.median(pulse[0:ddi])
    pulse -= vl
    vr = np.median(pulse[-ddi:])
    slope = vr/nl
    ilist = np.arange(nl,dtype=float)
    pulse -= slope*ilist

# find FWHM of the peak within tleft and tright and plot it
# arguments:
#  tarr is time array
#  apulse is either acceleration or velocity from an accelerometer
#  tlef, tright, find peak within this time range
# returns:
#   dt: fwhm
#   top: peak value
#   t_top: peak time
def find_fwhm(tarr,apulse,tleft,tright,ax):
    #plt.figure()
    ileft = np.argmin(np.abs(tarr-tleft))
    iright = np.argmin(np.abs(tarr-tright))
    #print(ileft,iright)
    tarr_short = np.copy(tarr[ileft:iright])
    apulse_short = np.copy(apulse[ileft:iright])
    top = np.max(apulse_short)  #top value
    itop = np.argmax(apulse_short)  # index of peak
    if (itop==0):
        return 0,0,0
    t_top = tarr_short[itop]  # time of peak
    toph = top/2  #half height
    i2=itop
    while ((apulse_short[i2] > toph) and (i2 < len(apulse_short)-1)) :
        i2+=1
    i1=itop
    while ((apulse_short[i1] > toph) and (i1 >0)):
        i1-=1

    ax.plot(tarr_short,apulse_short)
    ax.plot([tleft,tright],[top,top])
    ax.plot([tleft,tright],[toph,toph])
    ax.plot([t_top],[top],'ko')
    
    #i1 = np.argmin(np.fabs(apulse_short[:itop] - toph))
    #i2 = j
    #print(tarr_short[i1],tarr_short[i2])
    
    ax.plot(tarr_short[i1:i2],apulse_short[i1:i2])
    dt = tarr_short[i2] - tarr_short[i1]
    #print('dt = {:.2f} ms'.format(dt*1e3))
    #print('top val = {:.3f}'.format(top))
    #print('top time = {:.4f}'.format(t_top))
    #plt.close()
    return dt,top,t_top
    # return pulse width, value of peak, and time of peak
    
    
    
# compute total seismic energy from accelerometer  in vpulse
# here v_p is plulse velocity
# rho_s is substrate density
# x0 is distance to channel
def find_energy(tarr,vpulse,tleft,tright,v_p,rho_s,x0):
    fac = np.pi*2.0*np.power(x0,2)*rho_s*v_p  #half sphere
    ileft = np.argmin(np.abs(tarr-tleft))
    iright = np.argmin(np.abs(tarr-tright))
    dt = tarr[1] - tarr[0]
    tarr_short = np.copy(tarr[ileft:iright])
    vpulse_short = np.copy(vpulse[ileft:iright])
    vpulse_short -= np.median(vpulse[ileft-10:ileft])
    plt.plot(tarr_short,vpulse_short)
    v2sum =np.sum(vpulse_short**2)*dt
    EJ =v2sum*fac
    print('E={:.6f} (J)'.format(EJ))
    return EJ
    # no factor of 1/2 in energy, assuming equipartition

# return an array of energies estimated from previous routine
# only using radial component?
def find_energy_ds(ds,tleft,tright,v_p,rho_s):
    tarr = ds.tarr_d
    EJ_arr = np.zeros(nchannels)
    for k in range(nchannels):
        vpulseR = np.squeeze(ds.vset_R[:,k])
        vpulsez = np.squeeze(ds.vset_z[:,k])
        vpulse = np.sqrt(vpulseR**2 + vpulsez**2)
        # could change to include vr only components here!
        x0 = ds.rcoords[k]
        EJ_arr[k] =find_energy(tarr,vpulse,tleft,tright,v_p,rho_s,x0)
    #plt.close()
    return EJ_arr


# plot one set of accelerations,
# within [tleft,tright] with vertical offsets dy
# does accelerations if av = 1
# does velocities if av = 0
# rz  does r if rz==1, does z if rz==0
def plt_1_fig_wide_ds(ds,dy,\
            tleft,tright,tshift,rz,av,ofile):
      
    fig,ax = plt.subplots(1,1,figsize=(3,1.5),\
            dpi=200,sharex=True,sharey=False)
    plt.subplots_adjust(hspace=0,wspace=0,top=0.97, bottom=0.20, left=0.19,right=0.99)
    tarr = ds.tarr_d
    tfac = 1e3;  # to ms
    if (av==1):
        setz = ds.aset_z
        setr = ds.aset_r
        avlabel = r'accel (m/s$^2$)'
    else:
        setz = ds.vset_z
        setr = ds.vset_r
        avlabel = r'velocity (m/s)'
        
    if (rz==1):
        mset = setr
        rzlabel =r'$\hat\mathbf{r}$ '
    else:
        mset = setz
        rzlabel =r'$\hat\mathbf{z}$ '
        
    ax.set_ylabel(rzlabel + avlabel +  '\n + offset')
    if (tfac >1):
        ax.set_xlabel(r'time (ms)')
    else:
        ax.set_xlabel(r'time (s)')
    colorlist = ['red','orange','gold','green','cyan','blue','purple']
    
    tt = (np.copy(tarr) + tshift)*tfac  # shift in time
    ii = (tt>tleft*tfac) & (tt < tright*tfac)
    mset = mset[ii,:]
    tt = tt[ii]
    pulse_last = np.copy(np.squeeze(mset[:,nchannels-1]) )
    
    max1 = np.max(mset[:,0])
    min1 = np.min(mset[:,nchannels-1] - (nchannels-1)*dy)
    ax.set_ylim([min1*1.05,max1*1.1])
   
    for k in range(nchannels):
        pulse = np.copy(np.squeeze(mset[:,k]))
        ax.plot(tt,pulse - k*dy,lw=1,color=colorlist[k])
 
    if (len(ofile)>3):
        plt.savefig(ofile)
    plt.close()


# find peaks in all channels
# arguments:  ds  data structure
# av: if av==1 do accels
#   if av==0 do vels
# rz:  if rz ==1 do r components, else do z components
# tleft, tright: search in this range of time
# swin:  savgol filter before looking for maxima
# returns arrays:
#   dt_arr: width of peak, fwhm
#   top_arr: peak values
#   t_top_arr:  time of  peak values
def find_peaks(ds,tleft,tright,av,rz,swin):
    tarr  = ds.tarr_d
    dt_arr = []
    top_arr = []
    t_top_arr = []
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    for k in range(nchannels):
        if (av ==1):
            if (rz ==1):
                pulse = savgol_filter(ds.aset_r[:,k],swin,2)
            else:
                pulse = savgol_filter(ds.aset_z[:,k],swin,2)
        else:
            if (rz ==1):
                pulse = savgol_filter(ds.vset_r[:,k],swin,2)
            else:
                pulse = savgol_filter(ds.vset_z[:,k],swin,2)
        dt,top_val,t_top =find_fwhm(tarr,pulse,tleft,tright,ax)
        dt_arr = np.append(dt_arr,dt)
        top_arr = np.append(top_arr,top_val)
        t_top_arr = np.append(t_top_arr,t_top)
    return dt_arr,top_arr,t_top_arr
    
    
# find peaks in all channels, using radial components
#   both for vels and accels
# arguments:  ds  data structure
#   tleft, tright_a or tright_v: search in this range of time
#  tright_a, tright_v can be different for velocity and acccels
#   swin:  savgol filter before looking for maxima

# return measurement structure containing arrays
#   dt_arr: width of peak, fwhm
#   top_arr: peak values
#   t_top_arr:  time of  peak values
#   theta_arr: direction based on velocity
def find_peaks_rcombo(ds,tleft,tright_a,tright_v,swin):

    tarr  = ds.tarr_d
    ms =measured_struct()
    
    dt_arr = []
    top_arr = []
    t_top_arr = []
    theta_arr = []
    ii = (tarr >  tleft-0.001) & (tarr < tleft)
    fig,axarr = plt.subplots(2,nchannels,figsize=(10,3))
    plt.subplots_adjust(wspace=0)
    for k in range(nchannels):
        aR = np.squeeze(ds.aset_R[:,k])
        az = np.squeeze(ds.aset_z[:,k])
        #apulse = np.sqrt(aR**2 + az**2)
        apulse = (aR*ds.Rcoords[k] + az*ds.zcoords[k])/ds.rcoords[k]
        atheta = np.arctan2(az,aR)
        apulse = savgol_filter(apulse,swin,2)
        apulse -= np.median(apulse[ii])
        dt,top_val,t_top =find_fwhm(tarr,apulse,tleft,tright_a,axarr[0][k])
        # get theta at the time of peak
        ith = np.argmin(abs(tarr-t_top))
        theta_top = atheta[ith]
        dt_arr = np.append(dt_arr,dt)
        top_arr = np.append(top_arr,top_val)
        t_top_arr = np.append(t_top_arr,t_top)
        theta_arr = np.append(theta_arr,theta_top)
    
    ms.set_apeaks(dt_arr,top_arr,t_top_arr,theta_arr)
    
    dt_arr = []
    top_arr = []
    t_top_arr = []
    theta_arr = []
    for k in range(nchannels):
        vR = np.squeeze(ds.vset_R[:,k])
        vz = np.squeeze(ds.vset_z[:,k])
        #vpulse = np.sqrt(vR**2 + vz**2)
        vpulse = (vR*ds.Rcoords[k] + vz*ds.zcoords[k])/ds.rcoords[k]
        vtheta = np.arctan2(vz,vR)
        vpulse = savgol_filter(vpulse,swin,2)
        vpulse -= np.median(vpulse[ii])
       
        dt,top_val,t_top =find_fwhm(tarr,vpulse,tleft,tright_v,axarr[1][k])
        # get theta at the time of peak
        ith = np.argmin(abs(tarr-t_top))
        theta_top = vtheta[ith]
        
        dt_arr = np.append(dt_arr,dt)
        top_arr = np.append(top_arr,top_val)
        t_top_arr = np.append(t_top_arr,t_top)
        theta_arr = np.append(theta_arr,theta_top)
        
    ms.set_vpeaks(dt_arr,top_arr,t_top_arr,theta_arr)
    return ms
    
# same as above but fix a single channel, choose velocity or accel
def find_peaks_rsingle(ds,ms,tleft,tright_a,tright_v,swin,ichan,av,rz):
    tarr  = ds.tarr_d
    #ms =measured_struct()
    k = ichan
    ii = (tarr >  tleft-0.001) & (tarr < tleft)

    aR = np.squeeze(ds.aset_R[:,k])
    az = np.squeeze(ds.aset_z[:,k])
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    if (av==1):
        apulse = (aR*ds.Rcoords[k] + az*ds.zcoords[k])/ds.rcoords[k]
        atheta = np.arctan2(az,aR)
        if (rz==1):
            apulse=aR
        if (rz==0):
            apulse=np.abs(az)
        # else use ar
        apulse = savgol_filter(apulse,swin,2)
        apulse -= np.median(apulse[ii])
        dt,top_val,t_top =find_fwhm(tarr,apulse,tleft,tright_a,ax)
        # get theta at the time of peak
        ith = np.argmin(abs(tarr-t_top))
        theta_top = atheta[ith]
        ms.a_fwhm[k] = dt
        ms.a_pk[k] = top_val
        ms.a_tprop[k] = t_top
        ms.a_theta[k] = theta_top
        
    else:
        vR = np.squeeze(ds.vset_R[:,k])
        vz = np.squeeze(ds.vset_z[:,k])
        vpulse = (vR*ds.Rcoords[k] + vz*ds.zcoords[k])/ds.rcoords[k]
        vtheta = np.arctan2(vz,vR)
        vpulse = savgol_filter(vpulse,swin,2)
        vpulse -= np.median(vpulse[ii])
       
        dt,top_val,t_top =find_fwhm(tarr,vpulse,tleft,tright_v,ax)
        # get theta at the time of peak
        ith = np.argmin(abs(tarr-t_top))
        theta_top = vtheta[ith]
        
        ms.v_fwhm[k] = dt
        ms.v_pk[k] = top_val
        ms.v_tprop[k] = t_top
        ms.v_theta[k] = theta_top



def plt_3_peaks(fig,ax,ds,tleft,tright,fac,av,label,color,ofile):
    #fig,ax = plt.subplots(1,1,figsize=(5,3.5),\
    #        dpi=100,sharex=True,sharey=False)
    #plt.subplots_adjust(hspace=0,wspace=0,top=0.97, bottom=0.20, left=0.19,right=0.99)
    tarr = ds.tarr_d
    ii = (tarr > tleft) & (tarr< tright)
    tarr = tarr[ii]
    if (av ==1):
        mset = ds.aset_r[ii,:]
    else:
        mset = ds.vset_r[ii,:]
    for k in range(3):
        ss = label+'{:d}'.format(k)
        t_toplot = tarr*1e3
        ax.plot(t_toplot,mset[:,k]*fac**k,label=ss,lw=k+1)
    plt.legend()
    ax.set_xlabel('t (ms) + offset')
    if (av ==1):
        ax.set_ylabel('a scaled + offset')
    else:
        ax.set_ylabel('v scaled + offset')
    #return fig,ax
    


def plt_peaks(ds,ms,av,rz,tleft,tright,fac):
    colorlist = ['red','orange','gold','green','cyan','blue','purple']
    ii = (ds.tarr_d > tleft) & (ds.tarr_d < tright)
    tt = ds.tarr_d[ii]
    for k in range(nchannels):
        if (av==1) and (rz == 1):
            plt.plot(tt,ds.aset_R[ii,k]*fac**k,color=colorlist[k])
            plt.plot(ms.a_tprop[k],ms.a_pk[k],'o',color=colorlist[k])
        if (av==0) and (rz == 1):
            plt.plot(tt,ds.vset_R[ii,k]*fac**k,color=colorlist[k])
            plt.plot(ms.v_tprop[k],ms.v_pk[k],'o',color=colorlist[k])
        if (av==1) and (rz == 0):
            plt.plot(tt,ds.aset_z[ii,k]*fac**k,color=colorlist[k])
            plt.plot(ms.a_tprop[k],ms.a_pk[k],'o',color=colorlist[k])
        if (av==0) and (rz == 0):
            plt.plot(tt,ds.vset_z[ii,k]*fac**k,color=colorlist[k])
            plt.plot(ms.v_tprop[k],ms.v_pk[k],'o',color=colorlist[k])
        if (av==1) and (rz == 2):
            ar = (ds.aset_R[ii,k]*ds.Rcoords[k] + ds.aset_z[ii,k]*ds.zcoords[k])/ds.rcoords[k]
            plt.plot(tt,ar*fac**k,color=colorlist[k])
            plt.plot(ms.a_tprop[k],ms.a_pk[k],'o',color=colorlist[k])
        if (av==0) and (rz == 2):
            vr = (ds.vset_R[ii,k]*ds.Rcoords[k] + ds.vset_z[ii,k]*ds.zcoords[k])/ds.rcoords[k]
            plt.plot(tt,vr*fac**k,color=colorlist[k])
            plt.plot(ms.v_tprop[k],ms.v_pk[k],'o',color=colorlist[k])

