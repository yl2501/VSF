import matplotlib
matplotlib.use('agg')

import yt
from yt.units import *
from yt.utilities.physical_constants import *
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import random

name_pattern = 'DD%04i/stest_%04i'
#all_I_want=np.array([82, 103, 114, 193, 245])

def vsf_all(r_max=150,n_points=10000,dir="/FB_again/",imin=77):
  savedir=dir
  stest = 'stest_%04i' %imin
  ds = yt.load(dir+name_pattern%(imin,imin))
  ds.add_gradient_fields(("gas","temperature"))
  ds.add_gradient_fields(("gas","entropy"))
  ds.add_gradient_fields(("gas","logP"))
  def _ShockMach(field,data):  #value is shock Mach number
     ep=0.002  # Mach of 1.001
#     ep=0.223  # Mach of 1.1
     new_field = np.zeros(data["pressure"].shape, dtype='float64')
     dog=np.maximum(np.array(data["logP_gradient_x"])*np.array(data["index", "dx"].in_cgs()),np.array(data["logP_gradient_y"])*np.array(data["index", "dy"].in_cgs()))
     dlogP=np.maximum(dog,np.array(data["logP_gradient_z"])*np.array(data["index", "dz"].in_cgs()))
     too_big=dlogP>2  # if the shock is too strong (mach > 5ish), mouse will cause overflow problem
     dlogP[too_big]=2
     useSam=True
     if useSam==True:
       TSx=np.array(data['gas', 'temperature_gradient_x'])*np.array(data['gas', 'entropy_gradient_x'])
       TSy=np.array(data['gas', 'temperature_gradient_y'])*np.array(data['gas', 'entropy_gradient_y'])
       TSz=np.array(data['gas', 'temperature_gradient_z'])*np.array(data['gas', 'entropy_gradient_z'])
       TS1=np.logical_or((TSx>0), (TSy>0))
       TS=np.logical_or(TS1,  (TSz>0))
       good=(TS & (np.array(data["gas", "velocity_divergence"])<0)) & (dlogP > ep)
     else:
       cat=np.array(data["gas","thermal_energy"]*data["gas","cell_mass"])/np.array(data["gas","kinetic_energy"]*data["index","cell_volume"].in_cgs()) > (1.0/9.0)
       good=(cat & (np.array(data["gas", "velocity_divergence"])<0))& (dlogP > ep)
#     good=dlogP > ep
     mouse = (np.exp(dlogP)-1.0)*0.4 + 1
     new_field[good]=mouse[good]
     return new_field
  ds.add_field(('gas','ShockMach'), function=_ShockMach, units="", take_log=False,display_name='Shock Mach')
  print("begin hot shell")
  sp=ds.sphere([0.5, 0.5, 0.5],(r_max, 'kpc'))
#  hot_sp=sp.cut_region(["obj['temperature'] > 1e6","obj['temperature'] < 1e8"])
  hot_sp=sp.cut_region(["obj['temperature'] > 1e6","obj['temperature'] < 1e8","obj['ShockMach']<1.05"])
  #hot_sp=sp.cut_region(["obj['temperature'] > 1e6","obj['temperature'] < 1e8","obj[('index', 'radius')].in_units('kpc') > 10"])
    # Randomly draw a sample of pixels in this sphere, and store position and index in points
#  points = rand_shell(hot_sp, n_points)  # n_points in obs is 7000 pixels
  (dist,v_diff,r1,r2)=vsf(hot_sp, n_points)
  np.savez(savedir+stest+"_vsf_hot_%i"%r_max+"kpc.npz", dist=dist,v_diff=v_diff,r1=r1,r2=r2)
  (dist_final,v_diff_mean_hot,v_diff_mean_in_hot,v_diff_mean_out_hot)=make_one_vsf(dist,v_diff,r1,r2,d_max=r_max,n_bins=180)
  print("begin cold shell")
  cold_sp=sp.cut_region("obj['temperature'] < 3e4")
  (dist,v_diff,r1,r2)=vsf(cold_sp, n_points)
  np.savez(savedir+stest+"_vsf_cold_%i"%r_max+"kpc.npz", dist=dist,v_diff=v_diff,r1=r1,r2=r2)
  (dist_final,v_diff_mean_cold,v_diff_mean_in_cold,v_diff_mean_out_cold)=make_one_vsf(dist,v_diff,r1,r2,d_max=r_max,n_bins=180)
  print("begin hot-cold correlation")
  (dist,v_diff,r1,r2)=vsf12(cold_sp, hot_sp, n_points)
  np.savez(savedir+stest+"_vsf_cold_hot_%i"%r_max+"kpc.npz", dist=dist,v_diff=v_diff,r1=r1,r2=r2)
  (dist_final,v_diff_mean_cold_hot,v_diff_mean_in_cold_hot,v_diff_mean_out_cold_hot)=make_one_vsf(dist,v_diff,r1,r2,d_max=r_max,n_bins=180)
  np.savez(savedir+stest+"_vsf_final.npz", dist_final=dist_final,v_diff_mean_hot=v_diff_mean_hot,v_diff_mean_cold=v_diff_mean_cold,\
  v_diff_mean_cold_hot=v_diff_mean_cold_hot,v_diff_mean_in_hot=v_diff_mean_in_hot,v_diff_mean_out_hot=v_diff_mean_out_hot,\
  v_diff_mean_in_cold=v_diff_mean_in_cold,v_diff_mean_out_cold=v_diff_mean_out_cold, \
  v_diff_mean_in_cold_hot=v_diff_mean_in_cold_hot,v_diff_mean_out_cold_hot=v_diff_mean_out_cold_hot)
  return

def make_one_vsf(dist,v_diff,r1,r2,d_max,n_bins=180,plot_hist):
 dist_cold=dist*LengthUnits/kpc
 #v_diff_cold=np.sqrt(cold["v_diff"])
 v_diff_cold=v_diff
 r1=cold["r1"]*LengthUnits/kpc
 r2=cold["r2"]*LengthUnits/kpc
 dist_round=np.round(dist_cold,decimals=3)
 unique= np.unique(dist_round)
 dist_array=np.append(unique[1:11],np.logspace(np.log10(unique[11]),np.log10(d_max*3),n_bins-10))
 v_diff_mean_cold=np.zeros(n_bins)
 v_diff_mean_in=np.zeros(n_bins)
 v_diff_mean_out=np.zeros(n_bins)
 for i in range(0,n_bins-1):
    this_bin=(dist_cold>=dist_array[i])&(dist_cold<dist_array[i+1])
    v_diff_mean_cold[i]=np.mean(v_diff_cold[this_bin])
    if plot_hist==True:
      pylab.hist(v_diff_cold[this_bin],bins=50)
      pylab.savefig(savedir+stest+"v_distribution%i.png"%i)
    isin=this_bin&(r1<12)&(r2<12)
    v_diff_mean_in[i]=np.mean(v_diff_cold[isin])
    isout=this_bin&(r1>=12)&(r2>=12)
    v_diff_mean_out[i]=np.mean(v_diff_cold[isout])
 return(dist_cold,v_diff_mean_cold,v_diff_mean_in,v_diff_mean_out)

def make_plot_vsf(file_name="stest_0193_vsf_",d_max=40,n_bins=180):
  f = pylab.figure(figsize = (10,8))
  (dist_array,v_diff_mean,v_diff_mean_in,v_diff_mean_out)=make_one_vsf(file_npz=file_name+"cold_40kpc.npz")
  y_expect=dist_array**(1.0/3)*300
  pylab.loglog(dist_array,y_expect,linestyle="-",label="1/3")
  pylab.loglog(dist_array,v_diff_mean,marker="o",linestyle="None",markersize=4,label = 'Cold 3D',color="blue")
  (dist_array,v_diff_mean,v_diff_mean_in,v_diff_mean_out)=make_one_vsf(file_npz=file_name+"hot_40kpc.npz")
  pylab.loglog(dist_array,v_diff_mean,marker="o",linestyle="None",markersize=4,label = 'Hot 3D',color="red")
  pylab.loglog(dist_array,v_diff_mean_in,marker="o",linestyle="None",markersize=2,label = 'Hot 3D < 10 kpc',color="darkred")
  pylab.loglog(dist_array,v_diff_mean_out,marker="o",linestyle="None",markersize=2,label = 'Hot 3D > 10kpc',color="pink")
  (dist_array,v_diff_mean,v_diff_mean_in,v_diff_mean_out)=make_one_vsf(file_npz=file_name+"cold_hot_40kpc.npz")
  pylab.loglog(dist_array,v_diff_mean,marker="o",linestyle="None",markersize=4,label = 'Cold Hot 3D',color="grey")
  pylab.plot([7,14],[185,197],linestyle="-",color="blue",linewidth=4,alpha=0.5,label="Irina 0-30 kpc")
  pylab.plot([10,20],[110,140],linestyle="-",color="green",linewidth=4,alpha=0.5,label="Irina 30-60 kpc")
  pylab.plot([0.052,0.104],[27,27],linestyle="-",color="k",label="Virgo",marker="|",linewidth=2)
#  pylab.ylim(20,1000)
  pylab.xlabel("separation (kpc)")
  pylab.ylabel(r"$|\delta v|$")
  pylab.legend(loc="best", prop={'size': 16})
  pylab.grid()
  pylab.savefig(file_name+".png")
  return





def vsf(ds, n_points):
    print("start drawing rand_shell")
    points = rand_shell(ds, n_points)  # n_points in obs is 7000 pixels
    print("Done drawing rand_shell in ds")
    dist=[]
    v_diff=[]
    v_diff_111=[]
    angle12=[]
    angle_from_1=[]
    r1_list=[]
    r2_list=[]
    dist_111=[]
    velx_a = np.reshape(points[:][5], (n_points, 1))
    velx_b = np.reshape(points[:][5], (1, n_points))
    vely_a = np.reshape(points[:][6], (n_points, 1))
    vely_b = np.reshape(points[:][6], (1, n_points))
    velz_a = np.reshape(points[:][7], (n_points, 1))
    velz_b = np.reshape(points[:][7], (1, n_points))
    v_diff_matrix = np.sqrt((velx_a - velx_b)**2+(vely_a - vely_b)**2+(velz_a - velz_b)**2)

    px_a = np.reshape(points[:][1], (n_points, 1))
    px_b = np.reshape(points[:][1], (1, n_points))
    py_a = np.reshape(points[:][2], (n_points, 1))
    py_b = np.reshape(points[:][2], (1, n_points))
    pz_a = np.reshape(points[:][3], (n_points, 1))
    pz_b = np.reshape(points[:][3], (1, n_points))
    dist_matrix = np.sqrt((px_a - px_b)**2 + (py_a - py_b)**2 + (pz_a - pz_b)**2)

    #now compute r
    acx=np.zeros((1,n_points))+0.5
    acy=np.zeros((1,n_points))+0.5
    acz=np.zeros((1,n_points))+0.5
    bcx=np.zeros((n_points,1))+0.5
    bcy=np.zeros((n_points,1))+0.5
    bcz=np.zeros((n_points,1))+0.5

    rA_matrix=np.sqrt((px_a - acx)**2 + (py_a - acy)**2+(pz_a - acz)**2)
    rB_matrix=np.sqrt((px_b - bcx)**2 + (py_b - bcy)**2+(pz_b - bcz)**2)

    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(np.triu(dist_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0
    rA_half = np.ndarray.flatten(np.triu(rA_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0
    rB_half = np.ndarray.flatten(np.triu(rB_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0

    good_dist = dist_half>0

    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    rA=rA_half[good_dist]
    rB=rB_half[good_dist]
    return np_dist, np_v_diff, rA, rB

def vsf12(ds1, ds2, n_points):
    print("start drawing rand_shell")
    points1 = rand_shell(ds1, n_points)  # n_points in obs is 7000 pixels
    points2 = rand_shell(ds2, n_points)  # n_points in obs is 7000 pixels
    print("Done drawing rand_shell in ds")
    velx_a = np.reshape(points1[:][5], (n_points, 1))
    velx_b = np.reshape(points2[:][5], (1, n_points))
    vely_a = np.reshape(points1[:][6], (n_points, 1))
    vely_b = np.reshape(points2[:][6], (1, n_points))
    velz_a = np.reshape(points1[:][7], (n_points, 1))
    velz_b = np.reshape(points2[:][7], (1, n_points))
    v_diff_matrix = np.sqrt((velx_a - velx_b)**2+(vely_a - vely_b)**2+(velz_a - velz_b)**2)

    px_a = np.reshape(points1[:][1], (n_points, 1))
    px_b = np.reshape(points2[:][1], (1, n_points))
    py_a = np.reshape(points1[:][2], (n_points, 1))
    py_b = np.reshape(points2[:][2], (1, n_points))
    pz_a = np.reshape(points1[:][3], (n_points, 1))
    pz_b = np.reshape(points2[:][3], (1, n_points))
    dist_matrix = np.sqrt((px_a - px_b)**2 + (py_a - py_b)**2 + (pz_a - pz_b)**2)

    #now compute r
    acx=np.zeros((1,n_points))+0.5
    acy=np.zeros((1,n_points))+0.5
    acz=np.zeros((1,n_points))+0.5
    bcx=np.zeros((n_points,1))+0.5
    bcy=np.zeros((n_points,1))+0.5
    bcz=np.zeros((n_points,1))+0.5

    rA_matrix=np.sqrt((px_a - acx)**2 + (py_a - acy)**2+(pz_a - acz)**2)
    rB_matrix=np.sqrt((px_b - bcx)**2 + (py_b - bcy)**2+(pz_b - bcz)**2)

    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(np.triu(dist_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0
    rA_half = np.ndarray.flatten(np.triu(rA_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0
    rB_half = np.ndarray.flatten(np.triu(rB_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0

    good_dist = dist_half>0

    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    rA=rA_half[good_dist]
    rB=rB_half[good_dist]
    return np_dist, np_v_diff, rA, rB

def rand_shell(shell, n_points):
    #get random points with positions stored in points
  length=len(shell['density'])
  ind=random.sample(range(length),n_points)
  print("len(shell['density'])",length)
  x = shell['x'][ind].in_units('code_length')
  y = shell['y'][ind].in_units('code_length')
  z = shell['z'][ind].in_units('code_length')
  v_111=shell['velocity_111'][ind].in_units('km/s')
  v_x=shell['velocity_x'][ind].in_units('km/s')
  v_y=shell['velocity_y'][ind].in_units('km/s')
  v_z=shell['velocity_z'][ind].in_units('km/s')
  points=np.array([ind,x,y,z,v_111,v_x,v_y,v_z])
  return points



#derived field in yt
def my_radial_velocity(field, data):
    xv = data["gas","velocity_x"]
    yv = data["gas","velocity_y"]
    zv = data["gas","velocity_z"]
    center = [0,0,0]
    x_hat = data["x"] - center[0]
    y_hat = data["y"] - center[1]
    z_hat = data["z"] - center[2]
    r = np.sqrt(x_hat*x_hat+y_hat*y_hat+z_hat*z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    return xv*x_hat + yv*y_hat + zv*z_hat

def my_velocity_theta(field, data):
    xv = data['gas','velocity_x']
    yv = data['gas','velocity_y']
    zv = data['gas','velocity_z']
    center = [0,0,0]
    x_hat = data['x'] - center[0]
    y_hat = data['y'] - center[1]
    z_hat = data['z'] - center[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    theta_v = (xv*y_hat - x_hat*yv)/(x_hat*x_hat + y_hat*y_hat)*rxy
    return theta_v

def my_velocity_phi(field, data):
    xv = data['gas','velocity_x']
    yv = data['gas','velocity_y']
    zv = data['gas','velocity_z']
    center = [0,0,0]
    x_hat = data['x'] - center[0]
    y_hat = data['y'] - center[1]
    z_hat = data['z'] - center[2]
    r = np.sqrt(x_hat*x_hat + y_hat*y_hat + z_hat*z_hat)
    rxy = np.sqrt(x_hat*x_hat + y_hat*y_hat)
    phi_v = (z_hat*(x_hat*xv + y_hat*yv)-zv*(x_hat*x_hat + y_hat*y_hat))/(r*r*rxy)*r
    return phi_v

def velocity_111(field,data):
    return (data['gas','velocity_x']+data['gas','velocity_y']+data['gas','velocity_z'])/np.sqrt(3.0)

# Constants
mu = 0.62
muH = 1./0.7
met_Z = 0.33333
gamma = 5./3.

# Add derived fields
yt.add_field(("gas","rv"), function=my_radial_velocity, units="km/s", take_log=False, \
             display_name='Radial velocity')
yt.add_field(("gas","v_theta"), function=my_velocity_theta, units="km/s", take_log=False, \
             display_name="Velocity_theta")
yt.add_field(("gas","v_phi"), function=my_velocity_phi, units="km/s", take_log=False, \
             display_name="Velocity_phi")
yt.add_field(("gas","velocity_111"), function=velocity_111, units="km/s", take_log=False, \
             display_name="Velocity_111")

from yt import derived_field
@derived_field(name="logP",units="")
def _logP(field,data):
    return np.log(np.array(data["gas","pressure"]))

#vsf_all(r_max=40,n_points=5000,dir="/Reo3Fd/",imin=57)
for imin in np.arange(60,65):
  vsf_all(r_max=40,n_points=20000,dir="./",imin=imin)
