"""
Create truth model
"""
# our libraries
import src.dataread as ce240data;
import src.model as ce240;

# external libraries
import numpy as np;
import pymap3d as pm;
from mpl_toolkits import mplot3d;
import matplotlib.pyplot as plt;
import circle_fit as cf;
import src.farid_conv as fc;

import pyproj;
import scipy as sp;
import scipy.signal as sg;
import math;
import datetime;

from scipy import optimize;

def func_f(guess,node,r):
    r_ = ce240.get_dist(guess,node);
    return r_ - r
                   
def func_F(guess,nodes,radii):
    errors = np.array([0.0,0.0,0.0]);
    for i in range(len(nodes)):
        errors += np.square(func_f(guess,nodes[i],radii[i]));
    return errors;

# convert gps lla to aeqd xyz
def gps_to_aeqd(lat,lon,alt):
#    ecef = pyproj.Proj(proj='epsg:2276', ellps='WGS84', datum='WGS84');
    lla  = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84');
    ecef = pyproj.Proj(init='epsg:32711');
    x,y,z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False);
    return x,y,z

# perform a rolling average, preserving original and tallying deltas
def find_residuals(truth,sample): # must be np.arrays
    residuals = []; d = len(sample); 
    for i in range(0,len(truth)-len(sample)):
        r = abs(truth[i:i+d] - sample);
        residuals.append(np.average(r));
    return np.array(residuals); # user should find minimal index

def downsample(data, fs):
    # author: shx2, 
    # https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
    pad = math.ceil(float(data.size)/fs)*fs - data.size;
    sample = np.append(data, np.zeros(pad)*np.NaN);
    return sp.nanmean(sample.reshape(-1,fs), axis=1);


REF_J = 0;

# get node locations
nodes = ce240data.get_node_locs();
xn,yn,zn = gps_to_aeqd(nodes[0],nodes[1],nodes[2]);
nodes = np.array([(i,j,k) for i,j,k in zip(xn,yn,zn)]);

# get sensor logs and predictions
sensor_logs = ce240data.get_sensor_logs();
xyz, t = ce240.get_series_out(nodes, sensor_logs, REF_J);
r = ce240.get_radii(sensor_logs);
x,y,z = np.hsplit(np.array(xyz),3);
t = np.array(t)

# get truth data
truth = ce240data.get_flight_logs();
xt,yt,zt = gps_to_aeqd(truth[0],truth[1],truth[2]);
xt = np.array(xt); yt = np.array(yt); zt = np.array(zt);
tt = np.array(truth[3]);
truth = np.array([(i,j,k) for i,j,k in zip(xt,yt,zt)]);

# calculate the radii of each node to the truthdrone
r_ = np.array([
        [ce240.get_dist(t,node_i) for node_i in nodes] 
        for t in truth]);
    
# create and add error
rand = np.random.rand(1412,8);
r_ += rand;
    
# compare the min/max of both radii vectors
mmr_ = np.array([(min(r_[:,[i]]),max(r_[:,[i]])) for i in range(0,8)]);
mmr  = np.array([(min(r[:,[i]]),max(r[:,[i]])) for i in range(0,8)]);

# get the distances between nodes and the reference
d_ = ce240.get_dist_vector(nodes,REF_J);

# get the linear matrix
A_ = ce240.get_lmatrix(nodes,REF_J);

# set up true input vector
x_ = np.array((np.array(xt) - xn[REF_J],
               np.array(yt) - yn[REF_J],
               np.array(zt) - zn[REF_J])); 
               
# calculate the output b_
b_ = np.dot(A_,x_);

bb_ = np.array([ce240.get_ldist_vector(i,d_,REF_J) for i in r_]);
xls = ce240.get_xls(A_,bb_.transpose(),scale=1);
#dxt,dyt,dzt = np.hsplit(np.array(xls),3);
dxt = xls[0];
dyt = xls[1];
dzt = xls[2];
xnls = []

for i in range(len(r_)):
    p = xls.transpose()[i];
    rad = r_[i];
    result,_ = optimize.leastsq(func_F, p, args=(nodes,rad));
    xnls.append(result);


dxt = dxt + xn[REF_J];
dyt = dyt + yn[REF_J];
dzt = dzt + zn[REF_J];
         
# determine time breaks 
break10s = np.argwhere(t[1:] - t[:-1] > datetime.timedelta(seconds = 10));
break10m = np.argwhere(t[1:] - t[:-1] > datetime.timedelta(minutes = 5));
tbreak10s = np.argwhere(tt[1:] - tt[:-1] > datetime.timedelta(seconds = 10));
tbreak10m = np.argwhere(tt[1:] - tt[:-1] > datetime.timedelta(minutes = 5));


# verify by calculating pythagorian distance to node j+1
a = xt[REF_J+1] - xn[REF_J+1];
b = yt[REF_J+1] - yn[REF_J+1];
c = zt[REF_J+1] - zn[REF_J+1];
h = np.sqrt(np.square(a) + np.square(b) + np.square(c));
print("b_="+str(b_[0])+", h="+str(h));

# regular xls calculations
sensor_logs = ce240data.get_sensor_logs();
xyz, t = ce240.get_series_out(nodes, sensor_logs, REF_J);
t = np.array(t);
x,y,z = np.hsplit(np.array(xyz),3);
x += xn[REF_J];
y += yn[REF_J];
z += zn[REF_J];

# find residuals (5 is experimentally determined consecutive series)
#samp = np.hstack((x[:5],y[:5],z[:5]));
#dxt = downsample(np.array(xt),2);
#dyt = downsample(np.array(yt),2);
#dzt = downsample(np.array(zt),2);
#dtruth = np.array([(i,j,k) for i,j,k in zip(dxt,dyt,dzt)]);
#residuals = find_residuals(dtruth,samp);



# offset truth to that index
#match = np.where(residuals == min(residuals));
#match = match[0][0]; d = match + len(samp);
#xt=xt[match:d]; yt=yt[match:d]; zt=zt[match:d];
#x=x[match:d]; y=y[match:d]; z=z[match:d];

## Plotting
f1 = plt.figure(1)
ax = plt.axes(projection='3d')
cm = plt.get_cmap("Spectral");
grad = [cm(float(i)/(len(xt))) for i in range(len(xt))];
ax.scatter(xn, yn, zn, c='Black', s=10, depthshade=False);
ax.scatter(xt, yt, zt, s=1, c=grad, depthshade=False); # c='Green',
ax.set_xlim3d(min(xn),max(xn))
ax.set_ylim3d(min(yn),max(yn))
ax.set_zlim3d(min(min(zt),min(zn)),max(zt))
ax.text2D(0.05, 0.95, "Truth Data with Projection", transform=ax.transAxes)

## Plotting
f2 = plt.figure(2)
ax = plt.axes(projection='3d')
cm = plt.get_cmap("Spectral");
grad = [cm(float(i)/(len(x))) for i in range(len(x))];
ax.scatter(xn, yn, zn, c='Black', s=10, depthshade=False);
ax.scatter(x, y, z, s=1, c=grad, depthshade=False); # c='Green',
ax.set_xlim3d(min(xn),max(xn))
ax.set_ylim3d(min(yn),max(yn))
ax.set_zlim3d(min(min(z),min(zn)),max(z))
ax.text2D(0.05, 0.95, "Calculated with Projection", transform=ax.transAxes)

## Plotting
f3 = plt.figure(3)
ax = plt.axes(projection='3d')
cm = plt.get_cmap("Spectral");
grad = [cm(float(i)/(len(dxt))) for i in range(len(dxt))];
ax.scatter(xn, yn, zn, c='Black', s=10, depthshade=False);
ax.scatter(dxt,dyt,dzt, s=1, c=grad, depthshade=False); # c='Green',
ax.set_xlim3d(min(xn),max(xn))
ax.set_ylim3d(min(yn),max(yn))
ax.set_zlim3d(min(min(dzt),min(zn)),max(dzt))
ax.text2D(0.05, 0.95, "Downsampled truth data", transform=ax.transAxes)

## Prepare and plot the radii graph
r8 = r[:,[7]];
r7 = r[:,[6]];
r6 = r[:,[5]];
r5 = r[:,[4]];
r4 = r[:,[3]];
r3 = r[:,[2]];
r2 = r[:,[1]];
r1 = r[:,[0]];
n = np.linspace(0,len(r[:,[0]]),len(r[:,[0]]));
r_8 = r_[:,[7]];
r_7 = r_[:,[6]];
r_6 = r_[:,[5]];
r_5 = r_[:,[4]];
r_4 = r_[:,[3]];
r_3 = r_[:,[2]];
r_2 = r_[:,[1]];
r_1 = r_[:,[0]];
n_ = np.linspace(0,len(r_[:,[0]]),len(r_[:,[0]]));

f4 = plt.figure(4)
ax = plt.axes()
ax.plot(n,r1,c='C1',label='r1')
ax.plot(n,r2,c='C2',label='r2')
ax.plot(n,r3,c='C3',label='r3')
ax.plot(n,r4,c='C4',label='r4')
ax.plot(n,r5,c='C5',label='r5')
ax.plot(n,r6,c='C6',label='r6')
ax.plot(n,r7,c='C7',label='r7')
ax.plot(n,r8,c='C8',label='r8')
for vline in break10s:
    plt.axvline(x=vline, color='k', linestyle='--', lw=0.5);
for vline in break10m:
    plt.axvline(x=vline, color='k', linestyle='--', lw=2);
ax.text(0.05, 0.95, "Node radius over time (sensor logs)")

f5 = plt.figure(5)
ax = plt.axes()
ax.plot(n_,r_1,c='C1',label='r1')
ax.plot(n_,r_2,c='C2',label='r2')
ax.plot(n_,r_3,c='C3',label='r3')
ax.plot(n_,r_4,c='C4',label='r4')
ax.plot(n_,r_5,c='C5',label='r5')
ax.plot(n_,r_6,c='C6',label='r6')
ax.plot(n_,r_7,c='C7',label='r7')
ax.plot(n_,r_8,c='C8',label='r8')
for vline in tbreak10s:
    plt.axvline(x=vline, color='k', linestyle='--', lw=0.5);
for vline in tbreak10m:
    plt.axvline(x=vline, color='k', linestyle='--', lw=2);
ax.text(0.05, 0.95, "Node radius over time (truth calc)")



f1.show();
f2.show();
f3.show();
f4.show();
f5.show();

#rpeak = []
#rpeak.append(np.argwhere(r2[0:1000] == max(r2[0:1000]))[0][0]);
#rpeak.append(np.argwhere(r2[2000:2200] == max(r2[2000:2200]))[0][0]+2000);
#rpeak.append(np.argwhere(r2[2800:2900] == max(r2[2800:2900]))[0][0]+2800);
#rpeak.append(np.argwhere(r2[3250:3300] == max(r2[3200:3300]))[0][0]+3250);
#
#r_peak = []
#r_peak.append(np.argwhere(r_2[0:1500] == max(r_2[0:1500]))[0][0]);
#r_peak.append(np.argwhere(r_2[2000:3200] == max(r_2[2000:3200]))[0][0]+2000);
#r_peak.append(np.argwhere(r_2[3500:4500] == max(r_2[3500:4500]))[0][0]+3500);
#r_peak.append(np.argwhere(r_2[5000:6000] == max(r_2[5000:6000]))[0][0]+5000);
#
#tdiff = tt[r_peak] - t[rpeak];