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

# convert gps lla to aeqd xyz
def gps_to_aeqd(lat,lon,alt):
    ecef = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84');
    lla  = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84');
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

# get truth data
truth = ce240data.get_flight_logs();
xt,yt,zt = gps_to_aeqd(truth[0],truth[1],truth[2]);
truth = np.array([(x,y,z) for x,y,z in zip(xt,yt,zt)]);

# get node locations
nodes = ce240data.get_node_locs();
xn,yn,zn = gps_to_aeqd(nodes[0],nodes[1],nodes[2]);
nodes = np.array([(x,y,z) for x,y,z in zip(xn,yn,zn)]);

# get sensor logs and predictions
sensor_logs = ce240data.get_sensor_logs();
xyz, t = ce240.get_series_out(nodes, sensor_logs, REF_J);
x,y,z = np.hsplit(np.array(xyz),3);

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

# verify by calculating pythagorian distance to node j+1
a = xt[REF_J+1] - xn[REF_J+1];
b = yt[REF_J+1] - yn[REF_J+1];
c = zt[REF_J+1] - zn[REF_J+1];
h = np.sqrt(np.square(a) + np.square(b) + np.square(c));
print("b_="+str(b_[0])+", h="+str(h));

sensor_logs = ce240data.get_sensor_logs();
xyz, t = ce240.get_series_out(nodes, sensor_logs, REF_J);
x,y,z = np.hsplit(np.array(xyz),3);
x += xn[REF_J];
y += yn[REF_J];
z += zn[REF_J];

# find residuals (5 is experimentally determined consecutive series)
samp = np.hstack((x[:5],y[:5],z[:5]));
dxt = downsample(np.array(xt),2);
dyt = downsample(np.array(yt),2);
dzt = downsample(np.array(zt),2);
dtruth = np.array([(i,j,k) for i,j,k in zip(dxt,dyt,dzt)]);
residuals = find_residuals(dtruth,samp);

# offset truth to that index
match = np.where(residuals == min(residuals));
match = match[0][0]; d = match + len(samp);
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
f1.show();

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
f2.show();

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
f3.show();