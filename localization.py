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

xn = np.array(( 37.6133603333333,
                37.6133198666667,
                37.6132646,
                37.61321225,
                37.6131987833333,
                37.6132384666667,
                37.6132954,
                37.6133465));
               
yn = np.array(( -119.021031533333,
                -119.02096225,
                -119.020950133333,
                -119.020994066667,
                -119.02106895,
                -119.021139883333,
                -119.021151716667,
                -119.0211086));
               
zn = np.array(( 2737.456704,
                2737.276664,
                2737.345904,
                2737.602288,
                2737.702424,
                2737.897384,
                2738.013144,
                2737.757144));

coords = [(37.6133603333333,-119.021031533333,2737.456704),
(37.6133198666667,-119.02096225,2737.276664),
(37.6132646,-119.020950133333,2737.345904),
(37.61321225,-119.020994066667,2737.602288),
(37.6131987833333,-119.02106895,2737.702424),
(37.6132384666667,-119.021139883333,2737.897384),
(37.6132954,-119.021151716667,2738.013144),
(37.6133465,-119.0211086,2737.757144)];

#    sw_corner = (37.56821,-119.07787);
#    ne_corner = (37.65835,-118.96423);
#xc = sw_corner[0];
#yc = sw_corner[1];
#    zc = 2735.643311532;
xc = min(xn);
yc = min(yn);

REF_J   = 1;
XY = False;

# read in the data from the csvs
node_data   = ce240data.get_nodes();
sensor_data = ce240data.get_sensor_logs();
flight_data = ce240data.get_flight_logs();

#xn = np.array(node_data[:,[0]]);
#yn = np.array(node_data[:,[1]]);
#xc,yc,r,_ = cf.least_squares_circle(tuple(zip(xn,yn)));
OBS = xc, yc, 2735.643311532;
#OBS = xn,yc,zn;

nodexyz = ce240.get_points(node_data[:,[0]],node_data[:,[1]],node_data[:,[2]],OBS);
#nodes = np.array(fc.convert_coords(coords));
print("GOT NODES")
truexyz = ce240.get_points(flight_data[0],flight_data[1],flight_data[2],OBS);

ufo_x = truexyz[:,[0]];
ufo_y = truexyz[:,[1]];
ufo_z = truexyz[:,[2]];
ufo_t = flight_data[3];

# TODO move this to dataread.py
if XY is True:
    nodes = ce240.get_points2D(xn,yn,OBS);
    xn,yn = np.hsplit(nodes,2); # legacy to get the x,y,z vectors
else:
    nodes = ce240.get_points(xn,yn,zn,OBS);
    xn,yn,zn = np.hsplit(nodes,3); # legacy to get the x,y,z vectors
    #xn = nodes[0];yn = nodes[1]; zn = nodes[2];

# get the time series xls output
if XY is True:
    calcxyz,t = ce240.get_series_out2D(nodes,sensor_data, REF_J);
else:
    calcxyz,t = ce240.get_series_out(nodes,sensor_data, REF_J);


A = ce240.get_lmatrix(nodes,0);
newxyz = np.array([np.dot(A,x) for x in truexyz]);

"""
Plots nodes on a 3D coordinate system.
Anchor nodes are in black.
Mobile node is in green.
"""

fig = plt.figure()

if XY is True:
    ax = plt.axes()    
    x, y = list(zip(*calcxyz));
    x = np.asarray(x);
    y = np.asarray(y);
    
    # color gradient
    cm = plt.get_cmap("RdYlGn");
    grad = [cm(float(i)/(len(x))) for i in range(len(x))];
    
    ax.scatter(xn,yn,c='Black',s=10);
    ax.scatter(x, y, s=1, c=grad);
    
else:
    ax = plt.axes(projection='3d')
    # XYZ location of quadcopter (node 9)
    #x = -2455200.7931187
    #y = -4425468.23573119
    #z = 3873216.13338933
    
    x, y, z = list(zip(*calcxyz));
    x = np.asarray(x);
    y = np.asarray(y);
    z = np.asarray(z);
    
    # color gradient
    cm = plt.get_cmap("Spectral");
    grad = [cm(float(i)/(len(x))) for i in range(len(x))];
    
    
    # Print locations of nodes. Node 9 is in a different color
    ax.scatter(xn, yn, zn, c='Black', s=10, depthshade=False);
    #ax.scatter(ufo_x,ufo_y,ufo_z, s=1, c=grad);
    ax.scatter(x, y, z, s=1, c=grad, depthshade=False); # c='Green',
    
    
    ax.set_xlim3d(min(x),max(x))
    ax.set_ylim3d(min(y),max(y))
    ax.set_zlim3d(min(z),max(z))
    
    """"ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')"""
    
    
    # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
    ax.text2D(0.05, 0.95, "ECEF Coordinates of Anchor and Mobile Nodes", transform=ax.transAxes)

plt.show();
#print(max(ufo_x))
#print(max(nodexyz[:,[0]]))
#print(max(truexyz[:,[1]]))
#print(max(nodexyz[:,[1]]))
#print(max(truexyz[:,[2]]))
#print(max(nodexyz[:,[2]]))
#print(max(x))
#print(max(y))
#print(max(z))

#out_file = open('output/real_distances.csv', 'w+')
#
#for i in range(len(ufo_x)):
#    out_file.write(str(ufo_x[i][0]) + "," + 
#                   str(ufo_y[i][0]) + "," + 
#                   str(ufo_z[i][0]) + "," + 
#                   ufo_t[i] + "\n")
#    
#out_file.close();

out_file = open('output/calc_distances.csv', 'w+')

for i in range(len(x)):
    out_file.write(str(x[i]) + "," + str(y[i]) + "," + str(z[i]) + "," + t[i] + "\n")
    
out_file.close();