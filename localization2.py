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


REF_J   = 5; # node *index* to linearize about (REF_J=5 -> node 6)
# min/max of distance (from csv) and distance (trig on flight data)
IN_MIN = 3715.88; IN_MAX = 27001.1; OUT_MIN = 5.2703794; OUT_MAX = 78.3703818;

## Begin coordinate transformation constants/manipulation WIP Experimental! ##
##############################################################################

#xn = np.array((-2455199.6273,
#				-2455195.5372,
#				-2455196.4449,
#				-2455201.6579,
#				-2455207.9231,
#				-2455212.1721,
#				-2455211.2588,
#				-2455206.1501));
#
#yn = np.array((	-4425465.5558,
#				-4425470.7980,
#				-4425474.6403,
#				-4425476.0375,
#				-4425473.6961,
#				-4425468.4399,
#				-4425464.6391,
#				-4425463.2812));
#
#
#zn = np.array((3873208.9598,
#				3873205.2905,
#				3873200.4717,
#				3873196.0235,
#				3873194.9002,
#				3873198.5096,
#				3873203.5880,
#				3873207.9264));
#               
#zm = np.array(( 2737.456704,
#                2737.276664,
#                2737.345904,
#                2737.602288,
#                2737.702424,
#                2737.897384,
#                2738.013144,
#                2737.757144));
               
nodes = np.array([(-2455199.627, -4425465.556, 3873208.960),
                  (-2455195.537, -4425470.798, 3873205.291),
                  (-2455196.445, -4425474.640, 3873200.472),
                  (-2455201.658, -4425476.037, 3873196.023),
                  (-2455207.923, -4425473.696, 3873194.900),
                  (-2455212.172, -4425468.440, 3873198.509),
                  (-2455211.259, -4425464.639, 3873203.588),
                  (-2455206.150, -4425463.281, 3873207.926)]);

nodes = np.array([(2130566.175,624593.194,2737.457),
                  (2130572.362,624588.800,2737.277),
                  (2130573.528,624582.684,2737.346),
                  (2130569.742,624576.814,2737.602),
                  (2130563.155,624575.215,2737.702),
                  (2130556.824,624579.519,2737.897),
                  (2130555.680,624585.821,2738.013),
                  (2130559.396,624591.551,2737.757)]);




xn = nodes[:,[0]]; yn = nodes[:,[1]]; zn = nodes[:,[2]];
#chosen point for reference, center-mindrone alt - 2
origin =          (-2468963.271, -4427805.283, 3873199.813);
origin = (2114824.086, 624351.774, 2735.643);
#create the origin
# elevation at that point 2742.0m
# minimum drone = 2735.6433115m
#x0 = -2455220.4875; y0 = -4425487.0204; z0 = 3873193.8678;
x0 = origin[0]; y0 = origin[1]; z0 = origin[2];
#xn -= xn[REF_J];
#yn -= yn[REF_J];
#zn -= zn[REF_J];
xn -= x0;
yn -= y0;
zn -= z0;
#xn -= min(xn)
#yn -= min(yn)
#zn -= min(zn)
#zn = zm - min(zm) + 1;

#nodes = np.array([(x,y,z) for x,y,z in zip(xn,yn,zn)]);

## END coordinate transformation constants/manipulation WIP Experimental!   ##
##############################################################################

# read in the data from the csvs
sensor_data = ce240data.get_sensor_logs();

# calculate xls (these values made available for ipython, not used in plots)
d_ = ce240.get_dist_vector(nodes,REF_J);
A  = ce240.get_lmatrix(nodes,REF_J);
r_ = ce240.get_radii(sensor_data);

xyz, t = ce240.get_series_out(nodes, sensor_data, REF_J)
x,y,z = np.hsplit(np.array(xyz),3);
#x, y, z = list(zip(*xyz)); #TODO delete

# Add offset back in, depends on transformation
x = np.asarray(x);
x = xn[REF_J] + x;
y = np.asarray(y);
y = yn[REF_J] + y;
z = np.asarray(z);
z = zn[REF_J] + z;

# create radii,t vectors for each node and matching discrete space
r8 = r_[:,[7]];
r7 = r_[:,[6]];
r6 = r_[:,[5]];
r5 = r_[:,[4]];
r4 = r_[:,[3]];
r3 = r_[:,[2]];
r2 = r_[:,[1]];
r1 = r_[:,[0]];
n = np.linspace(0,len(r_[:,[0]]),len(r_[:,[0]]));

## Plotting
f1 = plt.figure(1)
ax = plt.axes(projection='3d')
cm = plt.get_cmap("Spectral");
grad = [cm(float(i)/(len(x))) for i in range(len(x))];
ax.scatter(xn, yn, zn, c='Black', s=10, depthshade=False);
ax.scatter(x, y, z, s=1, c=grad, depthshade=False); # c='Green',
ax.set_xlim3d(min(xn),max(xn))
ax.set_ylim3d(min(yn),max(yn))
ax.set_zlim3d(0,max(zn)+80)
ax.text2D(0.05, 0.95, "ECEF Coordinates of Anchor and Mobile Nodes", transform=ax.transAxes)
f1.show();

f2 = plt.figure(2)
ax = plt.axes()
ax.plot(n,r1,c='C1',label='r1')
ax.plot(n,r2,c='C2',label='r2')
ax.plot(n,r3,c='C3',label='r3')
ax.plot(n,r4,c='C4',label='r4')
ax.plot(n,r5,c='C5',label='r5')
ax.plot(n,r6,c='C6',label='r6')
ax.plot(n,r7,c='C7',label='r7')
ax.plot(n,r8,c='C8',label='r8')
f2.show();

f3 = plt.figure(3)
ax = plt.axes();
grad = [cm(float(i)/(len(x))) for i in range(len(x))];
ax.scatter(xn,yn,s=10, c='Black');
ax.scatter(x, y, s=0.5, c=grad); # c='Green',
f3.show();

## Create modified csv's
#out_file = open('output/calc_distances.csv', 'w+')
#for i in range(len(x)):
#    out_file.write(str(x[i][0]) + "," + str(y[i][0]) + "," + str(z[i][0]) + "," + t[i] + "\n")
#out_file.close();