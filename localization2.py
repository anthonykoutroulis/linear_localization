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

REF_J   = 5;
xn = np.array(( -2455199.6273,
				-2455195.5372,
				-2455196.4449,
				-2455201.6579,
				-2455207.9231,
				-2455212.1721,
				-2455211.2588,
				-2455206.1501));

yn = np.array((	-4425465.5558,
				-4425470.7980,
				-4425474.6403,
				-4425476.0375,
				-4425473.6961,
				-4425468.4399,
				-4425464.6391,
				-4425463.2812));


zn = np.array((	3873208.9598,
				3873205.2905,
				3873200.4717,
				3873196.0235,
				3873194.9002,
				3873198.5096,
				3873203.5880,
				3873207.9264));

zm = np.array(( 2737.456704,
                2737.276664,
                2737.345904,
                2737.602288,
                2737.702424,
                2737.897384,
                2738.013144,
                2737.757144));

#create the origin
x0 = -2455220.4875; y0 = -4425487.0204; z0 = 3873193.8678;
xn -= x0;
yn -= y0;
#zn -= z0;
zn = zm - min(zm) + 1;

nodes = np.array([(x,y,z) for x,y,z in zip(xn,yn,zn)]);

# read in the data from the csvs
sensor_data = ce240data.get_sensor_logs();

# calculate xls
d_ = ce240.get_dist_vector(nodes,REF_J);
A  = ce240.get_lmatrix(nodes,REF_J);
r_ = ce240.get_radii(sensor_data);

xyz, t = ce240.get_series_out(nodes, sensor_data, REF_J)
x,y,z = np.hsplit(np.array(xyz),3);

#out_file = open('output/calc_distances.csv', 'w+')
#for i in range(len(x)):
#    out_file.write(str(x[i][0]) + "," + str(y[i][0]) + "," + str(z[i][0]) + "," + t[i] + "\n")
#out_file.close();



x, y, z = list(zip(*xyz));
x = np.asarray(x);
x = xn[REF_J] + x;
y = np.asarray(y);
y = yn[REF_J] + y;
z = np.asarray(z);
z = zn[REF_J] + z;


r8 = r_[:,[7]];
r7 = r_[:,[6]];
r6 = r_[:,[5]];
r5 = r_[:,[4]];
r4 = r_[:,[3]];
r3 = r_[:,[2]];
r2 = r_[:,[1]];
r1 = r_[:,[0]];
n = np.linspace(0,len(r_[:,[0]]),len(r_[:,[0]]));

f1 = plt.figure(1)
ax = plt.axes(projection='3d')
cm = plt.get_cmap("Spectral");
grad = [cm(float(i)/(len(x))) for i in range(len(x))];
ax.scatter(xn, yn, zn, c='Black', s=10, depthshade=False);
ax.scatter(x, y, z, s=1, c=grad, depthshade=False); # c='Green',
ax.set_xlim3d(min(x),max(x))
ax.set_ylim3d(min(y),max(y))
ax.set_zlim3d(min(z),max(z))
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
ax.scatter(xn,yn,s=2, c='Black');
ax.scatter(x, y, s=0.5, c=grad); # c='Green',
f3.show();
