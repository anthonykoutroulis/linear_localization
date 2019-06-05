import numpy as np;
import pymap3d as pm;

from scipy import interpolate;
from scipy import optimize;
import matplotlib.mlab as ml;
import src.farid_conv as fc;

# WIP apply lla->xyz transformation on vector
def get_points(xn,yn,zn,obs):

#    sw_corner = (37.56821,-119.07787);
#    ne_corner = (37.65835,-118.96423);
#    xc = sw_corner[0];
#    yc = sw_corner[1];
#    zc = 2735.643311532;

       # Convert lat,lon,alt to ECEF x,y,z
    #xn,yn,zn = pm.geodetic2ecef(xn,yn,zn)
#    print(min(xn));
    #xn,yn,zn = pm.geodetic2enu(xn,yn,zn,xc,yc,min(zn));

    #xi = np.linspace(min(xn),max(xn));
    #yi = np.linspace(min(yn),max(yn));
    #X,Y = np.meshgrid(xi,yi);
    #Z = interpolate.griddata((xn,yn,zn,xi,yi);
    #ell_clrk66 = pm.Ellipsoid('clrk66');
    #xn,yn,zn = pm.geodetic2ned(xn,yn,zn,obs[0],obs[1],obs[2],ell=ell_clrk66, deg=True);
    nodes = [(x,y,z) for x,y,z in zip(xn,yn,zn)];
    #nodes = [(x,y,z) for x,y,z in zip(xn,yn,zn)];

    return np.array(nodes);

def get_points2D(xn,yn,obs):
    X,Y,_ = pm.geodetic2enu(xn,yn,0,obs);
    nodes = [(x,y) for x,y in zip(X,Y)];
    return np.array(nodes);

# return euclidean distance b/w two points (x,y,z)
def get_dist(node_i, node_j):
    return np.sqrt(np.square(node_i[0] - node_j[0]) +
                   np.square(node_i[1] - node_j[1]) +
                   np.square(node_i[2] - node_j[2]));
                   
# return euclidean distance b/w two points (x,y)
def get_dist2D(node_i, node_j):
    return np.sqrt(np.square(node_i[0] - node_j[0]) +
                   np.square(node_i[1] - node_j[1]));

# get linearized radius to target with reference j
def get_ldist(rad_i, rad_j, dist_ij):
    return 0.5 * (np.square(rad_j) - np.square(rad_i) + np.square(dist_ij));

# compute single row of matrix A, reference j
def get_lrow_vector(node_i, node_j):
    return np.array([node_i[0] - node_j[0],
                     node_i[1] - node_j[1],
                     node_i[2] - node_j[2]]);

def get_lrow_vector2D(node_i, node_j):
    return np.array([node_i[0] - node_j[0],
                     node_i[1] - node_j[1]]);

# get_dist for a list of nodes, excepting reference node j
def get_dist_vector(nodes,ref_j):
    return [get_dist(node_i,nodes[ref_j]) for node_i in
            np.vstack((nodes[:ref_j],nodes[ref_j+1:]))];
            
def get_dist_vector2D(nodes,ref_j):
    return [get_dist2D(node_i,nodes[ref_j]) for node_i in
            np.vstack((nodes[:ref_j],nodes[ref_j+1:]))];

# get_ldist for a list of nodes, excepting reference node j
def get_ldist_vector(r_,d_,ref_j):
    return [get_ldist(rad_i,r_[ref_j],d) for rad_i,d in
            zip(np.hstack((r_[:ref_j],r_[ref_j+1:])),d_)];

# build complete matrix A
def get_lmatrix(nodes,ref_j):
    return np.array([get_lrow_vector(node_i,nodes[ref_j]) for node_i in
                     np.vstack((nodes[:ref_j],nodes[ref_j+1:]))]);

def get_lmatrix2D(nodes,ref_j):
    return np.array([get_lrow_vector2D(node_i,nodes[ref_j]) for node_i in
                     np.vstack((nodes[:ref_j],nodes[ref_j+1:]))]);

# calculate linear least squares x, assuming A is skinny, well-behaved, etc.
def get_xls(A,b_,scale=0.000001):
    AT = A.transpose();
    ATA = np.dot(AT,A);
    At = np.dot(np.linalg.inv(ATA),AT);
    return scale * np.dot(At,b_);

# get xls vector for an entire time series of nodes, ref j
def get_series_out(nodes,full_readings_list,j):
    d_ = get_dist_vector(nodes,j);
    A = get_lmatrix(nodes,j)

    output_vector_list = [];
    time_list = [];

    ## Iterate over sets from the full_reading_list
    for distance_list in full_readings_list:

        # Assumed millimeters
        rn = np.array((float(distance_list[0][3]),
                       float(distance_list[1][3]),
                       float(distance_list[2][3]),
                       float(distance_list[3][3]),
                       float(distance_list[4][3]),
                       float(distance_list[5][3]),
                       float(distance_list[6][3]),
                       float(distance_list[7][3])));
        # time reading from node 1 distance measurement
        time_list.append(distance_list[0][0]);
        #rn /= 328.084;

        # 750 mm offset, then convert to meters
        rn -= 750.0;
        rn /= 1000;
        b_ = get_ldist_vector(rn,d_,j);
        xls = get_xls(A,b_,scale=1);
        #xls = optimize.least_squares(A,b_,bounds=(0,100));
        #xls,r,r,s = np.linalg.lstsq(A,b_,rcond=0;
        output_vector_list.append(xls);

    return output_vector_list, time_list;

def get_series_out2D(nodes,full_readings_list,j):
    d_ = get_dist_vector2D(nodes,j);
    A = get_lmatrix2D(nodes,j)

    output_vector_list = [];
    time_list = [];

    ## Iterate over sets from the full_reading_list
    for distance_list in full_readings_list:

        # Assumed millimeters
        rn = np.array((float(distance_list[0][3]),
                       float(distance_list[1][3]),
                       float(distance_list[2][3]),
                       float(distance_list[3][3]),
                       float(distance_list[4][3]),
                       float(distance_list[5][3]),
                       float(distance_list[6][3]),
                       float(distance_list[7][3])));
        # time reading from node 1 distance measurement
        time_list.append(distance_list[0][0]);
        rn /= 328.084;
        #rn *= 0.3048;
        b_ = get_ldist_vector(rn,d_,j);
        xls = get_xls(A,b_,scale=1);
        #xls = optimize.least_squares(A,b_,bounds=(0,100));
        #xls,r,r,s = np.linalg.lstsq(A,b_,rcond=0;
        output_vector_list.append(xls);

    return output_vector_list,time_list;

# parse the sensor logs for an unmodified array of distances
def get_radii(full_readings_list):
    rn_list = [];
    ## Iterate over sets from the full_reading_list
    for distance_list in full_readings_list:

        # Assumed millimeters
        rn = np.array((float(distance_list[0][3]),
                       float(distance_list[1][3]),
                       float(distance_list[2][3]),
                       float(distance_list[3][3]),
                       float(distance_list[4][3]),
                       float(distance_list[5][3]),
                       float(distance_list[6][3]),
                       float(distance_list[7][3])));
        rn_list.append(rn);
    return np.array(rn_list);
