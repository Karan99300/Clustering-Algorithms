import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import chain
from clustering import *

def circle_generation(r,n,sigma):
    angle=np.random.uniform(low=0,high=2*np.pi,size=n)
    
    x_ep=np.random.normal(loc=0.0,scale=sigma,size=n)
    y_ep=np.random.normal(loc=0.0,scale=sigma,size=n)
    
    x=r*np.cos(angle)+x_ep
    y=r*np.sin(angle)+y_ep
    
    return x,y

def concentric_circle_generation(params):
    coordinates=[
        circle_generation(param[0],param[1],param[2])
        for param in params
    ]
    return coordinates

def data_frame_from_coordinates(coordinates): 
    xs = chain(*[c[0] for c in coordinates])
    ys = chain(*[c[1] for c in coordinates])

    return pd.DataFrame(data={'x': xs, 'y': ys})
        
if __name__ == '__main__':
    n=100
    r_list=[2,4,6]
    sigma_list=[0.1,0.25,0.5]

    params=[[(r,n,sigma) for r in r_list] for sigma in sigma_list]
    coordinates_list=[]
    for i,param in enumerate(params):
        coordinates=concentric_circle_generation(param)
        coordinates_list.append(coordinates)
        for j in range(0,len(coordinates)):
            x,y=coordinates[j]

    coordinates = coordinates_list[0]
    data = data_frame_from_coordinates(coordinates)
    print(data.shape)
    points=data[['x','y']].values
    DTClustering(points,beta=1.5)