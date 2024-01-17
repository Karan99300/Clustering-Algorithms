import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == '__main__':
    plt.rcParams['figure.figsize']=[10,10]
    plt.rcParams['figure.dpi']=80
    
    n=1000
    r_list=[2,4,6,8,10]
    sigma_list=[0.1,0.25,0.5]
    
    params=[[(r,n,sigma) for r in r_list] for sigma in sigma_list]
    coordinates_list=[]
    fig,axes=plt.subplots(3,1,figsize=(10,25))
    
    for i,param in enumerate(params):
        coordinates=concentric_circle_generation(param)
        coordinates_list.append(coordinates)
        ax=axes[i]
        for j in range(0,len(coordinates)):
            x,y=coordinates[j]
            sns.scatterplot(x=x,y=y,color='black',ax=ax)
            
    plt.show()
        