import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
@jit(nopython=True)
def tridiag_solve(a,b,c,V):
    #a,b,c are the three diagonal elements of some tridiagonal matrix T
    #This algorithm solves TM=V for M
    #a,b,c length n,n-1,n-1 resp. vectors. V is nxm matrix
    n = len(b)
    m = V.shape[1]
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros((n,m))
    w[0] = c[0]/b[0]
    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(m):
        d= V[:,i].copy()
        g[0] = d[0]/b[0]
        for j in range(1,n):
            g[j] = (d[j] - a[j-1]*g[j-1])/(b[j] - a[j-1]*w[j-1])
        p[n-1,i]=g[n-1]
        for j in range(n-1,0,-1):
            p[j-1,i]=g[j-1]-w[j-1]*p[j,i]
    return p
@jit(nopython=True)
def i_lbound(I,i,n):
    if i<=I[0]:
        return 0
    elif i>=I[n-1]:
        return n-2
    else:
        i_min = 0
        i_max = n-1
        while i_max-i_min>1:
            i_cent = (i_max+i_min)//2
            if (I[i_cent]>i):
                i_max = i_cent
            else:
                i_min = i_cent
        return i_min
@jit(nopython=True)
def fast_maxfind(I,P,V,n,m,delta_t,c,d):
    W = np.zeros((n,m,2))
    for i in range(n):
        umax  = c*np.sqrt((I[n-1]-I[i])/I[n-1])
        umin  = -c*np.sqrt(I[i]/I[n-1])
        rmax = (I[n-1]-I[i])/delta_t
        rmin = (I[0]-I[i])/delta_t
        tmax = min(umax,rmax)
        tmin = max(umin,rmin)
        Imax = I[i]+delta_t*tmax
        Imin = I[i]+delta_t*tmin
        lbound = i_lbound(I,Imin,n)
        ubound = i_lbound(I,Imax,n)
        for j in range(m):
            y1 = ((Imin-I[lbound])*V[lbound,j]+(I[lbound+1]-Imin)*V[lbound+1,j])/(I[lbound+1]-I[lbound])
            u1=tmin
            vbest = y1-d*delta_t*u1**2-u1*delta_t*P[j]
            ubest = u1
            for k in range(lbound+1,ubound+1):
                y2 = V[k,j]
                u2 = (I[k]-I[i])/delta_t
                uloc = ((y2-y1)/(d*delta_t*(u2-u1))-P[j]/d)/2
                if (uloc>u1) and (uloc<u2):
                    vmax = d*delta_t*uloc**2 +(y1*u2-u1*y2)/(u2-u1)
                    if vmax>vbest:
                        vbest = vmax
                        ubest = uloc
                else:
                    vmax = y1-d*delta_t*u1**2-u1*delta_t*P[j]
                    if vmax>vbest:
                        vbest = vmax
                        ubest = u1
                y1=y2
                u1=u2
            y2 = ((Imax-I[ubound])*V[ubound,j]+(I[ubound+1]-Imax)*V[ubound+1,j])/(I[ubound+1]-I[ubound])
            u2 = tmax
            uloc = ((y2-y1)/(d*delta_t*(u2-u1))-P[j]/d)/2
            if (uloc>u1) and (uloc<u2):
                vmax = d*delta_t*uloc**2 +(y1*u2-u1*y2)/(u2-u1)
                if vmax>vbest:

                    vbest = vmax
                    ubest = uloc
            else:
                if uloc<=u1:
                    vmax = y1-d*delta_t*u1**2-delta_t*u1*P[j]
                    if vmax>vbest:
                        vbest = vmax
                        ubest = u1
                else:
                    vmax = y2-d*delta_t*u2**2-delta_t*u2*P[j]
                    if vmax>vbest:
                        vbest = vmax
                        ubest = u2
                    
            W[i,j,0]=vbest
            W[i,j,1]=ubest
    return W
def tri_const(a,b,c):
    n=len(b)
    v=np.zeros((n,n))
    v[0,0]=b[0]
    v[0,1]=c[0]
    v[n-1,n-2]=a[n-2]
    v[n-1,n-1]=b[n-1]
    for i in range(1,n-1):
        v[i,i-1]=a[i-1]
        v[i,i]=b[i]
        v[i,i+1]=c[i]
    return v
class solver:
    def __init__(self,I_grid,P_grid,phi_grid,c,mu,ss,delta_t,d):
        self.V = phi_grid
        self.controls=np.zeros(phi_grid.shape)
        self.I=I_grid
        self.P=P_grid
        self.mu=mu
        self.ss=ss
        self.best_control=np.zeros(phi_grid.shape)
        self.n = len(self.I)
        self.m = len(self.P)
        self.c=c
        self.d=d
        self.I_max = self.I[self.n-1]
        self.I_min = self.I[0]
        self.delta_t = delta_t
        beta = -mu*self.P[0]/(self.P[1]-self.P[0])
        self.L2=np.zeros(self.m)
        self.L1=np.zeros(self.m-1)
        self.L3=np.zeros(self.m-1)
        self.L2[0]=-beta
        self.L3[0]= beta
        alpha = mu*self.P[self.m-1]/(self.P[self.m-1]-self.P[self.m-2])
        self.L2[self.m-1]=-alpha
        self.L1[self.m-2]=alpha
        for j in range(1,self.m-1):
            d = -mu*self.P[j]/(self.P[j+1]-self.P[j-1])
            a = ss/((self.P[j]-self.P[j-1])*(self.P[j+1]-self.P[j-1]))
            b = ss/((self.P[j+1]-self.P[j])*(self.P[j+1]-self.P[j-1]))
            
            if (a-d)<0:
                alpha = a
                beta = b+d
            elif (b+d)<0:
                alpha = a-d
                beta = b
            else:
                alpha = a-d
                beta = b+d
            self.L1[j-1]=alpha
            self.L2[j]=-alpha-beta
            self.L3[j]=beta
        
    def set_V(self,new_V):
        self.V=new_V
    def umax(self,i):
        return None
    def umin(self,i):
        return None
    def cost(self,j,u):
        return None
    def bound(self,I_test):
        i_min = 0
        i_max = self.n-1
        if I_test<self.I_min:
            return [0,1]
        elif I_test>self.I_max:
            return [i_max-1,i_max]
        else:
            while i_max-i_min>1:
                i_cent = (i_max+i_min)//2
                if (self.I[i_cent]>I_test):
                    i_max = i_cent
                else:
                    i_min = i_cent
            return [i_min,i_max]
    def interpolate(self,i,j,u):
        new_I = self.I[i]+self.delta_t*u
        if (new_I>self.I_max):
            new_I=self.I_max
            u=(self.I_max-self.I[i])/self.delta_t
        elif (new_I<0):
            new_I=0
            u= -self.I[i]/self.delta_t
        ucost=self.cost(j,u)*self.delta_t
        i_min,i_max = self.bound(new_I)
        top = self.I[i_max]
        bot = self.I[i_min]
        topV=self.V[i_max,j]
        botV=self.V[i_min,j]
        out = botV*(new_I-bot)/(top-bot)+topV*(top-new_I)/(top-bot)-ucost
        return out
    def max_find(self,i,j):
        ux = self.umax(i)
        um = self.umin(i)
        vx = self.interpolate(i,j,ux)
        vm = self.interpolate(i,j,um)
        if (vx>vm):
            return [vx,ux]
        else:
            return [vm,um]
    def iterate(self):
        W=fast_maxfind(self.I,self.P,self.V,self.n,self.m,self.delta_t,self.c,self.d)
        self.V=tridiag_solve(-self.L1*self.delta_t,1-self.L2*self.delta_t,-self.L3*self.delta_t,W[:,:,0].T).T
        #self.V=W[:,:,0]
        self.controls=W[:,:,1]
        
class first_solver(solver):
    def umax(self,i):
        return self.c*np.sqrt((self.I_max-self.I[i])/self.I_max)
    def umin(self,i):
        return -self.c*np.sqrt(self.I[i]/self.I_max)
    def cost(self,j,u):
        return self.P[j]*u
    def max_find(self,i,j):
        ux = self.umax(i)
        um = self.umin(i)
        vx = self.interpolate(i,j,ux)
        vm = self.interpolate(i,j,um)
        if (vx>vm):
            v=vx
            u=ux
        else:
            v=vm
            u=um
        i_min = self.bound(self.I[i]+self.delta_t*um)[1]
        i_max = self.bound(self.I[i]+self.delta_t*ux)[0]
        for k in range(i_min,i_max+1):
            test_u = (self.I[k]-self.I[i])/self.delta_t
            ucost = self.cost(j,test_u)*self.delta_t
            if self.V[k,j]-ucost>v:
                v=self.V[k,j]-ucost
                u=test_u
        return [v,u]

n=400
m=400
I_max=1
P_max=5
c=0.5
d=1
delta_t=0.005
mu = 1
ss = 1
I = np.linspace(0,I_max,n)
P = np.linspace(-P_max,P_max,m)
V = np.zeros((n,m))
tester = first_solver(I,P,V,c,mu,ss,delta_t,d)
tester2 = first_solver(I,P,V,c,mu,ss,delta_t,d)
tester2.iterate()
for i in range(4000):
    print(i)
    new_w=tester.iterate()
L=tri_const(tester.L1*delta_t,1+tester.L2*delta_t,tester.L3*delta_t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
grid = np.meshgrid(I,P,indexing='ij')
ax.plot_surface(grid[0],grid[1],tester.controls-tester2.controls)
ax.set_xlabel('I')
ax.set_ylabel('P')
ax.set_zlabel('u')
plt.show()
"""
xaxis = np.linspace(0,1,100)
yaxis = np.zeros(100)
for i in range(100):
    val = tester.interpolate(0,0,xaxis[i])
    yaxis[i] = val
plt.plot(xaxis,yaxis)
plt.show()"""
