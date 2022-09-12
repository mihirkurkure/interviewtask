import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import numpy.linalg as npl
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_otsu

np.random.seed(1)  #to get the same data from random
y = np.random.rand(79,95,68,100)  #4-D arrays
#y = y[:,:,:,0]
Y = y.reshape(100,510340)  #reshaping data into 2-D
x = np.random.rand(100,1)  #subject variable, V
Y1 = y.reshape(510340,100)
n = y.shape[-1]
X = np.ones((n,1))
#print(X)

inv_x = npl.pinv(x)   #pseudo-inverse the subject variable
#print(piX.shape)

beta = inv_x @ Y   #get the beta values by dot product
print('Beta values:', beta)

#testing the difference of the slope
c = np.array([1])
ceil_t = c.dot(beta) #numerator

error = n - npl.matrix_rank(X)
#print('error', error) 
fit = X.dot(beta)
e = Y - fit
sigma = np.sum(e**2, axis = 0)/error
#print("sigma:", sigma)
c = c[:, None]
#print('cshape', c)
floor_t = c.T.dot(npl.pinv(X.T.dot(X))).dot(c)
#print('floor', floor_t)

t = ceil_t/np.sqrt(sigma*floor_t)
#print(t.shape)

#as the above calculation gives a 1D vector we need to put it back into the 3d space of voxels
t_3d = t.reshape(y.shape[:3])
#print(t_3d.shape)

#using otsu's method to calculate p-values
mean = y.mean(axis=-1)
thr = threshold_otsu(mean)
mask = mean > thr
y_mask = y[mask].T

#similar to above formula
beta_2 = npl.pinv(X).dot(y_mask)
fit_2 = X.dot(beta_2)
e_2 = y_mask - fit_2
sigma_2 = np.sum(e_2**2, axis = 0)/error
c = np.array([1])
c_f = c.dot(npl.pinv(X.T.dot(X))).dot(c)
t = c.T.dot(beta_2)/np.sqrt(sigma_2 * c_f)

t_3d = np.zeros(y.shape[:3])
t_3d[mask] = t

t_dist = sp.t(error)
p = 1 - t_dist.cdf(t)
print('P-values:', p)

#to find correlation factor
Y_mean = np.mean(Y)
Y_std = np.std(Y)
Y_z_scores = (Y - Y_mean) / Y_std


x_mean = np.mean(inv_x)
x_std = np.std(inv_x)
x_z_scores = (x - x_mean) / x_std


multiplied = Y_z_scores * x_z_scores

corr = np.mean(multiplied)
print('corr =', corr)




#the following code is scikit linear regression
'''
model = LinearRegression()
model.fit(x,Y)

print("intercept", model.intercept_)

print("slope", model.coef_)

beta = (model.intercept_ ) + (model.coef_.reshape(-1))
print('Beta values:', beta)


r2 =  model.score(x,Y)
print("R2", float(r2))
'''





#the script below was just me playing around with single voxel and seeing how it works and also an attempt to draw parametric beta image

'''
#drawing the beta image
a,b = beta.nonzero()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a, b, zdir='z', c= 'red')
plt.savefig("demo.png")
'''

'''
est = sm.OLS(Y1,inv_x)
est2 = est.fit()
print(est2.summary())
'''




''''
#just to check the size
df = pd.DataFrame(beta)
print(df)
'''
'''

#trial - working with a single voxel
plt.plot(x)
plt.title('x')
plt.show()


voxel = y[42,32,19,:]


plt.plot(voxel)
plt.title('voxel')
plt.show()

plt.scatter(x,voxel)
plt.show()

def z_scores(arr):
    return (arr - np.mean(arr))/np.std(arr)

r = np.mean(z_scores(x) * z_scores(voxel))
print(r)

best_slope = r * np.std(voxel) / np.std(x)
print('Best slope:', best_slope)
best_intercept = np.mean(voxel) - best_slope * np.mean(x)
print('Best intercept:', best_intercept)

plt.scatter(x, voxel)
x_vals = np.array([np.min(x), np.max(x)])
plt.plot(x_vals, best_intercept + best_slope * x_vals, 'r:')
plt.show()


#print(voxel)
#plt.plot(voxel)
#plt.show()

#df = pd.DataFrame(y)
#print(df)

#print(x)
'''
