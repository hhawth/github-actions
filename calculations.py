from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import numpy as np
r = skewnorm.rvs(-10, loc=2.5, scale=(5/4), size=100000)
r = r - min(r)      #Shift the set so the minimum value is equal to zero.
r = r / max(r)      #Standadize all the vlues between 0 and 1. 
r = r * 5         #Multiply the standardized values by the maximum value.
print(np.mean(r))
#Plot histogram to check skewness
plt.hist(r,30,density=True, color = 'red', alpha=0.1)
plt.savefig("test.pdf")