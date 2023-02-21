import numpy as np
import matplotlib.pyplot as plt

# Generate data
N = 500
mu = [0,0]
sigma = [6,1]
print("sigma :", sigma)
print("sigma-diag :", np.diag(sigma))

theta = 15*np.pi/180 # Angle of rotation for data2
rot1 = np.eye(2) # Rotation for data1
rot2 = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

data1 = np.dot(np.dot(np.random.randn(N,2), np.diag(sigma)) + mu, rot1)
data2 = np.dot(np.dot(np.random.randn(N,2), np.diag(sigma)) + mu, rot2)

d1 = np.dot(rot1, [1,0])
d2 = np.dot(rot2, [1,0])

# Plot the generated data and their directions
plt.subplot(1, 3, 1)
plt.scatter(data1[:,0], data1[:,1])
plt.scatter(data2[:,0], data2[:,1])
plt.plot([0, d1[0]]*int(np.max(data1)), [0, d1[1]]*int(np.max(data1)), linewidth=2)
plt.plot([0, d2[0]]*int(np.max(data2)), [0, d2[1]]*int(np.max(data2)), linewidth=2)
plt.legend(['class 1', 'class 2', 'd_1','d_2'])
plt.grid()
plt.axis('equal')
plt.title('Before CSP filtering')
plt.xlabel('Channel 1')
plt.ylabel('Channel 2')

print(data1)
print(data2)
print("data1: ", data1.shape)
print("data2: ", data2.shape)

# CSP
X1 = data1.T # Positive class data: X1~[C x T]
X2 = data2.T # Negative class data: X2~[C x T]
print(X1.shape)
print(X2.shape)

# Mean center the data
mean1 = np.mean(data1, axis=0)
mean2 = np.mean(data2, axis=0)
data1_centered = data1 - mean1
data2_centered = data2 - mean2

# Calculate covariance matrices
# 공분산 공식
# Cov(X, Y) = E[(X - E(X) * (Y - E(Y))]
cov1 = np.dot(data1_centered.T, data1_centered) / data1_centered.shape[0]
cov2 = np.dot(data2_centered.T, data2_centered) / data2_centered.shape[0]
# cov1 = np.cov(X1)
# cov2 = np.cov(X2)

# check is there NAN in the value
# if	False in np.isnan(data1) or np.isnan(data2):
# 	print("Sorry, NaN is detected in the data. Please try again, we will generate it continuously.")
	# exit()

# eigen vector and eigen value

[w, v] = np.linalg.eigh(
			np.dot(
				np.linalg.inv(
					np.sqrt(cov1)
				),
				np.dot(
					cov2,
					np.linalg.inv(
						np.sqrt(cov1)
					)
				)
			)
		)

X1_CSP = np.dot(v.T, X1)
X2_CSP = np.dot(v.T, X2)

# Plot the results
plt.subplot(1, 3, 2)
plt.scatter(X1_CSP[0,:], X1_CSP[1,:])
plt.scatter(X2_CSP[0,:], X2_CSP[1,:])
plt.legend(['class 1', 'class 2'])
plt.axis('equal')
plt.grid()
plt.title('After CSP filtering')
plt.xlabel('Channel 1')
plt.ylabel('Channel 2')

# plot third one

from scipy import linalg
[w, v] = linalg.eigh(
	cov1,
	cov1 + cov2
)
print(w.shape)
print("V : ", v, v.shape)

X1_CSP = np.dot(v.T, X1)
X2_CSP = np.dot(v.T, X2)

# Plot the results
plt.subplot(1, 3, 3)
plt.scatter(X1_CSP[0,:], X1_CSP[1,:])
plt.scatter(X2_CSP[0,:], X2_CSP[1,:])
plt.legend(['class 1', 'class 2'])
plt.axis('equal')
plt.grid()
plt.title('After CSP filtering')
plt.xlabel('Channel 1')
plt.ylabel('Channel 2')


plt.show()


