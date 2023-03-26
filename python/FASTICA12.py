import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(np.loadtxt("F:\python-project/test_data/test1.txt"))  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
np.savetxt(r'F:\python-project\test_data\test3.txt', S_)

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

data =np.loadtxt("F:\python-project/test_data/test1.txt")
print(data==X)

data1 =np.loadtxt("F:\python-project/test_data/test3.txt")
data2 =np.loadtxt("F:\python-project/test_data/test2.txt")
#print(S_==data2)
# #############################################################################
# Plot results

plt.figure()

models = [X, S, S_, H]
names = [
    "Observations (mixed signal)",
    "True Sources",
    "ICA recovered signals",
    "PCA recovered signals",
]
colors = ["red", "steelblue", "orange"]
#enumerate是内置枚举函数，zip函数是打包models，names，，
# 1代表start=1，意思是列表第一个元素代号为1
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)

    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()