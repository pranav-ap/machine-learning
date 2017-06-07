import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)


######### OLD ##########
mem_use = pd.DataFrame(np.random.randint( 5, 100, 
                                         size=(80, 3)), 
                                         columns=['mem_use_m','mem_use_v','mem_use_s'])

peak_mem_use = pd.DataFrame(np.random.randint( 48, 100, 
                                         size=(80, 3)), 
                                         columns=['peak_mem_use_m','peak_mem_use_v','peak_mem_use_s'])

threads = pd.DataFrame(np.random.randint( 1, 8, 
                                         size=(80, 3)), 
                                         columns=['threads_m','threads_v','threads_s'])

handles = pd.DataFrame(np.random.randint( 1, 14, 
                                         size=(80, 3)), 
                                         columns=['handles_m','handles_v','handles_s'])

packets = pd.DataFrame(np.random.randint( 11, 30, 
                                         size=(80, 3)), 
                                         columns=['packets_m','packets_v','packets_s'])

byte = pd.DataFrame(np.random.randint( 11, 44, 
                                         size=(80, 3)), 
                                         columns=['byte_m','byte_v','byte_s'])

flows = pd.DataFrame(np.random.randint( 1, 6, 
                                         size=(80, 3)), 
                                         columns=['flows_m','flows_v','flows_s'])

dataset = pd.concat([mem_use['mem_use_m'], mem_use['mem_use_v'], mem_use['mem_use_s'], 
                    peak_mem_use['peak_mem_use_m'], peak_mem_use['peak_mem_use_v'], peak_mem_use['peak_mem_use_s'],
                    threads['threads_m'], threads['threads_v'], threads['threads_s'],
                    handles['handles_m'], handles['handles_v'], handles['handles_s'],
                    packets['packets_m'], packets['packets_v'], packets['packets_s'],
                    byte['byte_m'], byte['byte_v'], byte['byte_s'],
                    flows['flows_m'], flows['flows_v'], flows['flows_s']], axis = 1)

#######################
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, :21].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, :21].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


