import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
import itertools

from visualization_fct import *
from bokeh.resources import CDN
from bokeh.embed import file_html


import matplotlib.pyplot as plt  # , mpld3

data = pd.read_csv("asm_data_for_ml.txt", sep='\t')
del data['MJD']
del data['error']
del data['errorA']
del data['errorB']
del data['errorC']
data['rateCA'] = data.rateC / data.rateA
data_thr = mask(data, 'orbit')  # rm too large values except for 'orbit'


np.random.seed(0)

X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
          data_thr.rateC, data_thr.rateCA]

Html_file = open("human_annotations_files/human.html", "w")

Html_file.write(' \n <video width="480" height="320" controls="controls"> \n <source src="streaming_asm.mp4" type="video/mp4"> \n </video>')

scaler = StandardScaler()
X = scaler.fit_transform(X)

x = data_thr.rateCA
y = data_thr.rate

upper = np.array(y > np.maximum(20, 350 * (x - 0.28))).astype(bool)
lower = np.array(y < np.maximum(20, 55 * (x - 0.28))).astype(bool)

preds = np.ones(X.shape[0]).astype(int)
preds[lower] = 0
preds[upper] = 2

data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+2]


# interactive transition probability :

p = interactive_transition_probability(data_thr,
                                       color_key=color_key,
                                       x_name='rateCA', y_name='rate',
                                       covs=None, means=None)
html = file_html(p, CDN, "interactive plot with human annotations")
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# # animation:  (work only in a jupyter notebook)
# fig, ax = plt.subplots()
# ax.set_xlim(0, 5)
# ax.set_ylim(0, 120)

# XX = np.c_[data_thr.rateCA, data_thr.rate]

# x = XX[:, 0]
# y = XX[:, 1]

# upper = np.array(y > np.maximum(20, 350 * (x - 0.28))).astype(bool)
# lower = np.array(y < np.maximum(20, 55 * (x - 0.28))).astype(bool)

# yy = np.ones(X.shape[0]).astype(int)
# yy[lower] = 0
# yy[upper] = 2

# line = ax.scatter(XX[:, 0], XX[:, 1])

# # plot the manual decisions (lines):
# t = np.linspace(0, 2, 1000)
# y1 = np.max([20 * np.ones(t.shape[0]), 55 * (t - 0.28)], axis=0)
# tt = t[:300]
# y2 = np.max([20 * np.ones(tt.shape[0]), 350 * (tt - 0.28)], axis=0)
# ax.plot(t, y1, color='black')
# ax.plot(tt, y2, color='black')


# def animate(i):
#     line.set_offsets(XX[10*i:10*(i+1), :])
#     #line.set_xdata(XX[10*i:10*(i+1), 0])  # update the data
#     #line.set_ydata(XX[10*i:10*(i+1), 1])   # update the data
#     #line.set_color(color_key[preds[1*i]])
#     color = [color_key[yy[10*i + k]] for k in range(10)]
#     line.set_color(color)
#     return line,

# ani = animation.FuncAnimation(fig, animate, np.arange(1000, 1010), #init_func=init,
#                               interval=1, blit=True)
# ani.save('streaming_asm_test.mp4', writer = 'mencoder', fps=30, extra_args=['-vcodec', 'libx264'])


Html_file.close()
