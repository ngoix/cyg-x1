from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import itertools

from visualization_fct import *
# from bokeh.plotting import output_file, show, save
# ACHTUNG!!! log transform of rateCA or without orbit may be activated
from bokeh.resources import CDN
from bokeh.embed import file_html


from pomegranate import HiddenMarkovModel
from pomegranate import MultivariateGaussianDistribution
from pomegranate import State, Distribution

import matplotlib.pyplot as plt  # , mpld3


data = pd.read_csv("asm_data_for_ml.txt", sep='\t')
del data['MJD']
del data['error']
del data['errorA']
del data['errorB']
del data['errorC']
data['rateCA'] = data.rateC / data.rateA
data_thr = mask(data, 'orbit')  # rm too large values except for 'orbit'


np.random.seed(1)
kmeans = False

X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
          data_thr.rateC, data_thr.rateCA]
if kmeans:
    Html_file = open("hmm_pomegranate_files/hmm5kmeans_pomegranate.html", "w")
else:
    Html_file = open("hmm_pomegranate_files/hmm5prior_pomegranate.html", "w")

scaler = StandardScaler()
X = scaler.fit_transform(X)


# Hmm with 3 components:
# looks better with data.rateCA in X
# looks like 3 is better than 4 components

# TODO kmeans + 4-5 components!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if kmeans:
    km = KMeans(n_clusters=5)
    y = km.fit_predict(X)
    X_1 = X[y == 0]
    X_11 = X[y == 1]

    X_2 = X[y == 2]
    X_22 = X[y == 3]

    X_3 = X[y == 4]

    # print 'km.cluster_centers_', km.cluster_centers_
    # plt.scatter(X[:, 5], X[:, 1], c=y)
    # plt.show()

else:
    X_1 = X[2000:3000]
    X_11 = X[3000:4000]

    X_2 = X[400:500]
    X_22 = X[500:600]

    X_3 = X[7000:7300]

a = MultivariateGaussianDistribution.from_samples(X_1)
aa = MultivariateGaussianDistribution.from_samples(X_11)

b = MultivariateGaussianDistribution.from_samples(X_2)
bb = MultivariateGaussianDistribution.from_samples(X_22)

c = MultivariateGaussianDistribution.from_samples(X_3)

s1 = State(a, name="M1")
s11 = State(aa, name="M11")

s2 = State(b, name="M2")
s22 = State(bb, name="M22")

s3 = State(c, name="M3")

hmm = HiddenMarkovModel()
hmm.add_states(s1, s11, s2, s22, s3)

hmm.add_transition(hmm.start, s1, 0.2)
hmm.add_transition(hmm.start, s11, 0.2)

hmm.add_transition(hmm.start, s2, 0.2)
hmm.add_transition(hmm.start, s22, 0.2)

hmm.add_transition(hmm.start, s3, 0.2)


hmm.add_transition(s1, s1, 0.92)
hmm.add_transition(s1, s11, 0.02)
hmm.add_transition(s1, s2, 0.02)
hmm.add_transition(s1, s22, 0.02)
hmm.add_transition(s1, s3, 0.02)

hmm.add_transition(s11, s1, 0.02)
hmm.add_transition(s11, s11, 0.92)
hmm.add_transition(s11, s2, 0.02)
hmm.add_transition(s11, s22, 0.02)
hmm.add_transition(s11, s3, 0.02)


hmm.add_transition(s2, s1, 0.02)
hmm.add_transition(s2, s11, 0.02)
hmm.add_transition(s2, s2, 0.92)
hmm.add_transition(s2, s22, 0.02)
hmm.add_transition(s2, s3, 0.02)

hmm.add_transition(s22, s1, 0.02)
hmm.add_transition(s22, s11, 0.02)
hmm.add_transition(s22, s2, 0.02)
hmm.add_transition(s22, s22, 0.92)
hmm.add_transition(s22, s3, 0.02)


hmm.add_transition(s3, s1, 0.02)
hmm.add_transition(s3, s11, 0.02)
hmm.add_transition(s3, s2, 0.02)
hmm.add_transition(s3, s22, 0.02)
hmm.add_transition(s3, s3, 0.92)


hmm.bake()
hmm.fit(X)  # , weights=w) hmm does not support weights in pomegranate
preds = hmm.predict(X)
probs = hmm.predict_proba(X)

data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+2]

covs = np.array([np.array(hmm.states[m].distribution.parameters[1])
                 for m in range(5)])
means = np.array([np.array(hmm.states[m].distribution.parameters[0])
                  for m in range(5)])

# transform cov for non-standardizeed data:
covs = np.array([np.dot(np.diag(np.sqrt(scaler.var_)),
                        np.dot(covs[j], np.diag(np.sqrt(scaler.var_))))
                 for j in range(covs.shape[0])])
means = np.array([scaler.inverse_transform(means[j].reshape(1, -1)).T
                  for j in range(means.shape[0])])


# # uncomment  to show interactive probas:
# p = plot_probas(data_thr, probs)
# plt.show()

# p = interactive_img_ds(data_thr, 'rateCA', 'rate')
# # waiting for InteractiveImage -> html


# # pair plots with predicted classes and ellipses:
# p = scatter_matrix(data_thr, spread=False, covs=covs, means=means,
#                    color_key=color_key)
# html = file_html(p, CDN, "pomegranate weighted hmm with 3 components")
# Html_file.write(html)
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


x = 5
y = 1
covs_xy = [covs[j][[x, y]][:, [x, y]] for j in range(len(covs))]
means_xy = [means[j][[x, y]] for j in range(len(covs))]

# # single plot rateCA vs rate with predicted classes and ellipses:

# # # # #uncommment if log
# # for j in range(len(covs)):
# #     covs_xy[j][0] = np.exp(covs_xy[j][0]) - 1
# #     means_xy[j][0] = np.exp(means_xy[j][0]) - 1

# single_plot = bokeh_datashader_plot(data_thr, covs=covs_xy, means=means_xy,
#                                     x_name='rateCA',
#                                     y_name='rate',
#                                     plot_width=900, plot_height=300,
#                                     pixel_width=3000, pixel_height=1000,
#                                     spread=True, color_key=color_key)
# html = file_html(single_plot, CDN, "pomegranate hmm with 3 components")
# Html_file.write(html)
# # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')

# # histogram rateCA:
# p = Figure()
# rateCA = np.array(data_thr['rateCA'])
# hist, edges = np.histogram(rateCA, bins=100)
# p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
# html = file_html(p, CDN, "hist rateCA")
# Html_file.write(html)
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# # probas with datashader:
# fig = plot_probs_datashader(probs)
# html = file_html(fig, CDN, "probas with datashader")
# Html_file.write(html)
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# linked brushing probas:
data_probs = data_thr.copy()
prob_names = []
for j in range(probs.shape[1]):
    prob_names += ['probs'+str(j)]
    data_probs['probs'+str(j)] = pd.Series(probs[:, j])

linkbru = plot_probs_bokeh_linked_brushing(data_probs, prob_names=prob_names,
                                           color_key=color_key,
                                           x_name='rateCA', y_name='rate',
                                           covs=covs_xy, means=means_xy,
                                           title='Hidden Markov Model with 5 states',
                                           title2='State belonging and state probability')
html = file_html(linkbru, CDN, "pomegranate hmm with 3 components")
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# transition matrix:
T = np.exp(hmm.dense_transition_matrix())[:5, :5]
plt.imshow(T, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
for i, j in itertools.product(range(T.shape[0]), range(T.shape[1])):
        plt.text(j, i, int(T[i, j]*100),
                 horizontalalignment="center",
                 color=(color_key[i] if i == j
                        else "white" if T[i, j] > 0.5
                        else "black"))
plt.title('Transition probability matrix of the HMM. \n Log likelihood value: %0.3f' % hmm.log_probability(X))
plt.savefig('hmm_pomegranate_files/hmm5_pomegranate_transition.png')
data_uri = open(
    'hmm_pomegranate_files/hmm5_pomegranate_transition.png',
    'rb').read().encode('base64').replace('\n', '')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
Html_file.write(img_tag)

# # biclusetering on T:
# biclust = SpectralCoclustering(n_clusters=4)
# biclust.fit(T)
# TT = T[np.argsort(biclust.row_labels_)]
# TT = TT[:, np.argsort(biclust.column_labels_)]
# plt.imshow(T, interpolation='nearest', cmap=plt.cm.Blues)
# plt.colorbar()
# for i, j in itertools.product(range(T.shape[0]), range(T.shape[1])):
#         plt.text(j, i, int(T[i, j]*100),
#                  horizontalalignment="center",
#                  color=(color_key[i] if i == j
#                         else "white" if T[i, j] > 0.5
#                         else "black"))
# plt.title('log_likelihood:%0.3f' % hmm.log_probability(X))
# plt.savefig('hmm_pomegranate_files/hmm5_pomegranate_transition.png')
# data_uri = open(
#     'hmm_pomegranate_files/hmm5_pomegranate_transition.png',
#     'rb').read().encode('base64').replace('\n', '')
# img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
# Html_file.write(img_tag)


# interactive transition probability :

p = interactive_transition_probability(data_thr,
                                       color_key=color_key,
                                       x_name='rateCA', y_name='rate',
                                       covs=covs_xy, means=means_xy)
html = file_html(p, CDN, "pomegranate hmm with 3 components")
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')



# # ######### single plot animated:
# fig, ax = plt.subplots()

# XX = np.c_[data_thr.rateCA, data_thr.rate]
# XX = XX[np.array(data_thr.rateCA < 5) * np.array(data_thr.rate < 80)]

# line, = ax.plot(XX[:, 0], XX[:, 1], '.')

# for n_comp in range(len(covs_xy)):
#     cov = covs_xy[n_comp]
#     mean = means_xy[n_comp]
#     v, w = np.linalg.eigh(cov)
#     e0 = w[0] / np.linalg.norm(w[0])
#     e1 = w[1] / np.linalg.norm(w[1])
#     t = np.linspace(0, 2 * np.pi, 10000)
#     # 4.605 corresponds to 90% quantile:
#     a = (mean[0]
#          + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[0]
#          + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[0])
#     b = (mean[1]
#          + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[1]
#          + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[1])

#     ax.plot(a, b, color=color_key[n_comp])

# def animate(i):
#     line.set_xdata(XX[10*i:10*(i+1), 0])  # update the data
#     line.set_ydata(XX[10*i:10*(i+1), 1])   # update the data
#     line.set_color(color_key[preds[10*i]])
#     return line,

# ani = animation.FuncAnimation(fig, animate, np.arange(1000, 9300), #init_func=init,
#                               interval=1, blit=True)
# plt.savefig('hmm_pomegranate_files/single_plot_animated.png')
# data_uri = open(
#     'hmm_pomegranate_files/single_plot_animated.png',
#     'rb').read().encode('base64').replace('\n', '')
# img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
# Html_file.write(img_tag)


Html_file.close()
