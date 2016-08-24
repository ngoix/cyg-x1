from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import itertools

from visualization_fct import *
# from bokeh.plotting import output_file, show, save
# ACHTUNG!!! log transform of rateCA or without orbit may be activated
from bokeh.resources import CDN
from bokeh.embed import file_html


from pomegranate import HiddenMarkovModel
from pomegranate import MultivariateGaussianDistribution
from pomegranate import State

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

Html_file = open("hmm_pomegranate_files/hmm3_dbscan.html", "w")

scaler = StandardScaler()
X = scaler.fit_transform(X)


# Hmm with 3 components:
# looks better with data.rateCA in X
# looks like 3 is better than 4 components

# TODO kmeans + 4-5 components!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

km = DBSCAN()
y = km.fit_predict(X)
print "number of components (dbscan):", len(set(y))
X_1 = X[y == 0]
X_2 = X[y == 1]
X_3 = X[y == 2]

a = MultivariateGaussianDistribution.from_samples(X_1)
b = MultivariateGaussianDistribution.from_samples(X_2)
c = MultivariateGaussianDistribution.from_samples(X_3)
s1 = State(a, name="M1")
s2 = State(b, name="M2")
s3 = State(c, name="M3")

hmm = HiddenMarkovModel()
hmm.add_states(s1, s2, s3)
hmm.add_transition(hmm.start, s1, 0.34)
hmm.add_transition(hmm.start, s3, 0.33)
hmm.add_transition(hmm.start, s2, 0.33)

hmm.add_transition(s1, s1, 0.9)
hmm.add_transition(s1, s2, 0.05)
hmm.add_transition(s1, s3, 0.05)

hmm.add_transition(s2, s1, 0.05)
hmm.add_transition(s2, s3, 0.05)
hmm.add_transition(s2, s2, 0.9)

hmm.add_transition(s3, s3, 0.9)
hmm.add_transition(s3, s2, 0.05)
hmm.add_transition(s3, s1, 0.05)
hmm.bake()
hmm.fit(X)  # , weights=w) hmm does not support weights in pomegranate
preds = hmm.predict(X)
probs = hmm.predict_proba(X)

data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+2]

covs = np.array([np.array(hmm.states[m].distribution.parameters[1])
                 for m in range(3)])
means = np.array([np.array(hmm.states[m].distribution.parameters[0])
                  for m in range(3)])

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


# single plot rateCA vs rate with predicted classes and ellipses:
x = 5
y = 1
covs_xy = [covs[j][[x, y]][:, [x, y]] for j in range(len(covs))]
means_xy = [means[j][[x, y]] for j in range(len(covs))]

# # # #uncommment if log
# for j in range(len(covs)):
#     covs_xy[j][0] = np.exp(covs_xy[j][0]) - 1
#     means_xy[j][0] = np.exp(means_xy[j][0]) - 1

single_plot = bokeh_datashader_plot(data_thr, covs=covs_xy, means=means_xy,
                                    x_name='rateCA',
                                    y_name='rate',
                                    plot_width=900, plot_height=300,
                                    pixel_width=3000, pixel_height=1000,
                                    spread=True, color_key=color_key)
html = file_html(single_plot, CDN, "pomegranate hmm with 3 components")
Html_file.write(html)
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# transition matrix
T = np.exp(hmm.dense_transition_matrix())[:3, :3]
plt.imshow(T, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
for i, j in itertools.product(range(T.shape[0]), range(T.shape[1])):
        plt.text(j, i, int(T[i, j]*100),
                 horizontalalignment="center",
                 color="white" if T[i, j] > 0.5 else "black")
plt.title('log_likelihood:%0.3f' % hmm.log_probability(X))
plt.savefig('hmm_pomegranate_files/hmm_dbscan_transition.png')
data_uri = open(
    'hmm_pomegranate_files/hmm_dbscan_transition.png',
    'rb').read().encode('base64').replace('\n', '')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
Html_file.write(img_tag)




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
                                           covs=covs_xy, means=means_xy)
html = file_html(linkbru, CDN, "pomegranate hmm with 3 components")
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# interactive transition probability :

p = interactive_transition_probability(data_thr,
                                       color_key=color_key,
                                       x_name='rateCA', y_name='rate',
                                       covs=covs_xy, means=means_xy)
html = file_html(p, CDN, "pomegranate hmm with 3 components")
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')

Html_file.close()
