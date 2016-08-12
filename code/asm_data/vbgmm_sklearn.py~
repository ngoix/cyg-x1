from sklearn.preprocessing import StandardScaler
from visualization_fct import *
# from bokeh.plotting import output_file, show, save

from bokeh.resources import CDN
from bokeh.embed import file_html

from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt  # , mpld3


data = pd.read_csv("asm_data_for_ml.txt", sep='\t')
del data['MJD']
# del data['error']
# del data['errorA']
# del data['errorB']
# del data['errorC']
data['rateCA'] = data.rateC / data.rateA
data_thr = mask(data, 'orbit')  # rm too large values except for 'orbit'


w = np.concatenate([np.array(data_thr[name]).reshape(-1, 1)
                    for name in ['error', 'errorA', 'errorB', 'errorC']],
                   axis=1)
# XXX add the error for the ratio?
w = np.concatenate([w,
                    (data_thr['errorC'] / data_thr['rateA']).reshape(-1, 1)],
                   axis=1)

del data_thr['error']
del data_thr['errorA']
del data_thr['errorB']
del data_thr['errorC']


np.random.seed(1)

X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
          data_thr.rateC, data_thr.rateCA]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 1 corresponds to data_thr.rate and 4=5-1 to data_thr.rateC
w = w / np.sqrt(scaler.var_[1:])
w = np.exp(-np.exp(3 * w.mean(axis=1)))


Html_file = open("gmm_sklearn_files/gmm3_sklearn.html", "w")

gmm = GaussianMixture(n_components=3, n_init=5)
gmm.fit(X)  # , weights=w) not implemented in sklearn yet
preds = gmm.predict(X)
probs = gmm.predict_proba(X)

data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+1]

covs = gmm.covariances_
means = gmm.means_

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
# html = file_html(p, CDN, "sklearn gmm with 3 components")
# Html_file.write(html)
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
# Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# single plot rateCA vs rate with predicted classes and ellipses:
x = 5
y = 1
covs_xy = [covs[j][[x, y]][:, [x, y]] for j in range(len(covs))]
means_xy = [means[j][[x, y]] for j in range(len(covs))]

single_plot = bokeh_datashader_plot(data_thr, covs=covs_xy, means=means_xy,
                                    x_name='rateCA',
                                    y_name='rate',
                                    plot_width=900, plot_height=300,
                                    pixel_width=3000, pixel_height=1000,
                                    spread=True, color_key=color_key)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
html = file_html(single_plot, CDN, "sklearn gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


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
for j in range(probs.shape[1]):
    data_probs['probs'+str(j)] = pd.Series(probs[:, j])

linkbru = plot_probs_bokeh_linked_brushing(data_probs,
                                           x_name='rateCA', y_name='rate',
                                           covs=covs_xy, means=means_xy)
html = file_html(linkbru, CDN, "sklearn gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# ##################
# fig = scatter_matrix_seaborn(data_thr)
# plt.title('seaborn_scatterplot')
# fig.fig.savefig(
#     'gmm3_sklearn_files/gmm3_sklearn_seaborn_scatterplot.png')
# data_uri = open(
#     'gmm3_sklearn_files/gmm3_sklearn_seaborn_scatterplot.png',
#     'rb').read().encode('base64').replace('\n', '')
# img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
# Html_file.write(img_tag)
Html_file.close()
