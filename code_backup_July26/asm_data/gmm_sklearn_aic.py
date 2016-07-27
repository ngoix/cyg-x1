from visualization_fct import *
from sklearn.mixture import GaussianMixture
# from bokeh.plotting import output_file, show, save

from bokeh.resources import CDN
from bokeh.embed import file_html

import matplotlib.pyplot as plt  # , mpld3

without_CA = False

data = pd.read_csv("asm_data_for_ml.txt", sep='\t')
del data['MJD']
del data['error']
del data['errorA']
del data['errorB']
del data['errorC']
data['rateCA'] = data.rateC / data.rateA
data_thr = mask(data, 'orbit')  # rm too large values except for 'orbit'


np.random.seed(0)

if without_CA:
    X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
              data_thr.rateC]
    Html_file = open("gmm_sklearn_aic/gmm3_sklearn_aic_without_rateCA.html",
                     "w")
else:
    X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
              data_thr.rateC, data_thr.rateCA]
    Html_file = open("gmm_sklearn_aic/gmm3_sklearn_aic.html", "w")

# gmm model selection with aic:
lowest_aic = np.infty
aic = []
n_components_range = range(1, 10)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=cv_type, n_init=5)
        gmm.fit(X)
        aic.append(gmm.aic(X))
        if aic[-1] < lowest_aic:
            lowest_aic = aic[-1]
            best_gmm = gmm
print best_gmm.covariance_type, best_gmm.n_components

preds = best_gmm.predict(X)
probs = best_gmm.predict_proba(X)

for name, col in zip(cv_types, np.array(aic).reshape(-1, len(cv_types)).T):
    plt.plot(n_components_range, col, label=name)
plt.legend()
if without_CA:
    plt.savefig('gmm_sklearn_aic/aic_withoutCA.pdf')
else:
    plt.savefig('gmm_sklearn_aic/aic.pdf')


data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+1]

covs = best_gmm.covariances_
means = best_gmm.means_

# # uncomment  to show interactive probas:
# p = plot_probas(data_thr, probs)
# plt.show()

# p = interactive_img_ds(data_thr, 'rateCA', 'rate')
# # waiting for InteractiveImage -> html


# pair plots with predicted classes and ellipses:
p = scatter_matrix(data_thr, spread=False, covs=covs, means=means,
                   color_key=color_key)

html = file_html(p, CDN, "sklearn_aic gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# single plot rateCA vs rate with predicted classes and ellipses:
if without_CA:
    covs_xy = None
    means_xy = None
else:
    x = 5
    y = 1
    covs_xy = [covs[j][[x, y]][:, [x, y]] for j in range(len(covs))]
    means_xy = [means[j][[x, y]] for j in range(len(covs))]

single_plot = bokeh_datashader_plot(data_thr, covs=covs_xy, means=means_xy,
                                    x_name='rateCA',
                                    y_name='rate',
                                    plot_width=900, plot_height=300,
                                    pixel_width=3000, pixel_height=1000,
                                    spread=False, color_key=color_key)
html = file_html(single_plot, CDN, "sklearn_aic gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# histogram rateCA:
p = Figure()
rateCA = np.array(data_thr['rateCA'])
hist, edges = np.histogram(rateCA, bins=100)
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
html = file_html(p, CDN, "hist rateCA")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# probas with datashader:
fig = plot_probs_datashader(probs)
html = file_html(fig, CDN, "probas with datashader")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# linked brushing probas:
data_probs = data_thr.copy()
for j in range(probs.shape[1]):
    data_probs['probs'+str(j)] = pd.Series(probs[:, j])

linkbru = plot_probs_bokeh_linked_brushing(data_probs,
                                           x_name='rateCA', y_name='rate',
                                           covs=covs_xy, means=means_xy)
html = file_html(linkbru, CDN, "sklearn_aic gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


##################
fig = scatter_matrix_seaborn(data_thr)
plt.title('seaborn_scatterplot')
fig.fig.savefig('gmm_sklearn_aic/gmm3_sklearn_aic_seaborn_scatterplot.png')
data_uri = open('gmm_sklearn_aic/gmm3_sklearn_aic_seaborn_scatterplot.png',
                'rb').read().encode('base64').replace('\n', '')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
Html_file.write(img_tag)
Html_file.close()
