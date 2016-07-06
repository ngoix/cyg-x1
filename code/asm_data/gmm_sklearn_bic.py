from visualization_fct import *
from sklearn.mixture import GaussianMixture
# from bokeh.plotting import output_file, show, save

from bokeh.resources import CDN
from bokeh.embed import file_html

import matplotlib.pyplot as plt  # , mpld3


without_CA = True


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
    Html_file = open("gmm_sklearn_bic/gmm3_sklearn_bic_without_rateCA.html",
                     "w")
else:
    X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
              data_thr.rateC, data_thr.rateCA]
    Html_file = open("gmm_sklearn_bic/gmm3_sklearn_bic.html", "w")

# gmm model selection with bic:
lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=cv_type, n_init=5)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
print best_gmm.covariance_type, best_gmm.n_components

preds = best_gmm.predict(X)
probs = best_gmm.predict_proba(X)

for name, col in zip(cv_types, np.array(bic).reshape(-1, len(cv_types)).T):
    plt.plot(n_components_range, col, label=name)
plt.legend()

if without_CA:
    plt.savefig('gmm_sklearn_bic/bic_withoutCA.pdf')
else:
    plt.savefig('gmm_sklearn_bic/bic.pdf')


data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "yellow", "blue", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+1]

covs = best_gmm.covariances_
means = best_gmm.means_

# uncomment to show probas plot
# p = plot_probas(data_thr, probs)
# # p.savefig('gmm3_sklearn_bic_proba_matplotlib.svg')
# plt.show()

# p = interactive_img_ds(data_thr, 'rateCA', 'rate')
# # waiting for InteractiveImage -> html

p = scatter_matrix(data_thr, covs=covs, means=means, color_key=color_key)
html = file_html(p, CDN, "sklearn_bic gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')

fig = plot_probs_datashader(probs)
html = file_html(fig, CDN, "probas with datashader")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


fig = scatter_matrix_seaborn(data_thr)
plt.title('seaborn_scatterplot')
fig.fig.savefig('gmm_sklearn_bic/gmm3_sklearn_bic_seaborn_scatterplot.png')
data_uri = open('gmm_sklearn_bic/gmm3_sklearn_bic_seaborn_scatterplot.png',
                'rb').read().encode('base64').replace('\n', '')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
Html_file.write(img_tag)
# html = mpld3.fig_to_html(fig.fig)
# Html_file.write(html)
Html_file.close()
