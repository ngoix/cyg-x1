from visualization_fct import *
from pomegranate import GeneralMixtureModel
from pomegranate import MultivariateGaussianDistribution

# from bokeh.plotting import output_file, show, save
# from bokeh.mpl import to_bokeh

from bokeh.resources import CDN
from bokeh.embed import file_html

import matplotlib.pyplot as plt  # , mpld3
import pyfits


# for now, no ratio in data (no rates A, B or C in this dataset)

only_flux = True
scale_flux = False

hdulist = pyfits.open('../iirc_data/all_data_for_ml.fits')
data = hdulist[1].data

X_flux, X, data_thr, data_fr_en, data_fr_err = get_iirc_data(
    data, only_flux=only_flux, scale_flux=scale_flux, thresholded=True)

# GMM with 3 components:
np.random.seed(0)

gmm = GeneralMixtureModel(MultivariateGaussianDistribution, n_components=3)
gmm.fit(X)
preds = gmm.predict(X)
probs = gmm.predict_proba(X)

data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "yellow", "blue", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+1]

covs = np.array([np.array(gmm.distributions[m].parameters[1])
                 for m in range(len(gmm.distributions))])
means = np.array([np.array(gmm.distributions[m].parameters[0])
                  for m in range(len(gmm.distributions))])


if only_flux:
    Html_file = open("gmm3_sklearn_files/gmm3_sklearn_only_flux.html", "w")
else:
    Html_file = open("gmm3_sklearn_files/gmm3_sklearn.html", "w")


# # uncomment  to show interactive probas:
# p = plot_probas(data_thr, probs)
# plt.show()

data5 = pd.DataFrame({name: data_thr[name]
                      for name in ['0', '20', '40', '70', 'preds']})

covs_ = np.array([covs[j][[0, 20, 40, 70], :][:, [0, 20, 40, 70]]
                  for j in range(covs.shape[0])])
means_ = np.array([means[j][[0, 20, 40, 70]] for j in range(means.shape[0])])
p = scatter_matrix(data5, covs=covs_, means=means_, color_key=color_key,
                   covs_indices=['0', '20', '40', '70'],
                   plot_width=200, plot_height=200)

html = file_html(p, CDN, "sklearn gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')

# p = interactive_img_ds(data_thr, 'rateCA', 'rate')
# waiting for InteractiveImage -> html

fig = plot_probs_datashader(probs)
html = file_html(fig, CDN, "probas with datashader")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# linked brushing probas:
x = 1
y = 60
covs_xy = [covs[j][[x, y]][:, [x, y]] for j in range(len(covs))]
means_xy = [means[j][[x, y]] for j in range(len(covs))]

data_probs = data_thr.copy()
for j in range(probs.shape[1]):
    data_probs['probs'+str(j)] = pd.Series(probs[:, j])

linkbru = plot_probs_bokeh_linked_brushing(data_probs, percent10=False,
                                           x_name='1', y_name='60',
                                           covs=covs_xy, means=means_xy)
html = file_html(linkbru, CDN, "pomegranate gmm with 3 components")
Html_file.write(html)


# fig = scatter_matrix_seaborn(data_thr)
# plt.title('seaborn_scatterplot')
# fig.fig.savefig('gmm3_sklearn_files/gmm3_sklearn_seaborn_scatterplot.png')
# data_uri = open('gmm3_sklearn_files/gmm3_sklearn_seaborn_scatterplot.png',
#                 'rb').read().encode('base64').replace('\n', '')
# img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
# Html_file.write(img_tag)
Html_file.close()
