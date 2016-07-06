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
np.random.seed(0)

hdulist = pyfits.open('all_data_for_ml.fits')
data = hdulist[1].data

# we need to change the byte order for fits -> dataframe:
data_fr = pd.DataFrame(data['flux'].byteswap().newbyteorder())
# convert attributes to str:
data_fr.columns = [str(j) for j in data_fr.columns]

if not only_flux:

    names = [data.columns[j].name for j in range(len(data.columns))]

    # rm str attribute:
    names.remove('block')

    # for now, rm multidim attributes:
    names.remove('en_lo')
    names.remove('en_hi')
    names.remove('flux')
    names.remove('flux_err')

    # also remove error terms, tstart and tstop:
    names.remove('rms1')
    names.remove('rms2')
    names.remove('rms3')
    names.remove('rms4')
    names.remove('tstart')
    names.remove('tstop')

    # we need to change the byte order for fits -> dataframe:
    data_fr2 = pd.DataFrame({name: data[name].byteswap().newbyteorder()
                             for name in names})

    data_fr = pd.concat([data_fr, data_fr2], axis=1)
# rm the rows with nan values:
data_fr = data_fr.dropna()

# rm too large values except for 'orbit':
if only_flux:
    data_thr = mask(data_fr)
else:
    data_thr = mask(data_fr, ['orbitalphase', 'smoothorbitalphase'])

# XXX try fusing columns 2 by 2:
data_thr2 = pd.DataFrame()
for j in range(len(data_thr.columns) / 2 - 1):
    data_thr2[str(j)] = (data_thr[str(2*j)] + data_thr[str(2*j+1)]) / 2.
data_thr = data_thr2
# XXX and again:
data_thr2 = pd.DataFrame()
for j in range(len(data_thr.columns) / 2 - 1):
    data_thr2[str(j)] = (data_thr[str(2*j)] + data_thr[str(2*j+1)]) / 2.
data_thr = data_thr2

X = np.concatenate([np.array(data_thr[name]).reshape(-1, 1)
                    for name in data_thr.columns], axis=1)
if only_flux:
    Html_file = open("gmm3_sklearn_files/gmm3_sklearn_only_flux.html", "w")
else:
    Html_file = open("gmm3_sklearn_files/gmm3_sklearn.html", "w")

# gmm with 3 comp from pomegranate:
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


# # uncomment  to show interactive probas:
# p = plot_probas(data_thr, probs)
# plt.show()


p = scatter_matrix(data_thr, covs=covs, means=means, color_key=color_key)
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


# fig = scatter_matrix_seaborn(data_thr)
# plt.title('seaborn_scatterplot')
# fig.fig.savefig('gmm3_sklearn_files/gmm3_sklearn_seaborn_scatterplot.png')
# data_uri = open('gmm3_sklearn_files/gmm3_sklearn_seaborn_scatterplot.png',
#                 'rb').read().encode('base64').replace('\n', '')
# img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
# Html_file.write(img_tag)
Html_file.close()
