from visualization_fct import *
# from bokeh.plotting import output_file, show, save

from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.plotting import show


from pomegranate import GeneralMixtureModel
from pomegranate import MultivariateGaussianDistribution

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
data_thr = data

np.random.seed(0)

if without_CA:
    X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
              data_thr.rateC]
    Html_file = open("gmm3_pomegranate_files/gmm3_pomegranate_without_rateCA.html", "w")
else:
    X = np.c_[data_thr.orbit, data_thr.rate, data_thr.rateA, data_thr.rateB,
              data_thr.rateC, data_thr.rateCA]
    Html_file = open("gmm3_pomegranate_files/gmm3_pomegranate.html", "w")


gmm = GeneralMixtureModel(MultivariateGaussianDistribution, n_components=3)
gmm.fit(X)

preds = gmm.predict(X)
probs = gmm.predict_proba(X)

data_thr['preds'] = pd.Series(preds).astype("category")

color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
             "brown", "green", "orange"]  # Spectral9
color_key = color_key[:len(set(preds))+1]

covs = np.array([np.array(gmm.distributions[m].parameters[1]) for m in range(len(gmm.distributions))])
means = np.array([np.array(gmm.distributions[m].parameters[0]) for m in range(len(gmm.distributions))])

# # uncomment  to show interactive probas:
# p = plot_probas(data_thr, probs)
# plt.show()

# p = interactive_img_ds(data_thr, 'rateCA', 'rate')
# # waiting for InteractiveImage -> html


p = scatter_matrix(data_thr, spread=False, covs=covs, means=means,
                   color_key=color_key)
html = file_html(p, CDN, "pomegranate gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')

x = 5
y = 1
covs_xy = [covs[j][[x, y]][:, [x, y]] for j in range(covs.shape[0])]
means_xy = [means[j][[x, y]] for j in range(covs.shape[0])]

single_plot = bokeh_datashader_plot(data_thr, covs=covs_xy, means=means_xy,
                                    x_name='rateCA',
                                    y_name='rate',
                                    plot_width=900, plot_height=300,
                                    pixel_width=3000, pixel_height=1000,
                                    spread=False, color_key=color_key)
html = file_html(single_plot, CDN, "pomegranate gmm with 3 components")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


p = Figure()
rateCA = np.array(data_thr['rateCA'])
hist, edges = np.histogram(rateCA[rateCA > 20], bins=100)
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
html = file_html(p, CDN, "hist rateCA")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


fig = plot_probs_datashader(probs)
html = file_html(fig, CDN, "probas with datashader")
Html_file.write(html)
Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


# fig = scatter_matrix_seaborn(data_thr)
# plt.title('seaborn_scatterplot')
# fig.fig.savefig('gmm3_pomegranate_files/gmm3_pomegranate_seaborn_scatterplot.png')
# data_uri = open('gmm3_pomegranate_files/gmm3_pomegranate_seaborn_scatterplot.png',
#                 'rb').read().encode('base64').replace('\n', '')
# img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
# Html_file.write(img_tag)
Html_file.close()
