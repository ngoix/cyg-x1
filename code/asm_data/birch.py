from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from visualization_fct import *
# from bokeh.plotting import output_file, show, save
# ACHTUNG!!! log transform of rateCA or without orbit may be activated
from bokeh.resources import CDN
from bokeh.embed import file_html



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
Html_file = open("clustering_files/birch.html", "w")

scaler = StandardScaler()
X = scaler.fit_transform(X)


for n_clusters in range(2, 10):

    km = Birch(n_clusters=n_clusters)
    preds = km.fit_predict(X)

    print "components:", set(preds)
    print np.bincount(preds)

    data_thr['preds'] = pd.Series(preds).astype("category")

    color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
                 "brown", "green", "orange"] * 2  # Spectral9
    # color_key = color_key[:len(set(preds))+2]


    # single plot rateCA vs rate with predicted classes and ellipses:

    single_plot = bokeh_datashader_plot(data_thr, covs=None, means=None,
                                        x_name='rateCA',
                                        y_name='rate',
                                        plot_width=900, plot_height=300,
                                        pixel_width=3000, pixel_height=1000,
                                        spread=True, color_key=color_key)
    html = file_html(single_plot, CDN, "birch")
    Html_file.write(html)
    # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
    # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
    # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


Html_file.close()
