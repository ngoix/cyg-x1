from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from visualization_fct import *
# from bokeh.plotting import output_file, show, save

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
Html_file = open("clustering_files/ward.html", "w")

# consider only 10000 data (spectralclustering memory complexity):
ind = np.array(10000 * [1] + (X.shape[0] - 10000) * [0]).astype(bool)
ind = shuffle(ind)
data_thr10 = pd.DataFrame(X[ind])
data_thr10.columns = data.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X[ind]

for n_clusters in range(2, 10):

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    km = AgglomerativeClustering(n_clusters=n_clusters,
                                 linkage='ward',
                                 connectivity=connectivity)
    preds = km.fit_predict(X)

    print "components:", set(preds)
    print np.bincount(preds)

    data_thr10['preds'] = pd.Series(preds).astype("category")
    color_key = ["red", "blue", "yellow", "grey", "black", "purple", "pink",
                 "brown", "green", "orange"]

    title = str(np.bincount(preds))
    TOOLS = "wheel_zoom,box_zoom,reset,box_select,pan"
    plot_width = 900
    plot_height = 300
    x_name = 'rateCA'
    y_name = 'rate'
    xmin_p = np.percentile(data_thr10[x_name], 0.1)
    xmax_p = np.percentile(data_thr10[x_name], 99)
    ymin_p = np.percentile(data_thr10[y_name], 0.1)
    ymax_p = np.percentile(data_thr10[y_name], 99)

    fig = Figure(x_range=(xmin_p, xmax_p),
                 y_range=(ymin_p, ymax_p),
                 plot_width=plot_width,
                 plot_height=plot_height,
                 title=title,
                 tools=TOOLS)

    source = ColumnDataSource(data_thr10)
    colors = [color_key[int(x)] for x in data_thr10['preds']]
    colors = np.array(colors)
    fig.circle(x_name, y_name, source=source, color=colors)  # , radius=radius1

    # plot the manual decisions (lines):
    t = np.linspace(0, 2, 1000)
    y1 = np.max([20 * np.ones(t.shape[0]), 55 * (t - 0.28)], axis=0)
    y2 = np.max([20 * np.ones(t.shape[0]), 350 * (t - 0.28)], axis=0)
    fig.line(t, y1, color='black')
    fig.line(t, y2, color='black')

    html = file_html(fig, CDN, "ward_tree")
    Html_file.write(html)
    # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
    # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')
    # Html_file.write('<br><br><br><br><br><br><br><br><br><br><br><br>')


Html_file.close()
