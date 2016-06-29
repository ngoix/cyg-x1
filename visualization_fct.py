import numpy as np
import sys
sys.path.insert(20, '/home/nicolas/anaconda2/lib/python2.7/site-packages')

import datashader as ds
import datashader.transfer_functions as tf
from datashader.callbacks import InteractiveImage

# from bokeh.palettes import Spectral3, Spectral9
from bokeh.io import gridplot
from bokeh.plotting import Figure

import seaborn as sns
import pandas as pd


def mask(data, exception_cols, p=99.9):
    mask = np.ones(data.shape[0]).astype(bool)
    for col in data.columns:
        if col not in exception_cols:
            mask = mask & (data[col] < np.percentile(data[col], p))
    data_thr = pd.DataFrame({col: data[col][mask] for col in data.columns})
    return data_thr


def base_plot(data, x_name, y_name):
    p = Figure(
        x_range=(0, data[x_name].max()),
        y_range=(0, data[y_name].max()),
        tools='pan,wheel_zoom,box_zoom,reset',
        plot_width=800,
        plot_height=300,
    )
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.axis_label = x_name
    p.yaxis.axis_label = y_name
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'
    return p


def myplot(data, x_name='rateCA', y_name='rate', preds=True,
           pred_name='preds'):
    def create_image(x_range, y_range, w, h):
        cvs = ds.Canvas(plot_height=300, plot_width=900, x_range=x_range,
                        y_range=y_range)
        if preds is True:
            agg = cvs.points(data, x_name, y_name, ds.count_cat(pred_name))
            color_key = ['red', 'blue', 'yellow']  # Spectral9
            img = tf.colorize(agg, color_key)
            mask_img = np.array([[1, 1, 1, 1, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1]])

            img2 = tf.spread(img, mask=mask_img)  # to make pixel larger
            return img2
        else:
            agg = cvs.points(data, x_name, y_name, ds.count())
            return tf.interpolate(agg)

    p = base_plot(data, x_name, y_name)
    return InteractiveImage(p, create_image)


def bokeh_datashader_plot(data, x_name='rateCA', y_name='rate',
                          pred_name='preds', title=None, spread=False):

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

    xmin = np.percentile(data[x_name], 0.1)
    xmax = np.percentile(data[x_name], 99)
    ymin = np.percentile(data[y_name], 0.1)
    ymax = np.percentile(data[y_name], 99)

    cvs = ds.Canvas(plot_width=500,
                    plot_height=500,
                    x_range=(xmin, xmax),
                    y_range=(ymin, ymax))

    agg = cvs.points(data, x_name, y_name, ds.count_cat(pred_name))
    color_key = ['blue', 'red', 'yellow']  # Spectral9
    img = tf.colorize(agg, color_key)
    if spread:  # to make pixel larger
        mask_img = np.array([[1, 1, 1, 1, 1],
                             [1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1],
                             [1, 1, 1, 1, 1]])
        img = tf.spread(img, mask=mask_img)

    fig = Figure(x_range=(xmin, xmax),
                 y_range=(ymin, ymax),
                 plot_width=150,
                 plot_height=150,
                 title=title,
                 tools=TOOLS)

#    fig.background_fill_color = 'black'
#    fig.toolbar_location = None
    fig.axis.visible = False
    fig.grid.grid_line_alpha = 0
    fig.min_border_left = 0
    fig.min_border_right = 0
    fig.min_border_top = 0
    fig.min_border_bottom = 0

    fig.image_rgba(image=[img.data],
                   x=[xmin],
                   y=[ymin],
                   dw=[xmax-xmin],
                   dh=[ymax-ymin])
    return fig


def scatter_matrix(data, pred_name='preds', spread=False):
    figs = []
    columns = list(data.columns)
    columns.remove(pred_name)
    for y_name in columns:
        for x_name in columns:
            if y_name == x_name:
                fig = bokeh_datashader_plot(data, x_name=x_name, y_name=y_name,
                                            pred_name=pred_name, title=x_name,
                                            spread=spread)
            else:
                fig = bokeh_datashader_plot(data, x_name=x_name, y_name=y_name,
                                            pred_name=pred_name, spread=spread)
            figs += [fig]
    f = zip(*[iter(figs)]*len(columns))
    f = map(list, f)
    return gridplot(f)


def scatter_matrix_seaborn(data):
    sns.set()
    sns.pairplot(data, hue="preds", palette="Set2",
                 diag_kind="kde", size=1.7)  # ,hue_order=['3', '2', '1'])
