# import pdb
import numpy as np

from sklearn.utils import shuffle

import datashader as ds
import datashader.transfer_functions as tf
from datashader.bokeh_ext import InteractiveImage

from bokeh.palettes import Spectral9
from bokeh.io import gridplot
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource

import seaborn as sns
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

from collections import OrderedDict

from sklearn.preprocessing import scale

def mask(data, exception_cols=[], p=99.9):
    mask = np.ones(data.shape[0]).astype(bool)
    for col in data.columns:
        if col not in exception_cols:
            mask = mask & (data[col] < np.percentile(data[col], p))
    data_thr = pd.DataFrame(np.array([data[col][mask]
                                      for col in data.columns]).T)
    data_thr.columns = data.columns
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


def interactive_img_ds(data, x_name='rateCA', y_name='rate', preds=True,
                       pred_name='preds'):
    def create_image(x_range, y_range, w, h):
        cvs = ds.Canvas(plot_height=300, plot_width=900, x_range=x_range,
                        y_range=y_range)
        if preds is True:
            agg = cvs.points(data, x_name, y_name, ds.count_cat(pred_name))
            color_key = ["red", "blue", "yellow", "grey", "black", "purple",
                         "pink", "brown", "green", "orange"]  # Spectral9
            img = tf.colorize(agg, color_key)
            mask_img = np.array([[1, 1, 1, 1, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 1],
                                 [1, 1, 1, 1, 1]])

            img2 = tf.spread(img, mask=mask_img)  # to make pixel larger
            return img2
        else:
            agg = cvs.points(data, x_name, y_name, ds.any())
            return tf.interpolate(agg)

    p = base_plot(data, x_name, y_name)
    return InteractiveImage(p, create_image)


def bokeh_datashader_plot(data, covs=None, means=None, covs_indices=None,
                          x_name='rateCA',
                          y_name='rate',
                          pred_name='preds', title=None,
                          plot_width=150, plot_height=150,
                          pixel_width=500, pixel_height=500,
                          spread=False, color_key=None):

    if color_key is None:
        color_key = ["red", "blue", "yellow", "grey", "black", "purple",
                     "pink", "brown", "green", "orange"]

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

    xmin_p = np.percentile(data[x_name], 0.1)
    xmax_p = np.percentile(data[x_name], 99)
    ymin_p = np.percentile(data[y_name], 0.1)
    ymax_p = np.percentile(data[y_name], 99)

    xmin = np.percentile(data[x_name], 0.01)
    xmax = np.percentile(data[x_name], 99.9)
    ymin = np.percentile(data[y_name], 0.01)
    ymax = np.percentile(data[y_name], 99.9)

    # xmin = data[x_name].min()
    # xmax = data[x_name].max()
    # ymin = data[y_name].min()
    # ymax = data[y_name].max()

    cvs = ds.Canvas(plot_width=pixel_width,
                    plot_height=pixel_height,
                    x_range=(xmin, xmax),
                    y_range=(ymin, ymax))

    agg = cvs.points(data, x_name, y_name, ds.count_cat(pred_name))
    img = tf.colorize(agg, color_key)
    if spread:  # to make pixel larger
        mask_img = np.array([[1, 1, 1, 1, 1],
                             [1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1],
                             [1, 1, 1, 1, 1]])
        img = tf.spread(img, mask=mask_img)

    fig = Figure(x_range=(xmin_p, xmax_p),
                 y_range=(ymin_p, ymax_p),
                 plot_width=plot_width,
                 plot_height=plot_height,
                 title=title,
                 tools=TOOLS)

    fig.image_rgba(image=[img.data],
                   x=[xmin],
                   y=[ymin],
                   dw=[xmax-xmin],
                   dh=[ymax-ymin])

    if covs is not None:
            # pdb.set_trace()
            for n_comp in range(len(covs)):
                cov = covs[n_comp]
                mean = means[n_comp]
                v, w = np.linalg.eigh(cov)
                e0 = w[0] / np.linalg.norm(w[0])
                e1 = w[1] / np.linalg.norm(w[1])
                t = np.linspace(0, 2 * np.pi, 10000)
                # 4.605 corresponds to 90% quantile:
                a = (mean[0]
                     + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[0]
                     + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[0])
                b = (mean[1]
                     + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[1]
                     + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[1])
                # ellipse = pd.DataFrame(np.c_[a, b], columns=['a', 'b'])
                # agg_ell = cvs.line(ellipse, 'a', 'b', agg=ds.any())
                # img_ell = tf.interpolate(agg_ell, cmap=[color_key[n_comp]])
                # img = tf.stack(img, img_ell)

                fig.line(a, b, color=color_key[n_comp])

#    fig.background_fill_color = 'black'
#    fig.toolbar_location = None
    fig.axis.visible = True
    fig.grid.grid_line_alpha = 0
    fig.min_border_left = 0
    fig.min_border_right = 0
    fig.min_border_top = 0
    fig.min_border_bottom = 0
    return fig


def scatter_matrix(data, covs=None, means=None, covs_indices=None,
                   pred_name='preds',
                   spread=False, color_key=None,
                   plot_width=150, plot_height=150, datashader=True):
    # if the covariance matrix has lower dim than data (because the estimator
    # has not been trained with all features), the missing features are assumed
    # to be the last of data.columns excepting if covs_indices is not None.

    if color_key is None:
        color_key = ["red", "blue", "yellow", "grey", "black", "purple",
                     "pink", "brown", "green", "orange"]
    figs = []
    columns = list(data.columns)
    columns.remove(pred_name)
    for y, y_name in enumerate(columns):
        for x, x_name in enumerate(columns):
            if y_name == x_name:
                if datashader:
                    fig = bokeh_datashader_plot(data, x_name=x_name,
                                                y_name=y_name,
                                                pred_name=pred_name,
                                                title=x_name,
                                                spread=spread,
                                                color_key=color_key,
                                                plot_width=plot_width,
                                                plot_height=plot_height)
                else:
                    fig = bokeh_plot_cov(data, x_name=x_name, y_name=y_name,
                                         pred_name=pred_name, title=x_name,
                                         color_key=color_key,
                                         plot_width=plot_width,
                                         plot_height=plot_height)
            else:
                if covs is None:
                    plot_ellipse = False
                else:
                    plot_ellipse = ((max(x, y) < covs.shape[1]) if covs_indices is None else (x_name in covs_indices) * (y_name in covs_indices))
                if plot_ellipse:
                    covs_xy = [covs[j][[x, y]][:, [x, y]]
                               for j in range(covs.shape[0])]
                    means_xy = [means[j][[x, y]]
                                for j in range(covs.shape[0])]

                    if datashader:
                        fig = bokeh_datashader_plot(data, covs_xy, means_xy,
                                                    covs_indices,
                                                    x_name=x_name,
                                                    y_name=y_name,
                                                    pred_name=pred_name,
                                                    spread=spread,
                                                    color_key=color_key,
                                                    plot_width=plot_width,
                                                    plot_height=plot_height)
                    else:
                        fig = bokeh_plot_cov(data, covs_xy, means_xy,
                                             covs_indices,
                                             x_name=x_name,
                                             y_name=y_name,
                                             pred_name=pred_name,
                                             color_key=color_key,
                                             plot_width=plot_width,
                                             plot_height=plot_height)

                else:
                    if datashader:
                        fig = bokeh_datashader_plot(data,
                                                    x_name=x_name,
                                                    y_name=y_name,
                                                    pred_name=pred_name,
                                                    spread=spread,
                                                    color_key=color_key,
                                                    plot_width=plot_width,
                                                    plot_height=plot_height)
                    else:
                        fig = bokeh_plot_cov(data,
                                             x_name=x_name,
                                             y_name=y_name,
                                             pred_name=pred_name,
                                             color_key=color_key,
                                             plot_width=plot_width,
                                             plot_height=plot_height)
            figs += [fig]
    f = zip(*[iter(figs)]*len(columns))
    f = map(list, f)
    return gridplot(f)


def scatter_matrix_seaborn(data, y='preds', vars=None, size=1.7):
    sns.set()
    return sns.pairplot(data, hue=y, palette="Set2",
                        diag_kind="kde", vars=vars, size=size)


# plot the probas:


def plot_probas(data, probs, cola='rateCA', colb='rate',
                delta=10, window=100, color_key=None):

    if color_key is None:
        color_key = ["red", "blue", "yellow", "grey", "black", "purple",
                     "pink", "brown", "green", "orange"]

    preds = data['preds']
    cm = ListedColormap(color_key)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].hexbin(data[cola], data[colb], cmap='coolwarm',
                   bins='log')  # cmap='viridis'
    axes[0].set_xlabel('rateC/rateA')
    axes[0].set_ylabel('rate')

    l = axes[0].scatter(data[cola][:5 * window], data[colb][:5 * window],
                        c=preds[:5 * window], alpha=1, s=20, cmap=cm)
    axes[0].set_xlabel(cola)
    axes[0].set_ylabel(colb)

    # axfreq = plt.axes([0.1, 0.0, 0.8, 0.03])
    sfreq = Slider(axes[1], 'Start', 0, len(data[cola]), valinit=0,
                   facecolor=None, alpha=.1)
    for prob, color in zip(probs.T, color_key):
        axes[1].plot(pd.Series(prob).rolling(window=1).mean(),
                     color=color, alpha=.9, linewidth=1)

    def update(val):
        start = int(sfreq.val)
        inds = range(start, start + window)
        l.set_offsets(np.c_[data[cola].iloc[inds], data[colb].iloc[inds]])
        l.set_array(preds[inds])
        plt.gcf().canvas.draw_idle()
    sfreq.on_changed(update)

    axes[0].set_xlim(data[cola].min(), data[cola].max())
    axes[0].set_ylim(data[colb].min(), data[colb].max())
    axes[1].set_ylim(-0.1, 1.1)

    return fig  # plt.show()


def plot_probs_bokeh_linked_brushing(data,
                                     prob_names=['probs0', 'probs1', 'probs2'],
                                     pred_name='preds',  # time_name='MJD',
                                     preds=True,
                                     percent10=True,
                                     x_name='rateCA', y_name='rate',
                                     covs=None, means=None,
                                     spread=False, color_key=None,
                                     plot_width=900, plot_height=300,
                                     # radius1=0.01,
                                     # radius2=30., line_width2=0,
                                     title=None):
    '''
    return a linked brushing interactive bokeh plot
    '''
    TOOLS = "wheel_zoom,box_zoom,reset,box_select,pan"  # ,lasso_select,save"

    if color_key is None:
        color_key = ["red", "blue", "grey", "yellow", "black", "purple",
                     "pink", "brown", "green", "orange"]  # Spectral9

    xmin_p = np.percentile(data[x_name], 0.1)
    xmax_p = np.percentile(data[x_name], 99)
    ymin_p = np.percentile(data[y_name], 0.1)
    ymax_p = np.percentile(data[y_name], 99)

    # xmin = np.percentile(data[x_name], 0.01)
    # xmax = np.percentile(data[x_name], 99.9)
    # ymin = np.percentile(data[y_name], 0.01)
    # ymax = np.percentile(data[y_name], 99.9)

    fig = Figure(x_range=(xmin_p, xmax_p),
                 y_range=(ymin_p, ymax_p),
                 plot_width=plot_width,
                 plot_height=plot_height,
                 title=title,
                 tools=TOOLS)

    if percent10:
        X = np.concatenate([np.array(
            data[name]).reshape(-1, 1) for name in list(data.columns)],
                           axis=1)
        #X = shuffle(X)
        a = X.shape[0] / 10
        data10 = pd.DataFrame(X[:a])
        data10.columns = data.columns
        source = ColumnDataSource(data10)
        colors = [color_key[int(x)] for x in data10[pred_name]]
        n_samples = data10.shape[0]
    else:
        source = ColumnDataSource(data)
        colors = [color_key[x] for x in data[pred_name]]
        n_samples = data.shape[0]

    fig.circle(x_name, y_name, source=source, color=colors)  # , radius=radius1

    if covs is not None:
            for n_comp in range(len(covs)):
                cov = covs[n_comp]
                mean = means[n_comp]
                v, w = np.linalg.eigh(cov)
                e0 = w[0] / np.linalg.norm(w[0])
                e1 = w[1] / np.linalg.norm(w[1])
                t = np.linspace(0, 2 * np.pi, 10000)
                # 4.605 corresponds to 90% quantile:
                a = (mean[0]
                     + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[0]
                     + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[0])
                b = (mean[1]
                     + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[1]
                     + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[1])

                fig.line(a, b, color=color_key[n_comp])

    print n_samples
    fig2 = Figure(x_range=(0, n_samples),
                  y_range=(-0.01, 1.01),
                  plot_width=plot_width,
                  plot_height=plot_height,
                  title=title,
                  tools=TOOLS)
    for n, prob_name in enumerate(prob_names):
        fig2.circle(range(n_samples), prob_name, source=source,
                    color=color_key[n])
        # fig2.line(range(n_samples), prob_name, source=source,
        #           color=color_key[n])
    p = gridplot([[fig], [fig2]])
    return p


def plot_probs_datashader(probs, title=None, color_key=None):

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

    xmin = 0
    xmax = probs.shape[0]
    ymin = 0
    ymax = 1

    states = [str(c) for c in range(probs.shape[1])]
    probs_df = {str(c): probs[:, c] for c in range(probs.shape[1])}
    probs_df['Time'] = np.linspace(0, probs.shape[0], probs.shape[0])
    # TODO: also try: probs_df['Time'] = data['MJD']
    probs_df = pd.DataFrame(probs_df)

    cvs = ds.Canvas(x_range=(xmin, xmax), y_range=(ymin, ymax),
                    plot_height=300, plot_width=30000)
    aggs = OrderedDict((c, cvs.line(probs_df, 'Time', c)) for c in states)

    if color_key is None:
        color_key = ["red", "blue", "grey", "yellow", "black", "purple",
                     "pink", "brown", "green", "orange"]  # Spectral9
    color_key = color_key[:probs.shape[1]]

    imgs = [tf.interpolate(aggs[i], cmap=[c]) for i, c in zip(states,
                                                              color_key)]
    img = tf.stack(*imgs)

    fig = Figure(x_range=(xmin, xmax),
                 y_range=(ymin, ymax),
                 plot_width=900,
                 plot_height=300,
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


# ###########  iirc data:


def get_iirc_data(data, scale_flux=True, only_flux=False, thresholded=True):
    # ## flux attribute:
    # we need to change the byte order for fits -> dataframe:
    if scale_flux:
        data_fr = pd.DataFrame(scale(
            data['flux'].byteswap().newbyteorder()))
    else:
        data_fr = pd.DataFrame(data['flux'].byteswap().newbyteorder())
    # convert attributes to str:
    flux_attr = [str(j) for j in data_fr.columns]
    data_fr.columns = flux_attr

    # ## average energy attribute:
    en = (data['en_lo'].byteswap().newbyteorder()
          + data['en_hi'].byteswap().newbyteorder()) / 2.
    data_fr_en = pd.DataFrame(en)
    # convert attributes to str:
    en_attr = ['en_' + str(j) for j in data_fr_en.columns]
    data_fr_en.columns = en_attr

    # ## errors attribute:
    err = data['flux_err'].byteswap().newbyteorder()
    data_fr_err = pd.DataFrame(err)
    # convert attributes to str:
    err_attr = ['err_' + str(j) for j in data_fr_en.columns]
    data_fr_err.columns = err_attr

    # ## other attributes:
    names = [data.columns[j].name for j in range(len(data.columns))]
    # rm str attribute:
    names.remove('block')
    names.remove('orbitalphase')
    names.remove('smoothorbitalphase')
    # rm useless attributes or already in:
    names.remove('en_lo')  # already in
    names.remove('en_hi')  # already in
    names.remove('flux')  # already in
    names.remove('flux_err')  # already in
    # also remove error terms, tstart and tstop:
    # names.remove('rms1')
    # names.remove('rms2')
    # names.remove('rms3')
    # names.remove('rms4')
    names.remove('tstart')
    names.remove('tstop')

    data_fr2 = pd.DataFrame({name: data[name].byteswap().newbyteorder()
                             for name in names})

    data_fr = pd.concat([data_fr, data_fr2, data_fr_en, data_fr_err], axis=1)

    # rm the rows with nan values:
    data_fr = data_fr.dropna()

    if thresholded:
        # rm too large values except for the orbits, 'gamma' and energy attr:
        unthresholded_attr = ['orbitalphase', 'smoothorbitalphase',
                              'gamma'] + en_attr + err_attr
        data_thr = mask(data_fr, unthresholded_attr)

    # define X (without gamma and nrj attribute):
    col_X = list(data_thr.columns)
    col_X.remove('gamma')
    for i in range(len(en_attr)):
        col_X.remove(en_attr[i])
    for i in range(len(err_attr)):
        col_X.remove(err_attr[i])

    X = np.concatenate([np.array(
        data_thr[name]).reshape(-1, 1) for name in col_X], axis=1)
    X_flux = np.concatenate([np.array(
        data_thr[name]).reshape(-1, 1) for name in flux_attr], axis=1)

    if only_flux:
        X = X_flux

    # restrict data_fr_en and data_fr_err to mask and nonnan values:
    data_fr_en = data_thr[data_fr_en.columns]
    data_fr_err = data_thr[data_fr_err.columns]

    a = np.array(data_thr['gamma'])
    y = (a > 2.5).astype('int') + (a > 2).astype('int')
    data_thr['y'] = y

    return X_flux, X, data_thr, data_fr_en, data_fr_err


def bokeh_plot_cov(data, covs=None, means=None, covs_indices=None,
                   x_name='rateCA',
                   y_name='rate',
                   pred_name='preds', title=None,
                   plot_width=150, plot_height=150,
                   color_key=None):

    if color_key is None:
        color_key = ["red", "blue", "yellow", "grey", "black", "purple",
                     "pink", "brown", "green", "orange"]

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

    xmin_p = np.percentile(data[x_name], 0.1)
    xmax_p = np.percentile(data[x_name], 99)
    ymin_p = np.percentile(data[y_name], 0.1)
    ymax_p = np.percentile(data[y_name], 99)

    xmin = np.percentile(data[x_name], 0.01)
    xmax = np.percentile(data[x_name], 99.9)
    ymin = np.percentile(data[y_name], 0.01)
    ymax = np.percentile(data[y_name], 99.9)

    # xmin = data[x_name].min()
    # xmax = data[x_name].max()
    # ymin = data[y_name].min()
    # ymax = data[y_name].max()

    fig = Figure(x_range=(xmin_p, xmax_p),
                 y_range=(ymin_p, ymax_p),
                 plot_width=plot_width,
                 plot_height=plot_height,
                 title=title,
                 tools=TOOLS)

    source = ColumnDataSource(data)
    colors = [color_key[x] for x in data[pred_name]]

    fig.circle(x_name, y_name, source=source, color=colors)  # , radius=radius1

    if covs is not None:
            # pdb.set_trace()
            for n_comp in range(len(covs)):
                cov = covs[n_comp]
                mean = means[n_comp]
                v, w = np.linalg.eigh(cov)
                e0 = w[0] / np.linalg.norm(w[0])
                e1 = w[1] / np.linalg.norm(w[1])
                t = np.linspace(0, 2 * np.pi, 10000)
                # 4.605 corresponds to 90% quantile:
                a = (mean[0]
                     + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[0]
                     + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[0])
                b = (mean[1]
                     + np.sqrt(4.605 * v[0]) * np.cos(t) * e0[1]
                     + np.sqrt(4.605 * v[1]) * np.sin(t) * e1[1])
                # ellipse = pd.DataFrame(np.c_[a, b], columns=['a', 'b'])
                # agg_ell = cvs.line(ellipse, 'a', 'b', agg=ds.any())
                # img_ell = tf.interpolate(agg_ell, cmap=[color_key[n_comp]])
                # img = tf.stack(img, img_ell)

                fig.line(a, b, color=color_key[n_comp])

#    fig.background_fill_color = 'black'
#    fig.toolbar_location = None
    fig.axis.visible = True
    fig.grid.grid_line_alpha = 0
    fig.min_border_left = 0
    fig.min_border_right = 0
    fig.min_border_top = 0
    fig.min_border_bottom = 0
    return fig