import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from sherlock.sherlockengines import sherlockstats as ss
import seaborn as sns
import sys


def progress_show(min_val, max_val, c_val):
    max_length = 20
    print('\r', end='')
    print('Completed percentage:', 100*(c_val-min_val)/(max_val-min_val), '%', end='')
    sys.stdout.flush()


def show_distributions(x1, x2, title, x_titles=None, fig_size=(18, 15), show_plot=True):
    # TODO: It only works for two distributions
    fig, ax = plt.subplots(1,1,figsize=fig_size)
    if x_titles==None:
        dic = {'class 1': x1, 'class 2': x2}
    else:
        dic = {x_titles[0]: x1, x_titles[1]: x2}

    df = pd.DataFrame.from_dict(dic, orient='index')
    df = df.transpose()
    melted = df.melt(value_name='values', var_name='variables')
    melted = melted.dropna()
    sns.set(font_scale=3)  # crazy big

    sns.boxplot(x='variables', y='values', data=melted)
    ax.set_title(title, fontsize=28)

    if show_plot==True:
        plt.tight_layout()
        plt.show()

    return fig


def show_relationship(x1, x2, title=None, axes_labels=None, fig_size=(18, 15), show_plot=True):
    fig, ax = plt.subplots(1,1,figsize=fig_size)
    df = pd.DataFrame({'x1': x1, 'x2': x2})
    sns.regplot(data=df, y='x2', x='x1', ax=ax)
    if axes_labels is not None:
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
    ax.set_title(title)
    if show_plot:
        plt.show()

    return fig


def box_plot_show(x, values, title=None, show_plot=True, figsize=(30,18)):
    # plt.figure(figsize=(30, 18))

    fig1, ax = plt.subplots(1,2, figsize=figsize)
    sns.set(font_scale=2)  # crazy big

    dists = values
    df = pd.DataFrame(dists, columns=x)
    sns.boxplot(x='variables', y='values', data=df.melt(value_name='values', var_name='variables'), ax=ax[0])
    ax[0].tick_params(labelrotation=90)
    ax[0].set_title(title)
    ax[0].set_xlabel('Columns')

    # plt.subplot(122)
    dists = np.zeros((1000, values.shape[1]))
    for i in range(values.shape[1]):
        _, _, dists[:, i] = ss.bootstrap(values[:, i], .95, n_sample=1000)
    df = pd.DataFrame(dists, columns=x)
    sns.boxplot(x='variables', y='values', data=df.melt(value_name='values', var_name='variables'), ax=ax[1])
    ax[1].tick_params(labelrotation=90)
    title = ''.join([title, ' (bootstrapped)'])
    ax[1].set_title(title)
    ax[1].set_xlabel('Columns')
    if show_plot:
        plt.show()

    return ax, fig1

    # ss.bootstrap()


def network_show(G, df, node_color=None, node_size=None, title=None, layout='spring', color_bar_title=None,
                 show_plot=True, fig_size=(18, 15)):
    color_bar = True
    if node_color is None:
        node_color = "red"
        color_bar = False

    if node_size is None:
        node_size = 600

    # if layout is 'brain':
    #     pos, min_pos_x, min_pos_y, max_pos_x, max_pos_y, fig = brain_pos(G, reg_cords)
    fig, ax = plt.subplots(1,1,figsize=fig_size)

    if layout is 'circ':
        if node_color is None:
            pos = circ_layout(G, 0)
        else:
            pos = circ_layout(G, node_color)
    if layout is 'spring':
        pos = nx.spring_layout(G, k=0.9)
    if layout is 'kkl':
        pos = nx.kamada_kawai_layout(G)
    if layout is 'rnd':
        pos = nx.random_layout(G)

    edge_color = df['weights']
    # edge_color = "gray"

    # Graph with Custom nodes:
    colors = node_color
    #     cmap=plt.cm.RdYlBu_r
    cmap = plt.cm.autumn_r
    vmin = min(colors)
    vmax = max(colors)
    ax.set_title(title, fontsize=28)

    nx.draw(G, with_labels=True, node_size=node_size, node_color=node_color, edgecolors='k', node_shape="o", alpha=.9,
            font_size=18, font_color="black", font_weight="bold", linewidth=6,
            width=4, pos=pos, edge_cmap=plt.cm.gray, edge_color=edge_color, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)

    if color_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        hndl = plt.colorbar(sm)
        hndl.set_label(color_bar_title, fontsize=28)
        hndl.ax.tick_params(labelsize=28)

    if show_plot:
        plt.show()

    return ax, fig


def circ_layout(G, values):
    unique_values_list = list(range(len(np.unique(values))))
    number_unique_values = len(unique_values_list)

    all_nodes = np.asarray(G.nodes)

    if number_unique_values == 1:
        nodes_correspondence_category = np.full(all_nodes.shape[0], unique_values_list[0])
    else:
        nodes_correspondence_category = values

    nodes_correspondence_category = np.asarray(nodes_correspondence_category)

    degs = 360 / number_unique_values
    pos = dict()
    r = 0.5
    int_r = 0.2
    node_degress = np.double(np.asarray(G.degree)[:, 1])

    for i in range(number_unique_values):
        indx = np.where(i == nodes_correspondence_category)
        nodes = np.asarray(all_nodes)[indx]
        node_degress_c = node_degress[indx]

        deg = i * degs
        c = [r * np.sin(deg * np.pi / 180), r * np.cos(deg * np.pi / 180)]

        if number_unique_values == 1:
            c = [0.0, 0.0]

        int_degs = 360 / len(nodes)
        indx = 0

        for j in nodes:
            per = (np.random.rand(2) - 0.5) * .1
            int_deg = indx * int_degs
            p = [int_r * np.sin(int_deg * np.pi / 180) + c[0] + per[0],
                 int_r * np.cos(int_deg * np.pi / 180) + c[1] + per[1]]
            pos[j] = p
            indx += 1
    return pos


def get_network(df_org, thr=None):
    df = df_org.copy(deep=True)
    if thr is not None:
        df = df.where(df['weights'] > thr).dropna()

    G = nx.from_pandas_edgelist(df, 'from', 'to', ['weights'])
    return G, df


def get_linear_interp(values, mi, ma):
    min_s = min(values)
    max_s = max(values)
    if min_s == max_s:
        s_values = values/max_s
    else:
        s_values = ((ma-mi)/(max_s-min_s))*(values-min_s)+mi
    return s_values


def show_importance_nodes(node_size_org, node_color_org, title, variables_names, ax, emphasise=1.0):

    node_size = np.floor(np.power(get_linear_interp(node_size_org, 0., 1.0), emphasise) * 3000)
    node_color = np.floor(np.power(get_linear_interp(node_color_org, 0., 1.0), emphasise) * 1000)

    node_size = np.append(node_size, 3000 * .3)
    node_size = np.append(node_size, 3000 * .5)
    node_size = np.append(node_size, 3000 * .9)

    node_color = np.append(node_color, 1000 * 0.7)
    node_color = np.append(node_color, 1000 * 0.7)
    node_color = np.append(node_color, 1000 * 0.7)

    G = nx.Graph()

    for r in variables_names:
        G.add_node(r)

    min_pos_x, min_pos_y, max_pos_x, max_pos_y = [1000, 1000, -1000, -1000]
    pos = nx.kamada_kawai_layout(G)
    # pos = nx.random_layout(G)
    for i in pos.values():
        min_pos_x = min(i[0], min_pos_x)
        min_pos_y = min(i[1], min_pos_y)
        max_pos_x = max(i[0], max_pos_x)
        max_pos_y = max(i[1], max_pos_y)

    p_x = (max_pos_x-min_pos_x)/20
    p_y = (max_pos_y-min_pos_y)/20,

    G.add_node('~30%')
    pos['~30%'] = [min_pos_x + p_x, max_pos_y]

    G.add_node('~50%')
    pos['~50%'] = [min_pos_x + 2*p_x, max_pos_y]

    G.add_node('~90%')
    pos['~90%'] = [min_pos_x + 3*p_x, max_pos_y]

    # pos = nx.spectral_layout(G)
    # nx.draw(G, pos=pos)

    colors = node_color
    cmap = plt.cm.autumn_r
    vmin = min(colors)
    vmax = max(colors)
    nx.draw(G, with_labels=False,
            node_size=node_size, edgecolors='k',
            node_shape="o", alpha=.9, node_color=colors, vmin=vmin, vmax=vmax,
            font_size=24, font_color="black", font_weight="bold", linewidth=4,
            width=2, pos=pos, cmap=cmap, ax=ax)

    ax.text(min_pos_x, max_pos_y+p_y, 'Importance stability factor', fontsize=14)
    pos_higher = pos

    #     for k, v in pos.items():
    #         y_off = int(10*(np.random.rand()-0.5))  # offset on the y axis
    #         pos_higher[k] = (v[0], v[1]+y_off)

    nx.draw_networkx_labels(G, pos_higher, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    hndl = plt.colorbar(sm)
    hndl.set_label('Bagged importance level')

    ax.set_title(title, fontsize=20)
    #     fig.suptitle(''.join(['TR: ', str(title)]), fontsize=20)

