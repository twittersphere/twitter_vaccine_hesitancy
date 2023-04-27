import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from matplotlib.colors import LinearSegmentedColormap

class Figures:
    def __init__(self) -> None:
        sns.set_context("poster")

    def mid_point(self, values):
        return (np.max(values) + np.min(values)) / 2

    def draw_correlation_figure(self, df, x, y, x_label, y_label, figsize=(8, 8), point_size_multiplexer=1, save=None, paper_figure=None,
                                colorbar=True):
        sns.set_style("white")
        palette = sns.color_palette("crest", as_cmap=True)
        correlation = scipy.stats.spearmanr(df[[x, y]].values)

        plt.figure(figsize=figsize)
        line = f'$\\rho$={correlation.correlation:.2f}, $P$={correlation.pvalue:.2e}'
        fig = sns.scatterplot(data=df, x=x, y=y, hue='tweet_counts', s=(df['tweet_counts'].values)*point_size_multiplexer, legend=False,
                            palette=palette, edgecolors=None, linewidth=0)
        for idx, row in df.iterrows():
            fig.text(row[x], row[y], row['state'], horizontalalignment='center', verticalalignment='center', fontdict=dict(size=15, color='black'))
        sm = plt.cm.ScalarMappable(cmap=palette)
        sm.set_array(df['tweet_counts'].values)
        if colorbar:
            fig.figure.colorbar(sm, pad=0.02)
            fig.text(np.max(df[x].values), self.mid_point(df[y].values), 'Tweet Count', 
                    horizontalalignment='center', rotation=90, fontdict=dict(size=19, color='black')) # Change y value 
        sns.regplot(data=df, y=y, x=x, scatter=False)
        sns.despine(offset=5, trim=True)

        plt.plot(np.max(df[x].values), np.max(df[y].values), label=line, color='tab:blue')
        plt.legend(facecolor='white', loc='upper right', fontsize=15)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches='tight')

    def map_heatmap(self, df, geopandas_states, column, colors, figsize=(16, 8),
                    aspect=20, fontsize=14, min_max=None, fraction=0.024, ticks_margin=None, ticks_round=1, cmap_orient='vertical', save=None, paper_figure=None):
        sns.set_style()
        colormap = LinearSegmentedColormap.from_list(
            name='test', 
            colors=colors)

        geopandas_states = geopandas_states.join(df.set_index('state'), on='state').set_index('state').dropna()

        # create an axis with 2 insets − this defines the inset sizes
        fig, continental_ax = plt.subplots(figsize=figsize, frameon=False)
        alaska_ax = continental_ax.inset_axes([.08, .01, .20, .28])
        hawaii_ax = continental_ax.inset_axes([.28, .01, .15, .19])

        # Set bounds to fit desired areas in each plot
        continental_ax.set_xlim(-130, -64)
        continental_ax.set_ylim(22, 53)

        alaska_ax.set_ylim(51, 72)
        alaska_ax.set_xlim(-180, -127)

        hawaii_ax.set_ylim(18.8, 22.5)
        hawaii_ax.set_xlim(-160, -154.6)

        # Plot the data per area - requires passing the same choropleth parameters to each call
        # because different data is used in each call, so automatically setting bounds won’t work
        vmin, vmax = geopandas_states[column].agg(['min', 'max'])
        vmin, vmax = -np.max(np.abs([vmin, vmax])), np.max(np.abs([vmin, vmax]))
        if min_max is not None:
            vmin, vmax = min_max

        geopandas_states.drop(index=['HI', 'AK']).plot(column=column, ax=continental_ax, cmap=colormap, vmin=vmin, vmax=vmax)
        geopandas_states.loc[['AK']].plot(column=column, ax=alaska_ax, vmin=vmin, cmap=colormap, vmax=vmax)
        geopandas_states.loc[['HI']].plot(column=column, ax=hawaii_ax, vmin=vmin, cmap=colormap, vmax=vmax)

        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_array(geopandas_states[column].values)
        sm.set_clim(vmin, vmax)
        if ticks_margin is not None:
            ticks = list(np.arange(round(vmin, ticks_round), round(vmax, ticks_round)+ticks_margin, ticks_margin))
        else:
            ticks = None
        cb = fig.colorbar(sm, ax=continental_ax, fraction=fraction, pad=0,
                        aspect=aspect, ticks=ticks, orientation=cmap_orient)
        cb.outline.set_edgecolor('white')

        geopandas_states = geopandas_states.reset_index()

        geopandas_states[~geopandas_states['state'].isin(['AK', 'HI'])].apply(lambda x: continental_ax.annotate(text=x.state, xy=x.geometry.centroid.coords[0], ha='center', fontsize=fontsize),axis=1);
        geopandas_states[geopandas_states['state'] == 'AK'].apply(lambda x: alaska_ax.annotate(text=x.state, xy=x.geometry.centroid.coords[0], ha='center', fontsize=fontsize),axis=1);
        geopandas_states[geopandas_states['state'] == 'HI'].apply(lambda x: hawaii_ax.annotate(text=x.state, xy=x.geometry.centroid.coords[0], ha='center', fontsize=fontsize),axis=1);

        # remove ticks
        for ax in [continental_ax, alaska_ax, hawaii_ax]:
            ax.set_yticks([])
            ax.set_xticks([])
            for l in ['top', 'bottom', 'right', 'left']:
                ax.spines[l].set_visible(False)

        plt.box(False)
        plt.margins(x=0, y=0)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches='tight', pad_inches=0)

    def trim_white_space(self, im_path):
        # Open input image
        im = Image.open(im_path)

        # Get rid of existing black border by flood-filling with white from top-left corner
        ImageDraw.floodfill(im, xy=(0,0), value=(255,255,255), thresh=10)

        # Get bounding box of text and trim to it
        bbox = ImageOps.invert(im.convert('RGB')).getbbox()
        trimmed = im.crop(bbox)
        return trimmed

    def volcano_plot(self, df, x, y, colors, figsize=(8, 8),
                         y_axis_formatter=None, point_size_multiplexer=1,
                          save=None, paper_figure=None):
        colormap = LinearSegmentedColormap.from_list(
            name='test', 
            colors=colors
        )
        plt.figure(figsize=figsize)

        min_max = np.max(np.abs([df[x].values.min(),
                                df[x].values.max()]))

        fig = sns.scatterplot(data=df, x=x, y=y, linewidth=0,
                            c=df[x].values, s=df['tweet_counts'].values*point_size_multiplexer,
                            cmap=colormap, vmin=-min_max, vmax=min_max)

        for idx, row in df.iterrows():
            fig.text(row[x], row[y], row['state'], horizontalalignment='center',
                     verticalalignment='center', fontdict=dict(size=19, color='black'))

        plt.xlabel('ATV Score')
        plt.ylabel('$-log_{10}(P_{adj})$')

        if y_axis_formatter is not None:
            fig.yaxis.set_major_formatter(y_axis_formatter)

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches='tight')

    def scatter_pca(self, df, x, y, colors, figsize=(8, 8), point_size_multiplexer=1,
                          save=None, paper_figure=None):

        colormap = LinearSegmentedColormap.from_list(
            name='test', 
            colors=colors
        )

        plt.figure(figsize=figsize)

        min_max = np.max(np.abs([df['odd_ratios'].values.min(), df['odd_ratios'].values.max()]))
        fig = sns.scatterplot(data=df, x=x, y=y, c=df['odd_ratios'].values, s=df['tweet_counts'].values*point_size_multiplexer,
                             cmap=colormap, vmin=-min_max, vmax=min_max)
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')

        for idx, row in df.iterrows():
            fig.text(row[x], row[y], row['state'], horizontalalignment='center',
             verticalalignment='center', fontdict=dict(size=16, color='black'))

        sns.despine(offset=5, trim=True)

        fig.text(np.max(df[x]) + np.max(df[x])*0.13, self.mid_point(df[y].values), 'ATV Score',
         horizontalalignment='center', verticalalignment='center', rotation=90, fontdict=dict(size=18, color='black'))

        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_array(df['odd_ratios'].values)
        sm.set_clim(-min_max, min_max)
        fig.figure.colorbar(sm, pad=0.04)

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        if paper_figure is not None:
            plt.savefig(paper_figure, bbox_inches='tight')

        