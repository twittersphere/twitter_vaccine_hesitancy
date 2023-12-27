import os
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_icons():
    twitter_icon = cv2.imread("figures/icons/twitter_bird_icon.png",
                              cv2.IMREAD_UNCHANGED)
    user_icon = cv2.imread("figures/icons/avatar_icon.png",
                           cv2.IMREAD_UNCHANGED)
    location_icon = cv2.imread("figures/icons/location_icon.png",
                               cv2.IMREAD_UNCHANGED)

    return twitter_icon, user_icon, location_icon

def plot_figure(eda_df, all_figures_path, twitter_icon, user_icon, location_icon):
    sns.set_context('poster', font_scale=1)

    fig = plt.figure(figsize=(8, 8), dpi=600)
    ax = sns.barplot(data=eda_df, x='Counts', y='Labels', orient='h')

    for idx, p in enumerate(ax.patches):
        ax.annotate(f"{'':12s}{eda_df['Counts'].values[idx]:,}\n{'':12s}"\
                    f"{eda_df['Labels'].values[idx]}",
                    xy=(p.get_width(), p.get_y()+p.get_height()/2),
                    xytext=(5, 0), textcoords='offset points', ha="left",
                    va="center")

    plt.xscale('log')
    ax.axes.yaxis.set_visible(False)

    plt.figimage(twitter_icon, 3800, 3450)
    plt.figimage(user_icon, 2350, 2200)
    plt.figimage(location_icon, 400, 1000)

    sns.despine(bottom=True)
    plt.savefig(f"{all_figures_path}/tweet_user_geoloc_counts.tiff",
                format='tiff', pil_kwargs={"compression": "tiff_lzw"},
                bbox_inches='tight')

def main():
    all_figures_path = "figures/all_figures"
    os.makedirs(all_figures_path, exist_ok=True)

    eda_df = pd.read_parquet("data/processed/dataframes/eda_df.parquet")
    twitter_icon, user_icon, location_icon = load_icons(all_figures_path)
    plot_figure(eda_df, all_figures_path, twitter_icon,
                user_icon, location_icon)

if __name__ == "__main__":
    main()