import os
import pdb
import numpy as np
from scipy import spatial
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt

src_folder = "Final_dataset/templates/"
genuine_scores_xlsx = "Final_dataset/matching_score/intra_subject_score.xlsx"
genuine_scores = "Final_dataset/matching_score/intra_subject_score"
figure_save_folder = "Final_dataset/figures/"
dict_makeup = {
    'final_dataset_without_makeup': '_00',
    'eyebrows_thin': '_01',
    'eyebrows_thick': '_02',
    'lipstick_deepshade_thick': '_03',
    'lipstick_lightshade_normal': '_04',
    'eyelashes_normal': '_05',
    'eyelashes_full': '_06',
    'eyelashes_dramatic': '_07',
    'eyeliner_normal': '_08',
    'eyeliner_thick': '_09',
    'full_makeup': '_10',

 }

def draw_box_plot(df_final):
    makeup_component_alias = list(dict_makeup.values())
    makeup_component = list(dict_makeup.keys())
    fig, ax = plt.subplots()
    fig.canvas.draw()
    green_diamond = dict(markerfacecolor='g', marker='D')
    box = df_final.boxplot(column=makeup_component, grid=False, labels=makeup_component_alias, manage_ticks=True, flierprops=green_diamond, patch_artist=True)

    #pdb.set_trace()
    """
    colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red', 'yellow', 'cyan', 'orange', 'white', 'black']
    for row_key, (ax, row) in box.items():
        ax.set_xlabel('')
        for i, box in enumerate(row['boxes']):
            box.set_facecolor(colors[i])

    
     ax.annotate(str(dict_makeup), xy=(1, 0), xycoords='axes fraction',
                size=7, ha='center', va='bottom',
                bbox=dict(boxstyle='round', fc='w'))
    """

    ax.legend(makeup_component, loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), prop={'size': 7})
    ax.set_xticklabels(makeup_component_alias)
    plt.savefig(figure_save_folder + 'boxplot.png')

def plot_histogram(df_final):

    df_final['eyebrows_thin'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder+'eyebrows_thin.png')

    df_final['eyebrows_thick'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'eyebrows_thick.png')

    df_final['lipstick_deepshade_thick'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'lipstick_deepshade_thick.png')

    df_final['lipstick_lightshade_normal'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'lipstick_lightshade_normal.png')

    df_final['eyelashes_normal'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'eyelashes_normal.png')

    df_final['eyelashes_full'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'eyelashes_full.png')

    df_final['eyelashes_dramatic'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'eyelashes_dramatic.png')

    df_final['eyeliner_normal'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'eyeliner_normal.png')

    df_final['eyeliner_thick'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'eyeliner_thick.png')

    df_final['full_makeup'].plot.hist(bins=12, alpha=0.5)
    plt.savefig(figure_save_folder + 'full_makeup.png')

def generate_genuine_match_score(no_makeup_template, makeup_template):
    x = np.load(no_makeup_template)
    y = np.load(makeup_template)
    score = (1 - spatial.distance.cosine(x, y)) * 100
    return score

def main():

    for root, dirs, files in os.walk(src_folder):
        no_makeup_src_dir = [d for d in dirs if 'without_makeup' in d]
        subject_ids = sorted([d[:-7] for d in os.listdir(root+no_makeup_src_dir[0])])
        df = pd.DataFrame(columns=dict_makeup.keys(), index=subject_ids)
        final_score_dict = df.to_dict('index')
        for i in subject_ids:
            no_makeup_template_path = root+no_makeup_src_dir[0]+"/"+i+"_00.npy"
            for makeup in dirs:
                makeup_template_path = root+makeup+"/"+i+dict_makeup[makeup]+".npy"
                score = generate_genuine_match_score(no_makeup_template_path, makeup_template_path)
                final_score_dict[str(i)][makeup] = score
        df_final = pd.DataFrame.from_dict(final_score_dict, orient='index')
        df_final.to_csv(genuine_scores, sep=',')
        df_final.to_excel(genuine_scores_xlsx, engine='xlsxwriter')
        draw_box_plot(df_final)
        plot_histogram(df_final)
        break


if __name__ == '__main__':
    main()