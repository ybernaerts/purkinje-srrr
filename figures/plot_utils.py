import matplotlib.pyplot as plt
import numpy as np

def adjust_spines(ax, spines, spine_pos=5, color='k', linewidth=None, smart_bounds=True):
    """Convenience function to adjust plot axis spines."""

    # If no spines are given, make everything invisible
    if spines is None:
        ax.axis('off')
        return

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', spine_pos))  # outward by x points
            #spine.set_smart_bounds(smart_bounds)
            spine.set_color(color)
            if linewidth is not None:
                spine.set_linewidth = linewidth
        else:
            spine.set_visible(False)  # make spine invisible
            # spine.set_color('none')  # this will interfere w constrained plot layout

    # Turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # No visible yaxis ticks and tick labels
        # ax.yaxis.set_visible(False)  # hides whole axis, incl. ax label
        # ax.yaxis.set_ticks([])  # for shared axes, this would delete ticks for all
        plt.setp(ax.get_yticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.yaxis.get_ticklines(), color='none')  # changes tick color to none
        # ax.tick_params(axis='y', colors='none')  # (same as above) changes tick color to none

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # No visible xaxis ticks and tick labels
        # ax.xaxis.set_visible(False)  # hides whole axis, incl. ax label
        # ax.xaxis.set_ticks([])  # for shared axes, this would  delete ticks for all
        plt.setp(ax.get_xticklabels(), visible=False)  # hides ticklabels but not ticks
        plt.setp(ax.xaxis.get_ticklines(), color='none')  # changes tick color to none


def adjustlabels(fig, ax, labels, max_iter=1000, eps=0.01, delta=0.1):
    N = len(labels)
    widths = np.zeros(N)
    heights = np.zeros(N)
    centers = np.zeros((N, 2))
    for i,l in enumerate(labels):
        bb = l.get_window_extent(renderer=fig.canvas.get_renderer())
        bb = bb.transformed(ax.transData.inverted())
        widths[i] = bb.width
        heights[i] = bb.height
        centers[i] = (bb.min + bb.max)/2

    for i in range(max_iter):
        stop = True
        for a in range(N):
            for b in range(N):
                if ((a!=b) and
                    (np.abs(centers[a,0]-centers[b,0]) < (widths[a]+widths[b])/2 + delta) and
                    (np.abs(centers[a,1]-centers[b,1]) < (heights[a]+heights[b])/2 +  delta)):
                    
                    d = centers[a] - centers[b]
                    centers[a] += d * eps
                    centers[b] -= d * eps
                    labels[a].set_position(centers[a])
                    labels[b].set_position(centers[b])
                    stop = False
        if stop:
            break