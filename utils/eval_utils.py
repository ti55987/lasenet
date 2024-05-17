import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score

def get_all_agents_rmse(y_true, y_pred, squared: bool=False):
  all_mse = []
  for a, pred in enumerate(y_pred):
    mse = mean_squared_error(y_true[a], pred, squared=squared)
    all_mse.append(mse)

  return all_mse

def get_all_agents_scores(y_true, y_pred, metrics_func=balanced_accuracy_score):
  return [metrics_func(y_true[a], pred) for a, pred in enumerate(y_pred)]


def plot_time_series(axes, y_true, y_pred, max_trial):
  for idx, ax in enumerate(axes.flat):
    result = pd.DataFrame({'true_label': y_true[idx][:max_trial], 'pred_label': y_pred[idx][:max_trial]})
    markers = {"true_label": "v", "pred_label": "."}
    show_legend = True if idx == 0 else False
    plot_ = sns.lineplot(result, markers=markers, ax=ax, legend=show_legend) #lineplot
    if show_legend:
      sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(1, 1.1), ncol=2, title=None, frameon=False,
      )

    ax.set_xlabel('trial #')
    ax.set_title(f'agent {idx}', fontsize=10)
    for ind, label in enumerate(plot_.get_yticklabels()):
      if ind == 0 or ind == 1.0:  # every 10th label is kept
          label.set_visible(True)
      else:
          label.set_visible(False)  
  return axes