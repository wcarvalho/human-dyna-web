import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import DIRECTORY
from simulations import wandb_result_plots
import matplotlib.pyplot as plt

model_to_group = {
    'ql': 'ql-final-5',
    'ql-sf': 'ql-sf-final-5',
    'dyna': 'dyna-final-5',
    'preplay': 'preplay-final-5',
}
df = wandb_result_plots.get_metric_data_by_group(
    model_to_group=model_to_group,
    debug=False,
)

directory = f"{DIRECTORY}/craftax_AI_results"
def save_figure(fig, filename,):
  os.makedirs(directory, exist_ok=True)
  plt.savefig(os.path.join(directory, f"{filename}.png"), bbox_inches='tight', dpi=300)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches='tight', dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


# First plot train and eval together
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
wandb_result_plots.plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=False,
    evaluation=False,
    ax=ax[0],
    ylim=(1.5, 7),
)
wandb_result_plots.plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=True,
    evaluation=True,
    ax=ax[1],
    ylim=(1.5, 7),
)
save_figure(fig, "train_eval")


# then plot eval by itself
fig, ax = wandb_result_plots.plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=True,
    evaluation=True,
)
save_figure(fig, "eval")


# then plot per achievement for all
for n in [8, 16, 32, 64, 128, 256, 512]:
    fig, ax = wandb_result_plots.plot_achievement_bars(df, n=n, figsize=(12, 5), show_legend=True)
    save_figure(fig, f"achievement_bars_{n}")

options = [8, 16, 32, 64, 128, 256, 512]
nrows = (len(options) + 1) // 2  # Ceiling division to handle odd number of plots
fig, ax = plt.subplots(nrows, 2, figsize=(20, 5*nrows))
ax = ax.flatten()  # Flatten to make indexing easier
for i, n in enumerate(options):
    wandb_result_plots.plot_achievement_bars(
        df, n=n, show_legend=i==0, fig=fig, ax=ax[i])
# Hide any empty subplots
for j in range(len(options), len(ax)):
    ax[j].set_visible(False)
save_figure(fig, "achievement_bars_all")
