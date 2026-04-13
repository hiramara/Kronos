import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, title=None):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    if title:
        fig.suptitle(title, fontsize=13)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained('/home/csc/huggingface/Kronos-Tokenizer-base/')
model = Kronos.from_pretrained("/home/csc/huggingface/Kronos-base/")

# 2. Instantiate Predictor
# Note: using cpu here since I don't always have a GPU available on my dev machine
# Reduced max_context from 512 to 400 to match the lookback window length exactly,
# avoiding unnecessary padding overhead during inference.
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=400)

# 3. Prepare Data
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

# Using 3 samples instead of 5 to keep batch inference quick during testing
num_samples = 3

# Fixed off-by-one: use i*lookback instead of i*400 so the step matches the
# actual lookback window length, making it easy to change lookback later.
dfs = []
xtsp = []
ytsp = []
for i in range(num_samples):
    idf = df.loc[(i*lookback):(i*lookback+lookback-1), ['open', 'high', 'low', 'close', 'volume', 'amount']]
    i_x_timestamp = df.loc[(i*lookback):(i*lookback+lookback-1), 'timestamps']
    i_y_timestamp = df.loc[(i*lookback+lookback):(i*lookback+lookback+pred_len-1), 'timestamps']

    dfs.append(idf)
    xtsp.append(i_x_timestamp)
    ytsp.append(i_y_timestamp)

pred_df = predictor.predict_batch(
    df_list=dfs,
    x_timestamp_list=xtsp,
    y_timestamp_list=ytsp,
    pred_len=pred_len,
)

# 4. Visualize results for each sample
# Iterate over the list of prediction DataFrames and plot each one against
# the corresponding input window so it's easy to eyeball forecast quality.
for i, (input_df, prediction_df) in enumerate(zip(dfs, pred_df)):
    plot_prediction(input_df, prediction_df, title=f"Sample {i + 1} — 5min XSHG 600977")
