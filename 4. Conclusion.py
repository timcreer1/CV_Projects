import pandas as pd
import matplotlib.pyplot as plt

# Load results with model category
df_conv = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Convolutional Based/model_results_convolutional_based.csv')
df_conv['Model_Category'] = 'Convolutional Based'

df_dl = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Deep Learning Based/model_results_convolutional_based.csv')
df_dl['Model_Category'] = 'Deep Learning Based'

df_fb = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Feature-based/model_results.csv')
df_fb['Model_Category'] = 'Feature-based'

df_ib = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Interval Based/model_results_interval_based.csv')
df_ib['Model_Category'] = 'Interval Based'

# Concatenate dataframes along rows
df_final = pd.concat([df_conv, df_dl, df_fb, df_ib], ignore_index=True)
pd.set_option('display.max_columns', None)

# Print the combined dataframe
print(df_final)

# Define a color palette
colors = plt.cm.tab10(range(len(df_final['Model_Category'].unique())))

# Plot each model category in a separate subplot
categories = df_final['Model_Category'].unique()
fig, axs = plt.subplots(1, len(categories), figsize=(20, 6), sharey=True)

for idx, category in enumerate(categories):
    category_data = df_final[df_final['Model_Category'] == category]
    bars = axs[idx].bar(category_data['Model'], category_data['Accuracy'], color=colors[idx])
    axs[idx].set_title(category)
    axs[idx].set_xlabel('Model')
    axs[idx].set_xticks(range(len(category_data['Model'])))
    axs[idx].set_xticklabels(category_data['Model'], rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        axs[idx].text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha='center')

axs[0].set_ylabel('Accuracy')
plt.tight_layout()
plt.show()
