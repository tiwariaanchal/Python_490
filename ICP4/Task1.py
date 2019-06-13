import pandas as pd
train_df = pd.read_csv('./train.csv')
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
