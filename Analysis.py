import pandas as pd


# Ensuring consistency in data and seeing if any NULL values are present

df = pd.read_csv("hand_data.csv");
print(df.shape);
print(df.info());
print(df.isnull().sum());

# Ensure balanced gesture classes. If not, note for augmentation or sampling.

print(df['label'].value_counts())
df['label'].value_counts().plot(kind='bar', title='Gesture Class Distribution')

#fist -- 131 