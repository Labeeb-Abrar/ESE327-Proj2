import pandas as pd

# Sample DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Value': [10, 20, 30, 40, 50, 60]}

df = pd.DataFrame(data)

# Grouping by 'Category'
grouped = df.groupby('Category')

# Applying a function (e.g., calculating the mean for each group)
result = grouped.mean()

print(result)