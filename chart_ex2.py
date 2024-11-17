import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('car_data.csv')
sns.boxplot(y='Color', x='Price ($)', data=data, orient='h', palette='bright').set_title('Color vs Price')
plt.xticks(rotation=30)
plt.show()
sns.scatterplot(y='Annual Income', x='Price ($)',  hue='Company', data=data).set_title('Income vs Price')
plt.legend(bbox_to_anchor=(1.01, 1), fontsize=6)
plt.xticks(rotation=30)
plt.show()
sns.catplot(x='Transmission', col='Color', hue='Gender', kind='count', data=data);
plt.show()

