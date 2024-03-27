import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('happyscore_income.csv')

selected_countries_1 = data[data['avg_satisfaction'] < 3]
selected_countries_2 = data[data['avg_satisfaction'] > 8]
happy = data['happyScore']
gdp = data['GDP']

def label_countries(countries):
    for k, row in countries.iterrows():
        plt.text(
            x = row['happyScore'],
            y = row['GDP'],
            s = row['country']
            )
label_countries(selected_countries_1)
label_countries(selected_countries_2)

plt.ylabel('GDP')
plt.xlabel('happy score')
plt.scatter(happy, gdp)
plt.show()
