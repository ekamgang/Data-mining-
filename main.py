from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import pandas as pd
import folium
from folium.plugins import HeatMap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


# only uncomment if you have the original dataset 

# batch = 1
# for chunk in pd.read_csv('Taxi_Trips.csv',chunksize=100000):
#     chunk.to_csv('taxi'+str(batch)+'.csv',index=False)
#     batch +=1

df = pd.read_csv('taxi5.csv')


time = df['Trip Start Timestamp'].str.slice(10,13) +" "+df['Trip Start Timestamp'].str.slice(20,22)
df.insert(loc=4,column='Time',value=time)


market_share = (df.groupby('Company')['Trip ID'].count() / df.groupby('Company')['Trip ID'].count().sum())*100
market_share = (market_share.loc[market_share < 1]).index
for i in range(0,len(market_share)):
    df.loc[df['Company'] == market_share[i],'Company'] = 'Others'


y = df.loc[df['Trip Total'] < 3.25].index
for i in range(0,len(y)):
    df.drop(y[i], inplace=True)

s = df.loc[(df['Trip Seconds'] == 0) & (df['Trip Miles'] == 0)].index
for i in range(0,len(s)):
    df.drop(s[i], inplace=True)


df.drop(["Pickup Centroid Location","Dropoff Centroid  Location"],inplace=True,axis=1)
df.drop(["Pickup Census Tract","Dropoff Census Tract"],inplace=True,axis=1)


y = df.loc[df['Trip Miles']  == 0]['Trip Seconds'] * 0.0036
df.loc[df['Trip Miles'] == 0,'Trip Miles'] = round(y,2)



df.insert(loc=11,column='Tip',value=df['Tips'] > 0)

payment = (df.groupby('Payment Type')['Trip ID'].count() / df.groupby('Payment Type')['Trip ID'].count().sum())*100
payment = (payment.loc[payment <.9]).index
for i in range(0,len(payment)):
    df.loc[df['Payment Type'] == payment[i],'Payment Type'] = 'Others'


sns.distplot(df['Trip Seconds']/60,bins=60,kde=False, hist_kws={'range':(0,60)},color='Red');
plt.xlabel("Trip Minutes")
plt.show()
plt.savefig('trip_seconds')


sns.distplot(df['Trip Total'],bins=100,kde=False, hist_kws={'range':(0,80)},color='Green');
plt.xlabel("Fare ")
plt.show()
plt.savefig('Trip_total')


sns.distplot(df['Trip Miles'],bins=60,kde=False, hist_kws={'range':(0,25)},color='Yellow');
plt.savefig("trip_miles")


s = df.groupby('Taxi ID')['Trip Total'].sum()
plt.hist(s,range=(100,2000),bins=50);
plt.show()
plt.savefig('salary.png')

pickup = df.groupby(['Pickup Centroid Latitude','Pickup Centroid Longitude']).count()
pickup = pickup['Trip ID'].reset_index().values.tolist()

def heatMap(default_location=[41.89915561, -87.62621053], default_zoom_start=10):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

base_map = heatMap()
HeatMap(data=pickup, radius=8, max_zoom=13).add_to(base_map)
base_map


plt.axis('off')
tip = df.groupby('Tip')['Trip ID'].count()
plt.pie(tip,labels=tip.index,autopct='%1.1f%%',colors=['Red','Green']);
plt.show()
plt.savefig('tips.png')

x = df[['Pickup Centroid Latitude','Pickup Centroid Longitude','Tips']].values
latlon = x[:1200]
mapit = folium.Map(location=[41.89915561, -87.62621053], zoom_start=12,disable_3d=True,tiles='Stamen Terrain')
for i in latlon:
        if i[2] <= 0:
            folium.Marker(icon=folium.Icon(color='red'),location=[ i[0], i[1] ], fill_color='#dedb43', radius=8 ).add_to( mapit )
        else:
            folium.Marker(icon=folium.Icon(color='green'),location=[ i[0], i[1] ], fill_color='#dedb43', radius=8 ).add_to( mapit )



time = (df.groupby([df['Time'].str.slice(4,6),df['Time'].str.slice(1,3)]).count())
label =  time.index.get_level_values(1) +" "+time.index.get_level_values(0)
x = time['Trip ID']
plt.figure(figsize=(20,5))
plt.bar(label,x,width=.9,);
plt.show()
plt.savefig('time.png')


explode = (.09, .09, .09, .09,.2,.2,.05,.09)
companies = df.groupby('Company').count()['Trip ID']
plt.pie(companies,labels=companies.index,explode=explode,shadow=True,autopct='%1.1f%%');
plt.show()
plt.savefig('companies.png')


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True);
plt.show()
plt.savefig('heatmap.png');

df.drop(df.loc[(df['Trip Seconds'] == 0) | (df['Trip Miles'] == 0)].index,inplace=True)
df = df.fillna(0)

X = df[['Trip Seconds','Pickup Community Area','Dropoff Community Area']].values
y = df['Fare'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_value:', metrics.r2_score(y_test, y_pred)*100)


p = pd.DataFrame()
p['y_test'] = y_test
p['y_pred'] = y_pred
df1 = p.head(40)
df1.plot(kind='bar',figsize=(20,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

df.to_csv("clean_data.csv")