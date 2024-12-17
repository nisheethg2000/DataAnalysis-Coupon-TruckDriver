### Required Assignment 5.1: Will the Customer Accept the Coupon?

**Context**

Imagine driving through town and a coupon is delivered to your cell phone for a restaurant near where you are driving. Would you accept that coupon and take a short detour to the restaurant? Would you accept the coupon but use it on a subsequent trip? Would you ignore the coupon entirely? What if the coupon was for a bar instead of a restaurant? What about a coffee house? Would you accept a bar coupon with a minor passenger in the car? What about if it was just you and your partner in the car? Would weather impact the rate of acceptance? What about the time of day?

Obviously, proximity to the business is a factor on whether the coupon is delivered to the driver or not, but what are the factors that determine whether a driver accepts the coupon once it is delivered to them? How would you determine whether a driver is likely to accept a coupon?

**Overview**

The goal of this project is to use what you know about visualizations and probability distributions to distinguish between customers who accepted a driving coupon versus those that did not.

**Data**

This data comes to us from the UCI Machine Learning repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. Answers that the user will drive there ‘right away’ or ‘later before the coupon expires’ are labeled as ‘Y = 1’ and answers ‘no, I do not want the coupon’ are labeled as ‘Y = 0’.  There are five different types of coupons -- less expensive restaurants (under \$20), coffee houses, carry out & take away, bar, and more expensive restaurants (\$20 - $50).

**Deliverables**

Your final product should be a brief report that highlights the differences between customers who did and did not accept the coupons.  To explore the data you will utilize your knowledge of plotting, statistical summaries, and visualization using Python. You will publish your findings in a public facing github repository as your first portfolio piece.





### Data Description
Keep in mind that these values mentioned below are average values.

The attributes of this data set include:
1. User attributes
    -  Gender: male, female
    -  Age: below 21, 21 to 25, 26 to 30, etc.
    -  Marital Status: single, married partner, unmarried partner, or widowed
    -  Number of children: 0, 1, or more than 1
    -  Education: high school, bachelors degree, associates degree, or graduate degree
    -  Occupation: architecture & engineering, business & financial, etc.
    -  Annual income: less than \\$12500, \\$12500 - \\$24999, \\$25000 - \\$37499, etc.
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she buys takeaway food: 0, less than 1, 1 to 3, 4 to 8 or greater
    than 8
    -  Number of times that he/she goes to a coffee house: 0, less than 1, 1 to 3, 4 to 8 or
    greater than 8
    -  Number of times that he/she eats at a restaurant with average expense less than \\$20 per
    person: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    

2. Contextual attributes
    - Driving destination: home, work, or no urgent destination
    - Location of user, coupon and destination: we provide a map to show the geographical
    location of the user, destination, and the venue, and we mark the distance between each
    two places with time of driving. The user can see whether the venue is in the same
    direction as the destination.
    - Weather: sunny, rainy, or snowy
    - Temperature: 30F, 55F, or 80F
    - Time: 10AM, 2PM, or 6PM
    - Passenger: alone, partner, kid(s), or friend(s)


3. Coupon attributes
    - time before it expires: 2 hours or one day


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import warnings

```


```python
warnings.filterwarnings("ignore")
```

### Problems

Use the prompts below to get started with your data analysis.  

1. Read in the `coupons.csv` file.





```python
data = pd.read_csv('data/coupons.csv')
```


```python
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>destination</th>
      <th>passanger</th>
      <th>weather</th>
      <th>temperature</th>
      <th>time</th>
      <th>coupon</th>
      <th>expiration</th>
      <th>gender</th>
      <th>age</th>
      <th>maritalStatus</th>
      <th>...</th>
      <th>CoffeeHouse</th>
      <th>CarryAway</th>
      <th>RestaurantLessThan20</th>
      <th>Restaurant20To50</th>
      <th>toCoupon_GEQ5min</th>
      <th>toCoupon_GEQ15min</th>
      <th>toCoupon_GEQ25min</th>
      <th>direction_same</th>
      <th>direction_opp</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No Urgent Place</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>55</td>
      <td>2PM</td>
      <td>Restaurant(&lt;20)</td>
      <td>1d</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Coffee House</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Carry out &amp; Take away</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>2PM</td>
      <td>Coffee House</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>2PM</td>
      <td>Coffee House</td>
      <td>1d</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



2. Investigate the dataset for missing or problematic data.


```python
#data['coupon'].value_counts() #slecting unique values in coupon column

#print(data['coupon'].value_counts())
#print(data['RestaurantLessThan20'].value_counts())
#print(data['Restaurant20To50'].value_counts())
print("Duplicate rows:",data.duplicated().sum()) #check for duplicate rows
print("Null Rows :", data.isnull().all(axis=1).sum()) #check for rows with all null values
print("Total Rows:",data.shape[0])
#Below for loop is to count alpanumeric values in each column for cleanup 
#for col in data.columns:
 #   if (data[col].dtype == 'object'):
  #      print(col,':',(data[col].apply(lambda p: ((not isinstance(p, float)) and (not p.isalnum()))).sum()))
   #     if((data[col].apply(lambda p: ((not isinstance(p, float)) and (not p.isalnum()))).sum()) > 0):
    #        print(data[col].value_counts(), end=" ") 
        #data['col'].str.replace(r'[^a-zA-Z0-9]', '_', regex=True)

#print(data)
```

    Duplicate rows: 74
    Null Rows : 0
    Total Rows: 12684
    

3. Decide what to do about your missing data -- drop, replace, other...


```python
#the car column has very few values which would not contribute much so dropping it
#data = data.drop('car', axis=1) 
#converting values of Y to make more readable
data['Y'] = data['Y'].astype(str)
data['Y'] = data['Y'].replace([re.escape(r'1'),re.escape(r'0')],['Accepted','NotAccepted'],regex=True) 

#dropping duplicates and keeping only 1 records
data = data.drop_duplicates(keep='first')

```


```python
#data = data.apply(lambda x: x.str.replace(' ', '') if x.dtype == "object" else x) #remove spaces from data
#data = data.apply(lambda x: x.str.replace('~', '-') if x.dtype == "object" else x) #remove spaces from data
#data = data.apply(lambda x: x.replace([re.escape(r'('),re.escape(r')')],['',''],regex=True) if x.dtype == "object" else x) #remove paranthesis from data
#data = data.apply(lambda x: x.replace([re.escape(r'&'),re.escape(r'<')],['And','LT'],regex=True) if x.dtype == "object" else x) #make the data mroe readable by translating some special characters
```


```python
data['coupon'].value_counts()
#data['coupon'] = data['coupon'].replace(re.escape(r'-'),'To',regex=True) #replacing special characters in coupon column
```




    coupon
    Coffee House             3989
    Restaurant(<20)          2779
    Carry out & Take away    2344
    Bar                      2010
    Restaurant(20-50)        1488
    Name: count, dtype: int64




```python
data['education'].value_counts()
#data['education'] = data['education'].replace(re.escape(r'-n'),'N',regex=True) #replacing special characters in coupon column
```




    education
    Some college - no degree                  4325
    Bachelors degree                          4323
    Graduate degree (Masters or Doctorate)    1827
    Associates degree                         1148
    High School Graduate                       899
    Some High School                            88
    Name: count, dtype: int64




```python
data['income'].value_counts()
#data['income'] = data['income'].replace([re.escape(r'orMore'),re.escape('Lessthan')],['OM','LT'],regex=True) 
```




    income
    $25000 - $37499     2006
    $12500 - $24999     1825
    $37500 - $49999     1795
    $100000 or More     1717
    $50000 - $62499     1655
    Less than $12500    1034
    $87500 - $99999      879
    $75000 - $87499      856
    $62500 - $74999      843
    Name: count, dtype: int64




```python
data['Bar'].value_counts()
#data['Bar'] = data['Bar'].replace(r'~','-', regex=True) #replace ~ with - to depict range for consistency 
```




    Bar
    never    5178
    less1    3438
    1~3      2468
    4~8      1071
    gt8       348
    Name: count, dtype: int64




```python
#inspect to see if there are columns with any special characters other then '-' which are using for range
#data['CoffeeHouse'].value_counts()
#data['CarryAway'].value_counts()
#data['Restaurant20To50'].value_counts()
#data['RestaurantLessThan20'].value_counts()
#data['Bar'].value_counts()
```

4. What proportion of the total observations chose to accept the coupon?




```python

#print(data['Y'].value_counts(normalize=True))
#plt.plot(data['Y'].value_counts(normalize=True),x='Y')
totalObs = data['Y'].value_counts(normalize=True)
print(totalObs)
#may be show a pie chart too


```

    Y
    Accepted       0.567565
    NotAccepted    0.432435
    Name: proportion, dtype: float64
    

5. Use a bar plot to visualize the `coupon` column.


```python
sns.displot(data,x='coupon',hue='Y')
plt.xticks(rotation=90)
plt.title('Distribution of Coupon Column')
```




    Text(0.5, 1.0, 'Distribution of Coupon Column')




    
![png](prompt_files/prompt_21_1.png)
    


6. Use a histogram to visualize the temperature column.


```python
sns.histplot(data=data, x='temperature', hue='Y',stat='probability',common_norm=True)
#data['temperature'].plot.hist(hue='temperature')
plt.xlabel('Temperature in Celsius')
```




    Text(0.5, 0, 'Temperature in Celsius')




    
![png](prompt_files/prompt_23_1.png)
    


Visualizing Time column with Accepted or not Accepted coupons


```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="time", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_25_0.png)
    


Visualizing expiration column with accepted or not accepted coupons


```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="expiration", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_27_0.png)
    


Visualizing weather column with accepted or not accepted coupons


```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="weather", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()

```


    
![png](prompt_files/prompt_29_0.png)
    


Visualization Destination label with respect Accepted/NotAccepted coupons


```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="destination", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()

```


    
![png](prompt_files/prompt_31_0.png)
    


**Investigating the Bar Coupons**

Now, we will lead you through an exploration of just the bar related coupons.  

1. Create a new `DataFrame` that contains just the bar coupons.



```python
BarCouponData = data[data['coupon'] == 'Bar']
print(type(BarCouponData))
```

    <class 'pandas.core.frame.DataFrame'>
    

2. What proportion of bar coupons were accepted?



```python
BarCouponData['Y'].value_counts().plot.pie(autopct='%1.1f%%') 
#BarCouponData.plot.pie(y='Y')
```




    <Axes: ylabel='count'>




    
![png](prompt_files/prompt_35_1.png)
    


3. Compare the acceptance rate between those who went to a bar 3 or fewer times a month to those who went more.



```python
ct = pd.crosstab(BarCouponData['Bar'].str.contains('1-3|never|less1'), BarCouponData['Y'], normalize='index').plot.pie(subplots=True,autopct='%1.1f%%') 

```


    
![png](prompt_files/prompt_37_0.png)
    


4. Compare the acceptance rate between drivers who go to a bar more than once a month and are over the age of 25 to the all others.  Is there a difference?



```python
tmp = BarCouponData.query('Y == "Accepted"').groupby((BarCouponData['Bar'].str.contains('never|less1')== False)
                                                     &(BarCouponData['age'].str.contains('26|31|50plus|36|41|46')))[['Y']].value_counts()
tmp.plot.bar(color=['blue','red']) 
plt.xlabel('Coupon Accepted')
plt.xticks([0, 1], [ 'LTOnceMth&LT25', 'MTOnceMth&GT25'])
plt.ylabel('proportion')
plt.title('Bar Coupon Distribution')
#sns.scatterplot(data=data, x='temperature', hue='temperature')
```




    Text(0.5, 1.0, 'Bar Coupon Distribution')




    
![png](prompt_files/prompt_39_1.png)
    


5. Use the same process to compare the acceptance rate between drivers who go to bars more than once a month and had passengers that were not a kid and had occupations other than farming, fishing, or forestry.



```python
tmp = BarCouponData.query('Y == "Accepted"').groupby((BarCouponData['Bar'].str.contains('never|less1')== False)
                                                     &(BarCouponData['passanger'].str.contains('kids')==False)
                                                     &(BarCouponData['occupation'].str.contains('FarmingFishingAndForestry')==False))['Y'].value_counts()
tmp.plot.bar(color=['blue','red']) 
plt.xlabel('Coupon Accepted Status')
plt.xticks([0, 1], ['LTOnceMth&WKids&FFF', 'MTOnceMth&WTKids&NotFFF'])
plt.ylabel('proportion')
plt.title('Bar Coupon Vs Passenger Wt Kids & Farmers')
```




    Text(0.5, 1.0, 'Bar Coupon Vs Passenger Wt Kids & Farmers')




    
![png](prompt_files/prompt_41_1.png)
    


6. Compare the acceptance rates between those drivers who:

- go to bars more than once a month, had passengers that were not a kid, and were not widowed *OR*
- go to bars more than once a month and are under the age of 30 *OR*
- go to cheap restaurants more than 4 times a month and income is less than 50K.




```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
tmp1 = BarCouponData.query('Y == "Accepted"').groupby((BarCouponData['Bar'].str.contains('never|less1')== False)
                                                      &(BarCouponData['passanger'].str.contains('kids')==False)
                   &(BarCouponData['maritalStatus'].str.contains('Widowed')==False))['Y'].value_counts()
tmp2 = BarCouponData.query('Y == "Accepted"').groupby((BarCouponData['Bar'].str.contains('never|less1')== False)
                   &(BarCouponData['age'].str.contains('below21|21|26')))['Y'].value_counts()
tmp3 = BarCouponData.query('Y == "Accepted"').groupby((BarCouponData['RestaurantLessThan20'].str.contains('4-8|gt8'))
                   &(BarCouponData['income'].str.contains('LT$12500|$12500-$24999|$25000-$37499|$37500-$49999|')))['Y'].value_counts(normalize=True)

tmp1.plot.bar(ax=axes[0],color=['purple','orange'])

tmp2.plot.bar(ax=axes[1],color=['blue','red']) 

tmp3.plot.bar(ax=axes[2],color=['green','yellow']) 

```




    <Axes: xlabel='None,Y'>




    
![png](prompt_files/prompt_43_1.png)
    


7.  Based on these observations, what do you hypothesize about drivers who accepted the bar coupons?

### Independent Investigation

Using the bar coupon example as motivation, you are to explore one of the other coupon groups and try to determine the characteristics of passengers who accept the coupons.  


```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="age", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "gender")
plt.legend()
# Show the plot
plt.show()

```


    
![png](prompt_files/prompt_46_0.png)
    



```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="weather", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "temperature")
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_47_0.png)
    



```python

```


```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="Bar", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_49_0.png)
    



```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="expiration", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_50_0.png)
    



```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="destination", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_51_0.png)
    



```python
# Create a FacetGrid
g = sns.FacetGrid(data, col="weather", hue="Y")

# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_52_0.png)
    



```python
# Create a FacetGrid
g = sns.FacetGrid(data, col='CarryAway', hue="Y")
# Map the countplot to each facet
g.map(sns.countplot, "coupon")
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend()
# Show the plot
plt.show()
```


    
![png](prompt_files/prompt_53_0.png)
    

