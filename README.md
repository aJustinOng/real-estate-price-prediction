# Real Estate Price Prediction ![](./assets/img/house-icon.svg) (README.md in progress)
##### Justin Ong, 20th January 2025

### Skills:
`Python | NumPy | Pandas | Matplotlib | scikit-learn | HTML | CSS | JavaScript`

### Tools:
`Jupyter Notebook | VS Code | PyCharm | Flask | Postman`

##### [See my other projects!](https://github.com/aJustinOng)

---

## Overview

This project is based on CodeBasic's [Real Estate Price Prediction](https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg) project.

In this data science project, I cleaned and visualized a real estate dataset from Kaggle and used it to build a model with scikit-learn using linear regression. Next, I created a Python Flask server that can use the model to run HTTP requests, which I tested using Postman. Lastly, I made a website using HTML, CSS, and JavaScript with a user-friendly UI, where the user can enter their desired house area (square feet), number of bedrooms and bathrooms, and state to get a predicted price.

The model building section covers a majority of data science concepts like data cleaning, outlier removal, feature engineering, dimensionality reduction, one hot encoding, and K-Fold cross-validation. This README.md is a complete documentation of the project.

<img src="/assets/img/website-ui-1.png" width="100%"/>

## Table of contents:
1. [Importing Libraries and Data Loading](#1-importing-libraries-and-data-loading)
2. [Data Cleaning: Outlier Removal and Feature Engineering](#2-data-cleaning-outlier-removal-and-feature-engineering)
3. [One Hot Encoding Using Pandas](#3-one-hot-encoding-using-pandas)
4. [Model Building Using Scikit-Learn](#4-model-building-using-scikit-learn)
5. [Creating a Python Flask Server](#5-creating-a-python-flask-server)
6. [Creating a User-Friendly Website](#6-creating-a-user-friendly-website)
7. [Summary](#summary)

---

## 1. Importing Libraries and Data Loading

I found a [kaggle dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) that contained 2.2M+ real estate listings in U.S. states (and territories) with suitable values I could use to train a prediction model, such as price, location, number of bedrooms, number of bathrooms, and area in square feet. The data was collected from Realtor, a popular real estate listing website in the United States. It is a CSV file with the following columns:

### realtor-data.csv (2,226,382 entries)
- brokered_by (categorically encoded agency/broker)
- status (housing status, either ready for sale or ready to build)
- price (housing price, either the current listing price or recently sold price)
- bed (number of bedrooms)
- bath (number of bathrooms)
- acre_lot (property/land size in acres)
- street (categorically encoded street address)
- city (city name)
- state (state name)
- zip_code (postal code of the area)
- house_size (house area/size/living space in square feet)
- prev_sold_date (last sold date)

First, I imported the necessary libraries and set certain matplotlib settings on a new Jupyter Notebook:

```
import pandas as pd  
import numpy as np  
from matplotlib import pyplot as plt  
%matplotlib inline  
import matplotlib  
matplotlib.rcParams['figure.figsize'] = (20, 10)
```

I read the dataset CSV file into the notebook using Pandas:

```
df1 = pd.read_csv('realtor-data.csv')
df1.head()
```

<img src="/assets/img/jupyter-output-1.png" width="100%"/>

And validated the dataset size (2,226,382 rows, 12 columns):

```
df1.shape
```

### Drop Unnecessary Columns

Now I removed the columns that are not necessary to train the model. To keep the project simple and manageable, I also dropped values like `acre_lot` and limited the location column to state by dropping `street`, `city`, and `zip_code`. By only categorizing the locations by state, it helps the model training process by dimensionality reduction, especially during the one-hot encoding process (around 50 columns for each state instead of thousands or millions of values).

```
df2 = df1.drop(['brokered_by', 'status', 'prev_sold_date', 'acre_lot', 'street', 'city', 'zip_code'], axis = 'columns')
df2.head()
```

<img src="/assets/img/jupyter-output-2.png" width="50%"/>

Notice that I put the new dataframe with dropped values into `df2`. The dataset was now ready for cleaning.

## 2. Data Cleaning: Outlier Removal and Feature Engineering

Most of the time in data science projects is spent on data cleaning, ensuring the our final model is presented with proper and suitable data.

### Data Cleaning: NA Values

The first step in data cleaning is usually removing null and junk data, as well as removing duplicate data in the dataset. We can find the number of null values by using:

```
df2.isnull().sum()
```

<img src="/assets/img/jupyter-output-3.png" width="25%"/>

Here we find a hefty amount of null values. I also discovered that some prices were set to $0, which does not make sense, so I also removed them.

```
df3 = df2.dropna()
df3 = df3[df3['price'] != 0]
df3 = df3.drop_duplicates()
df3.shape
```

After removing the null values and duplicates, we see that the shape has been reduced to (1,430,967 rows, 5 columns).

> Another alternative to just removing the null values is to replace them with a median of the total dataset. However, since they took up more than a third of the dataset, I decided that it was safer to just remove them to avoid skewing the data.

### Dimensionality Reduction

One mignt notice that the value `Puerto Rico` showed up in the `state` column when head() was displayed. This is because the dataset includes both U.S. states and territories. Since we only need U.S. states in this project, I decided to group the non-states into an `Other` category. We can do this by using a list of U.S. states (conveniently written by ChatGPT) to filter the rest into the `Other` category:

```
us_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 
    'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 
    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 
    'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

df3.state = df3.state.apply(lambda x: 'Other' if x not in us_states else x)
len(df3.state.unique())
```
Now the number of unique states is shown to be 51 (the 50 U.S. states + the Other category) which is our expected result.

### Outlier Removal: Prices

To reduce the amount of price outliers, I first used this function (we will see an identical function for price per sqft) based on one standard deviation from the median. The dataset is split according to state (since it is one of the most predominant differences in prices) reduced from outliers, and then concatenated back together.

```
def remove_price_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('state'):
        m = np.median(subdf.price)
        st = np.std(subdf.price)
        reduced_df = subdf[(subdf.price>(m-st)) & (subdf.price<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
```

After running the function, the shape has been further reduced to (1,339,113 rows, 5 columns):

```
df4 = remove_price_outliers(df3)
df4.shape
```

### Feature Engineering

Here I added a new feature `price_per_sqft` which is just `price` divided by `house_size` or total sqft. This value will be used in cleaning the dataset further.

```
df4['price_per_sqft'] = df4['price']/df4['house_size']
df4.head()
```

<img src="/assets/img/jupyter-output-4.png" width="50%"/>

### Outlier Removal: Bedroom per Square Feet

I also discovered that some houses contained ridiculous number of bedrooms (we will address bathrooms later), the most having 444 bedrooms. I suspected that some of these were error values so I looked up the realistic average sqft per bedroom, which was 300sqft.

> I would consult this type of information from an expert if I was working on a real-life model for a company.

I looked at some of the outliers with `house_size` less than `300` per bedroom using the following:

```
df4[df4.house_size/df4.bed<300].head()
```

<img src="/assets/img/jupyter-output-5.png" width="50%"/>

And removed them, slightly reducing the shape to (1,321,719 rows, 6 columns):

```
df5 = df4[~(df4.house_size/df4.bed<300)]
df5.shape
```

### Outlier Removal: Price Per Square Feet

Returning to our new value `price_per_sqft`, we want to check if there are outliers for that value.

```
df5.price_per_sqft.describe()
```

<img src="/assets/img/jupyter-output-6.png" width="25%"/>

We can see that the `min` and `max` values are drastic extremes, at 0.000109 and 8714.29 respectively. We can use a similar function to the one we used for price outliers, which removes the outliers one std from the median for each state.

```
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('state'):
        m = np.median(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
```

Now the shape is reduced to (1,045,858 rows, 6 columns):

```
df6 = remove_pps_outliers(df5)
df6.shape
```

And the dataset looks better:

```
df6.price_per_sqft.describe()
```

<img src="/assets/img/jupyter-output-7.png" width="25%"/>

### Outlier Removal: Price Per Bedroom

To help the model make a more distinct prediction between the number of bedrooms, I wanted to remove rows where there is a higher price even when there are less bedrooms. We can visualize the current situation between 2-bedroom houses and 3-bedroom houses using matplotlib's scatterplots:

```
def plot_scatter_chart(df, state):
    twoBed = df[(df.state == state) & (df.bed == 2)]
    threeBed = df[(df.state == state) & (df.bed == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(twoBed.house_size, twoBed.price, color = 'blue', label = '2 bedrooms', s = 50)
    plt.scatter(threeBed.house_size, threeBed.price, marker = '+', color = 'green', label = '3 bedrooms', s = 50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(state)
    plt.legend()
    plt.show()
```

So after using the following function to remove said outlier, the shape is drastically reduced to (485,655 rows, 6 columns):

```
# Remove bedroom outliers (same state, but 2 bed price is higher than 3 bed, etc.)
def remove_bed_outliers_by_number(df):
    exclude_indices = np.array([])
    for state, state_df in df.groupby('state'):
        bed_stats = {}
        for bed, bed_df in state_df.groupby('bed'):
            bed_stats[bed] = {
                'mean': np.mean(bed_df.price_per_sqft),
                'std': np.std(bed_df.price_per_sqft),
                'count': bed_df.shape[0]
            }
        for bed, bed_df in state_df.groupby('bed'):
            stats = bed_stats.get(bed-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bed_df[bed_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df7 = remove_bed_outliers_by_number(df6)
df7.shape
```

And by using `plot_scatter_chart(df6, 'New York')` and `plot_scatter_chart(df7, 'New York')` we can see the differences for the state of New York:

<img src="/assets/img/scatter-new-york.png" width="100%"/>

Another example, for the state of Arkansas:

<img src="/assets/img/scatter-new-york.png" width="100%"/>

### Outlier Removal: More Bathrooms than Bedrooms?

There was one more outlier I wanted to look at, which was the number of bathrooms. It is unlikely that a house would have 2 more bathrooms than bedrooms, so I consider those as outliers.

> Similar to the minimum sqft per bedroom outlier, I would consult this information from an expert if I was working on a real-life model for a company.

We can view those using:

```
df7[df7.bath>df7.bed+2]
```

<img src="/assets/img/jupyter-output-8.png" width="50%"/>

And now the dataset is reduced to (483,553 rows, 6 columns) after removing those values:

```
df8 = df7[df7.bath<df7.bed+2]
df8.shape
```

### Histogram of Price Per Square Feet

The `price_per_sqft` distribution looks much better now when we run:

```
df8.price_per_sqft.describe()
```

<img src="/assets/img/jupyter-output-9.png" width="25%"/>

We can also plot a histogram of `price_per_sqft` of the current dataset. We can see that we now have a nice distribution across the dataset.

```
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()
```

<img src="/assets/img/histogram-price-per-sqft.png" width="100%"/>

### Remove Price Per Square Feet

We have now used the `price_per_sqft` to perform all necessary cleaning and no longer need it. We can drop it with:

```
df9 = df8.drop(['price_per_sqft'],axis='columns')
df9.head()
```

<img src="/assets/img/jupyter-output-10.png" width="25%"/>

At this point we have a fully cleaned dataset and can move on to one final step before building the model.

## 3. One Hot Encoding Using Pandas

## 4. Model Building Using Scikit-Learn

## 5. Creating a Python Flask Server

## 6. Creating a User-Friendly Website

## Summary

[See my other projects!](https://github.com/aJustinOng)
