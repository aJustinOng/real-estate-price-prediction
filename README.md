# Real Estate Price Prediction ![](./assets/img/house-icon.svg)

**Skills:** `Python | NumPy | Pandas | Matplotlib | scikit-learn | HTML | CSS | JavaScript`

**Tools:** `Jupyter Notebook | VS Code | PyCharm | Flask | Postman`

##### [See my other projects!](https://github.com/aJustinOng)

---

## Overview

This project is based on CodeBasic's [Real Estate Price Prediction](https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg) project.

In this regression project, I used a U.S. real estate dataset (2.2M+ entries) on Kaggle that was extracted from Realtor.com to create a prediction model that estimates the price of a property based on house area (square feet), number of bedrooms and bathrooms, and state.

I started by preprocessing the dataset and used it to build a model with scikit-learn using linear regression. The model was then exported as a Pickle file. Next, I created a Python Flask server to run the model and receive GET and POST requests, which I tested using Postman. Lastly, I made a webpage using HTML, CSS, and JavaScript with a user-friendly UI, where the user can enter their desired inputs to get a predicted price.

The model building section covers a majority of data science concepts like data cleaning, outlier removal, feature engineering, dimensionality reduction, one hot encoding, and K-Fold cross-validation.

<img src="/assets/img/website-ui.gif" width="100%"/>

## Table of contents:
1. [Importing Libraries and Data Loading](#1-importing-libraries-and-data-loading)
2. [Data Cleaning: Outlier Removal and Feature Engineering](#2-data-cleaning-outlier-removal-and-feature-engineering)
3. [One Hot Encoding Using Pandas](#3-one-hot-encoding-using-pandas)
4. [Model Building Using Scikit-Learn](#4-model-building-using-scikit-learn)
5. [Creating a Python Flask Server](#5-creating-a-python-flask-server)
6. [Creating a User-Friendly Webpage](#6-creating-a-user-friendly-webpage)
7. [Summary](#summary)

---

## 1. Importing Libraries and Data Loading

### 1.1 Dataset

I found a [kaggle dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) that contained 2.2M+ real estate listings in U.S. states (and territories) with suitable values we can use to train a prediction model, such as price, location, number of bedrooms, number of bathrooms, and area in square feet. The data was collected from Realtor, a popular real estate listing website in the United States. It is a CSV file with the following columns:

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

### 1.2 Import Libraries

First, we need to import the necessary libraries and set certain matplotlib settings on a new Jupyter Notebook:

```
import pandas as pd  
import numpy as np  
from matplotlib import pyplot as plt  
%matplotlib inline  
import matplotlib  
matplotlib.rcParams['figure.figsize'] = (20, 10)
```

### 1.3 Load Dataset

We can read the dataset CSV file into the notebook using Pandas:

```
df1 = pd.read_csv('realtor-data.csv')
df1.head()
```

<img src="/assets/img/jupyter-output-1.png" width="100%"/>

And validate the dataset size (2,226,382 rows, 12 columns):

```
df1.shape
```

### 1.4 Drop Unnecessary Columns

We need to remove the columns that are not necessary to train the model. To keep the project simple and manageable, we also drop values like `acre_lot` and limit the location column to `state` by dropping `street`, `city`, and `zip_code`. By only categorizing the locations by state, it helps the model training process by dimensionality reduction, especially during the one-hot encoding process (around 50 columns for the 50 U.S. states instead of thousands or millions of values).

```
df2 = df1.drop(['brokered_by', 'status', 'prev_sold_date', 'acre_lot', 'street', 'city', 'zip_code'], axis = 'columns')
df2.head()
```

<img src="/assets/img/jupyter-output-2.png" width="50%"/>

Notice that the new dataframe with dropped values into `df2`. The dataset was now ready for cleaning.

## 2. Data Cleaning: Outlier Removal and Feature Engineering

Most of the time in data science projects is spent on data cleaning, ensuring the our final model is presented with proper and suitable data.

### 2.1 Data Cleaning: NA Values

The first step in data cleaning is usually removing null and junk data, as well as removing duplicate data in the dataset. We can find the number of null values by using:

```
df2.isnull().sum()
```

<img src="/assets/img/jupyter-output-3.png" width="25%"/>

Here we find a hefty amount of null values. Some prices were set to $0, which does not make sense, so we also remove them.

```
df3 = df2.dropna()
df3 = df3[df3['price'] != 0]
df3 = df3.drop_duplicates()
df3.shape
```

After removing the null values and duplicates, we see that the shape has been reduced to (1,430,967 rows, 5 columns).

> Another alternative to just removing the null values is to replace them with a median of the total dataset. However, since they took up more than a third of the dataset, I decided that it was safer to just remove them to avoid skewing the data.

### 2.2 Dimensionality Reduction

One mignt notice that the value `Puerto Rico` showed up in the `state` column when head() was displayed. This is because the dataset includes both U.S. states and territories. Since we only need U.S. states in this project, the non-states can be grouped into an `Other` category. We can do this by using a list of U.S. states (conveniently written by ChatGPT) to filter the rest into the `Other` category:

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

### 2.3 Outlier Removal: Prices

To reduce the amount of price outliers, we can use this function (we will see an identical function for price per sqft) based on one standard deviation from the median. The dataset is split according to state (since it is one of the most predominant differences in prices) reduced from outliers, and then concatenated back together.

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

### 2.4 Feature Engineering

Here we add a new feature called `price_per_sqft` which is just `price` divided by `house_size` or total sqft. This value will be used in cleaning the dataset further.

```
df4['price_per_sqft'] = df4['price']/df4['house_size']
df4.head()
```

<img src="/assets/img/jupyter-output-4.png" width="50%"/>

### 2.5 Outlier Removal: Bedroom per Square Feet

Some rows contained a ridiculous number of bedrooms (we will address bathrooms later), the most having 444 bedrooms. Some of these are probably error values so I looked up the realistic average sqft per bedroom, which was 300sqft.

> I would consult this type of information from an expert if I was working on a real-life model for a company.

We can look at some of the outliers with `house_size` less than `300` per bedroom by using the following:

```
df4[df4.house_size/df4.bed<300].head()
```

<img src="/assets/img/jupyter-output-5.png" width="50%"/>

And remove them, slightly reducing the shape to (1,321,719 rows, 6 columns):

```
df5 = df4[~(df4.house_size/df4.bed<300)]
df5.shape
```

### 2.6 Outlier Removal: Price Per Square Feet

Returning to our new value `price_per_sqft`, we want to check if there are outliers for that value.

```
df5.price_per_sqft.describe()
```

<img src="/assets/img/jupyter-output-6.png" width="40%"/>

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

<img src="/assets/img/jupyter-output-7.png" width="40%"/>

### 2.7 Outlier Removal: Price Per Bedroom

To help the model make a more distinct prediction between the number of bedrooms, we can remove rows where there is a higher price even when there are less bedrooms. We can visualize the current situation between 2-bedroom houses and 3-bedroom houses using matplotlib's scatterplots:

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

<img src="/assets/img/scatter-arkansas.png" width="100%"/>

### 2.8 Outlier Removal: More Bathrooms than Bedrooms?

There was one more outlier we need to look at, which was the number of bathrooms. It is unlikely that a house would have 2 more bathrooms than bedrooms, so we consider those as outliers.

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

### 2.9 Histogram of Price Per Square Feet

The `price_per_sqft` distribution looks much better now when we run:

```
df8.price_per_sqft.describe()
```

<img src="/assets/img/jupyter-output-9.png" width="40%"/>

We can also plot a histogram of `price_per_sqft` of the current dataset. We can see that we now have a nice distribution across the dataset.

```
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()
```

<img src="/assets/img/histogram-price-per-sqft.png" width="100%"/>

### 2.10 Remove Price Per Square Feet

We have now used the `price_per_sqft` to perform all necessary cleaning and no longer need it. We can drop it with:

```
df9 = df8.drop(['price_per_sqft'],axis='columns')
df9.head()
```

<img src="/assets/img/jupyter-output-10.png" width="40%"/>

At this point we have a fully cleaned dataset and can move on to one final step before building the model.

## 3. One Hot Encoding Using Pandas

We cannot build a model using categorical values, so we can use the one hot encoding method (aka dummy variables) to convert each state into a binary format. We do this with Pandas' built-in dummies function:

```
dummies = pd.get_dummies(df9.state)
dummies.head()
```

<img src="/assets/img/jupyter-output-11.png" width="100%"/>

Then we concatenate the `dummies` dataframe to the main dataframe, while dropping the `Other` category. We can drop it since if every other dummy variable is `False`, we know that the only column that can be `True` is `Other`.

```
df10 = pd.concat([df9, dummies.drop('Other', axis='columns')], axis='columns')
df10.head()
```

<img src="/assets/img/jupyter-output-12.png" width="100%"/>

Since we have the states as dummy variables, we can now drop the `state` column:

```
df11 = df10.drop('state', axis='columns')
df11.head()
```

<img src="/assets/img/jupyter-output-13.png" width="100%"/>

And verify the shape of the dataframe, which is now (483,553 rows, 54 columns). The columns are now `price`, `bed`, `bath`, `house_size`, and the 50 states.

```
df11.shape
```

## 4. Model Building Using Scikit-Learn

We can finally begin building our linear regression model that can predict real estate prices based on the number of bedrooms, number of bathrooms, area in sqft, and state.

### 4.1 Build Linear Regression Model

The first step of the model building process is to split the dataframe into two sub-dataframes: `y`, the value we are trying to predict (prices), and `X`, the values we use to do the prediction (the rest of the data). We get `X` by simply dropping `price` from the cleaned dataframe.

```
X = df11.drop('price', axis='columns')
X.head()
```

<img src="/assets/img/jupyter-output-14.png" width="100%"/>

And we get `y` by isolating `price` from the same dataframe:

```
y = df11.price
y.head()
```

<img src="/assets/img/jupyter-output-15.png" width="30%"/>

Using `X` and `y`, we split them into train and test datasets in a 80:20 ratio using scikit-learn:

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

And build the linear regression model using scikit-learn:

```
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)
```

We get a test score of `0.80876207`, which is not too bad for a simple model like this.

### 4.2 Use K-Fold Cross-Validation on Linear Regression model

We can further use K-fold cross-validation to test our model. K-fold cross-validation splits the data in `k` subsets (or folds), and then performs training and testing on each subset, averaging the the score from each subset. We continue to use scikit-learn to do this:

```
from sklearn.model_selection import ShuffleSplit, cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)
```

We get an array of scores `0.81316742`, `0.74627989`, `0.81752128`, `0.81101241`, and `0.7370295`. So our model has a score around 0.7-0.8, which is fairly accurate for a basic linear regression model.

### 4.3 Test Model on Properties

Now that the model is trained and test, we can try it out! Create a function that takes in `state`, `bed`, `bath`, and `sqft` parameters and outputs a predicted price value of a property.

```
def predict_price(state, bed, bath, sqft):    
    loc_index = np.where(X.columns==state)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = bed
    x[1] = bath
    x[2] = sqft
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
```

If we run `predict_price('Arkansas', 2, 2, 1000)` we get `63502.942380954366`.

If we run `predict_price('Arkansas', 3, 3, 1000)` we get `106350.11599686736`.

If we run `predict_price('New York', 2, 2, 1000)` we get `521071.2429128873`.

If we run `predict_price('New York', 3, 3, 1000)` we get `563918.4165288003`.

We expect a property in New York to have a much higher price than in Arkansas and more bedrooms/bathrooms to increase the price, so the model looks good!

### 4.4 Export the Model to Pickle File

To use the model in a server, we can export it as a pickle file. Import the `pickle` library and use it to write a pickle file.

```
import pickle
with open('usa_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)
```

### 4.5 Export the Column Data to a JSON File

We also need to export the `X` columns to a JSON file for future use. The columns in `X` are as such:

```
X.columns
```

<img src="/assets/img/jupyter-output-16.png" width="75%"/>

Convert all of them to lowercase and export them:

```
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
```

## 5. Creating a Python Flask Server

> In this section I will not be going over the steps in chronological order, but rather briefly explain each file in the `server` folder provided in this GitHub repository. This is because the steps frequently jump between `server.py` and `util.py`, and can cause a lot of confusion if done step-by-step.

In order to run the model on a website, we have to create a server to host the model. We can use a Python Flask server for this project. We create two files `server.py` and `util.py` and a folder called `artifacts` where we copy the `usa_home_prices_model.pickle` and `columns.json` files we just exported from Jupyter Notebook. Using PyCharm or another Python editor will work for this section.

### 5.1 `server.py`

This is the Python file which we use to run the server. We first import the needed libraries from Flask as well as `util.py`, which will contain the functions that we need to perform GET and POST requests.

```
from flask import Flask, request, jsonify
import util
```

We set the app as a Flask server:

```
app = Flask(__name__)
```

This is a GET method function to get a list of the states by calling `get_location_names()` from `util.py`. This is called when the server gets a HTTP GET request with the url `/get_location_names`.

```
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
```

This is a POST method function to return a predicted real estate price from the given `state`, `bed`, `bath`, and `sqft` parameters by calling `get_estimated_price()` from `util.py`. This is called when the server gets a HTTP POST request with the url `/predict_home_price` and returns a response if provided with valid parameters.

```
@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    sqft = float(request.form['sqft'])
    state = request.form['state']
    bed = int(request.form['bed'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(state, bed, bath, sqft)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
```

This main function runs when `server.py` is run. It logs a prompt and calls `load_saved_artifacts()` from `util.py` before starting the Flask server.

```
if __name__ == "__main__":
    print("Starting Python Flask Server for Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()
```

### 5.2 `util.py`

This Python file is where we store all the functions that are used in `server.py`. This is where we load the artifacts, get the locations from the JSON file, and feed the model with with the required parameters and return the predicted price back to `server.py` Here we import the libraries required to do so:

```
import json
import pickle
import numpy as np
```

Now declare global variables:

```
__locations = None
__data_columns = None
__model = None
```

This is the function that is automatically run when `server.py` starts, since it is called in its main function. It loads the JSON file `columns.json` into the global variable `__data_columns` and splits the array to remove the `bed`, `bath`, and `sqft` columns before storing the rest into the global variable `__locations`. If there is no existing model, it stores the model from `usa_home_prices_model.pickle` into the global variable `__model`.

```
def load_saved_artifacts():
    print("Loading save artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open("./artifacts/usa_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")
```

This is the function where we run the model to predict a price based on `state`, `bed`, `bath`, and `sqft`. The `x` array stores the values of the parameters, which we feed into the model before returning the result.

```
def get_estimated_price(state, bed, bath, sqft):
    try:
        loc_index = __data_columns.index(state.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bed
    x[1] = bath
    x[2] = sqft
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
```

The following should be self-explanatory:

```
def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns
```

And the main function is runs when `util.py` is run. Used for testing purposes.

```
if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Arkansas', 2, 2, 1000))
    print(get_estimated_price('Arkansas', 3, 3, 1000))
    print(get_estimated_price('New York', 2, 2, 1000))
    print(get_estimated_price('New York', 3, 3, 1000))
```

### 5.3 Postman Testing

Now we are ready to start the server up. Make sure that the `columns.json` and `usa_home_prices_model.pickle` files are in a `artifacts` folder in the same directory as `server.py` and `util.py`. Open up a terminal in PyCharm (or Python editor of choice) and make sure that it is in that directory. Run the server using:

```
python server.py
```

When the server loads it should say:

```
Running on http://127.0.0.1:5000
```

Open up Postman and do a GET request with the URL `127.0.0.1:5000/get_location_names` or `localhost:5000/get_location_names` and we should get the following 200 response:

<img src="/assets/img/postman-get-locations.png" width="100%"/>

Next, do a POST request with the URL `127.0.0.1:5000/predict_home_price` or `localhost:5000/predict_home_price` along with valid parameters in the form-data option in the body tab (check image if unsure). Type `state`, `bed`, `bath`, and `sqft` under the `Key` column and valid values in the `Value` column. If everything was entered correctly, we should get the following 200 response with `estimated_price`:

<img src="/assets/img/postman-post-predict-home-price.png" width="100%"/>

If everything went as expected, we can now create a UI website that call the same GET and POST requests and display `estimated_price` to a user.

## 6. Creating a User-Friendly Webpage

> In this section, I will not go over the code in the HTML and CSS files, since they are not the focus of the project. I will only briefly go over the design and necessary functions of them. The files for this webpage is all under the `client` folder provided in this GitHub repository.

The webpage is a basic one that serves the singular purpose of allowing a user to conviniently input their desired `state`, `bed`, `bath`, and `sqft` values and receive an output with the click of a button. It is created with basic HTML, CSS, and JavaScript. We can create `app.html`, `app.css`, and `app.js` in the same folder. Visual Studio Code is a suitable code editor for all three files.

### 6.1 HTML and CSS

As mentioned in the beginning notes of this section, I will only be going over `app.html` and `app.css` briefly. The files for this webpage is all under the `client` folder provided in this GitHub repository. The design is very simple: a form with the necessary inputs and output on a blurred image.

<img src="/assets/img/website-ui-0.png" width="100%">

The HTML has a text field with the id `uiSqft` for the user to input the desired sqft, two rows of radio buttons with names `uiBHK` and `uiBathrooms` where the user can choose between 1 to 7 bedrooms and/or bathrooms respectively, and a dropdown with the id `uiLocations` which the user can pick between states. The dropdown will be populated with the state names after calling the `/get_location_names` GET request. At the bottom there a "Estimate Price" buttom to submit the inputs and an empty result box which will display the result once it gets an output from the server.

Additional styling and tranisition animations is done in CSS. We can add a blurred image as the background just to help the webpage look more aesthetic, but it is not required.

### 6.2 JavaScript

To load in the state names and get an output from the model, we have to use JavaScript. First, we create a function to get the user-selected number of bedrooms from `uiBHK` by checking which radio button was selected:

```
function getBedValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for (var i in uiBHK) {
        if (uiBHK[i].checked) {
            return parseInt(i) + 1;
        }
    }
    return -1; // Invalid Value
}
```

Then do the same for the number of bathrooms from `uiBathrooms`:

```
function getBathValue() {
    var uiBathrooms = document.getElementsByName("uiBathrooms");
    for (var i in uiBathrooms) {
        if (uiBathrooms[i].checked) {
            return parseInt(i) + 1;
        }
    }
    return -1; // Invalid Value
}
```

We also need a function that will be called when the "Estimate Price" button is clicked. This function grabs all the input values with some help from `getBedValue()` and `getBathValue()` and sends a `/predict_home_price` POST request to the server. If the server returns a result, this function converts the result into a string and updates the HTML of the result box with it. We can add a dollar sign ($) to the front of the string so that the user understands that it is a currency value.

```
function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft");
    var bedrooms = getBedValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("uiEstimatedPrice");

    var url = "http://127.0.0.1:5000/predict_home_price";

    $.post(url, {
        state: location.value,
        bed: bedrooms,
        bath: bathrooms,
        sqft: parseFloat(sqft.value)
    }, function (data, status) {
        console.log(data.estimated_price);
        estPrice.innerHTML = "<h2>$" + data.estimated_price.toString() + "</h2>";
        console.log(status);
    });
}
```

This function runs when the page loads. The only thing we need from this function is to retrieve the state names by calling the `/get_location_names` GET request. Since the states are all in lowercase, we also capitalize each word in this function (e.g. "new york" -> "New York") before it is populated in the HTML dropdown menu.

```
function onPageLoad() {
    console.log("document loaded");
    var url = "http://127.0.0.1:5000/get_location_names";
    $.get(url, function (data, status) {
        console.log("got response for get_location_names request");
        if (data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            for (var i in locations) {
                var locationName = locations[i]  // Capitalize each state
                    .split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ');
                var opt = new Option(locationName);
                $('#uiLocations').append(opt);
            }
        }
    });
}
```

Finally, as with any JavaScript file, we put this at the end.

```
window.onload = onPageLoad;
```

### 6.3 Final Result

Now we can open up the HTML file on any browser (make sure the server is still running) and use the webpage UI!

Here is the result when submitting the default values of 2,000 sqft, 1 bedroom, 1 bathroom, and the state of Alabama:

<img src="/assets/img/website-ui-1.png" width="100%"/>

Of course, we can select other values, such as 2,400 sqft, 3 bedrooms, 2 bathrooms, and the state of New York:

<img src="/assets/img/website-ui-2.png" width="100%"/>

Now the webpage is ready to deploy to production on a cloud service if we wish to, but that is beyond the scope of this project (and my wallet). Thank you for reading through this project!

## Summary

In this Real Estate Price Prediction project, I took a Kaggle dataset of more than 2.2 million rows and thoroughly cleaned it with outlier removal, feature engineering, and dimensionality reduction. I then used one hot encoding to create dummy values for each state, a categorical value, and built a linear regression model using scikit-learn. I also ultilized K-Fold cross-validation to test the accuracy of the model, which was around 0.7-0.8.

To take this project further, I wanted to run the model on a webpage. To do this, I created a Python Flask server that hosted the model to receive HTTP requests, which I tested with Postman. Next, I used HTML, CSS, and JavaScript to create a webpage with a simple yet user-friendly design. The webpage sends GET and POST requests to the Flask server, and displays the final results to the user.

[See my other projects!](https://github.com/aJustinOng)
