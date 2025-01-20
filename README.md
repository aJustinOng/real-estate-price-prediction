# Real Estate Price Prediction ![](./assets/img/house-icon.svg)
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

The model building section covers a majority of data science concepts like data cleaning, outlier removal, feature engineering, dimensionality reduction, one hot encoding, and K-Fold cross-validation.

<img src="/assets/img/website-ui-1.png" width="100%"/>

## Table of contents:
1. [Data Loading and Cleaning](#1-data-loading-and-cleaning)
2. [Outlier Removal and Feature Engineering](#2-outlier-removal-and-feature-engineering)
3. [One Hot Encoding Using Pandas](#3-one-hot-encoding-using-pandas)
4. [Model Building Using Scikit-Learn](#4-model-building-using-scikit-learn)
5. [Creating a Python Flask Server](#5-creating-a-python-flask-server)
6. [Creating a User-Friendly Website](#6-creating-a-user-friendly-website)
7. [Summary](#summary)

---

## 1. Data Loading and Cleaning

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

## 2. Outlier Removal and Feature Engineering

## 3. One Hot Encoding Using Pandas

## 4. Model Building Using Scikit-Learn

## 5. Creating a Python Flask Server

## 6. Creating a User-Friendly Website

## Summary

[See my other projects!](https://github.com/aJustinOng)