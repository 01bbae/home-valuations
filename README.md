# Housing Valuation

The goal of this project is to develop a predictive model for housing prices using machine learning techniques, specifically XGBoost. By analyzing various features of residential properties, we aim to create an accurate and robust model that can estimate property values based on details of recently sold properties in California in the year 2024.

You can open this dashboard using this [link](https://avmshinydash.duckdns.org/)

## Getting Started

It is recommended that you download and upload these three datasets to the Google Colab notebook before running the notebook.

1. `lotwize_case.xlsx`
2. `nearest_features_distance.csv`
3. `climate_population_density`

<details>

<summary>If you want to reproduce the data from scratch (Not Recommended)</summary>

While the latter two are not required to download and have code inside the notebook to reproduce them, it will take a long time to get the results of these datasets. (Runtime is estimated to be around 14 hours to aquire the datasets)

The code to reproduce them are commented out so the notebook doesn't produce any errors and so that they do not run by default.

If you did want to run these cells, I highly recommend downloading the notebook and running the notebook in your local enviornment as Google Colab will likely disconnect while in the middle of running.

</details>

Installing

Most dependencies of our project will be installed by default by Google Colab. One dependency that is not there by default is `shap`. We install `shap` in the notebook for you but if it does not work, try install `shap` again by running

```
!pip install shap
```

in one of the cells

## Built With

[XGBoost](https://xgboost.readthedocs.io/en/stable/)

[Scikit-Learn](https://scikit-learn.org/stable/)

[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

[SHAP](https://shap.readthedocs.io/en/latest/)

[Seaborn](https://seaborn.pydata.org/)

[Matplotlib](https://matplotlib.org/stable/index.html)

## Authors

BJ Bae (4PM section)

- bbae@chapman.edu

Ben Weiskopf (7PM section)

- weiskopf@chapman.edu
