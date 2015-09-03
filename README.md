## [Liberty Mutual Group: Property Inspection Prediction](https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction)

#### Summary
* This contest asks participants to predict a continuous 'Hazard' score for home inspections using 32 anonymized variables. Some variables are continuous, while others are categorical.
* This repository provides not a specific solution, but a framework for implementing your own. Specifically, it stresses fast iteration and model composability. Machine learning models rarely stand alone: a diverse set of data, models, and model hyperparameters is almost always advised in a competition setting.
* The basic paradigm for creating a solution is: `Dataset` + `Model` = `Prediction`. 
    - A `Dataset` accepts a `data_func`, which returns `X_train`, `y_train`, and `X_test` sets - a pd.DataFrame, pd.Series, and pd.DataFrame, respectively. 
    - A `Model` is assumed to be a scikit-learn regression model, exposing `fit` and `predict` methods.
    - A `Prediction` optionally accepts a `TargetTransform` object which, for example, may take the square root of our continuous target before the model is fit, then square the model's final predictions - and a `FeatureSelector` object which, for example, may remove several low-variance features in each cross-validation fold before fitting, then remove the union-set of these removed features before the final model is fit. 

#### To compose your own solution
These objects made be composed as follows:

```
prediction = Prediction(
    dataset=Dataset(func=data_v9)
    model=xgb.XGBRegressor(**params),
    base_path=PREDICTION_PATH,
    target_transform=TrimOutliersTargetTransform,
    save=True
)
prediction.cross_validate()
```

Calling `cross_validate` on the `Prediction` object will compute out-of-folds predictions, leaderboard predictions, gini scores in each fold, and much more. This data is then persisted, such that if an identical `Prediction` is ever created, the computation will be skipped. "Never delete a model," they say.

Finally, predictions can be composed together with an `Ensemble` object, which combines the leaderboard predictions (and, seperately, out-of-folds predictions, for cross-validation) by means of a user-defined funtion, such as `np.mean`.

#### Some things I tried
* Datasets:
    - Some continuous features varied sinusoidally with respect to the target, so I tried binning them. This is more important in linear models than it is in tree-based models.
    - Replacing categorical feature values with count of occurences.
    - ["Leave-one-out encoding"](http://nycdatascience.com/featured-talk-1-kaggle-data-scientist-owen-zhang/) of categorical features.
    - 2-way feature interactions.
* Models:
    - XGBoostRegressor, ExtraTreesRegressor, Ridge Regression, Bayesian Ridge Regression, Linear Regression, GradientBoostingRegressor. As per often, XGBoostRegressor performed best.
* Target transforms:
    - Square root transform, target transform, trim outliers transform.
* Final ensemble:
    - In this contest, I trained thousands of models. I then selected only the most performant models - based on their out-of-folds gini score - giving me roughly 20. 
    - I then used the out-of-folds predictions of each of these models as feature of a new model, called a "stacker." I then trained several hundred stackers.
    - To select my final stackers, I computed a correlation matrix between the out-of-folds predictions of each, and selected the top ~15 that were, on average, least correlated with the rest. Finally, I took the mean of the predictions of each of these stackers to create my final ensemble.
    