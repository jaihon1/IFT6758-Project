---
layout: post
title: IFT6758 Milestone 2
---

### Question 2

Using our work from the previous milestone, we have extended our features by adding the following ones presented in the table below:

| Feature      | Description |
| ----------- | ----------- |
| distance_net | The shot/goal distance from the net |
| angle_net | The shot/goal angle from the net |
| is_goal | Whether or not the shot was a goal or not |
| empty_net | Whether or not the shot/goal was at an empty net |


<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/histogram_goals_nogoals_vs_distance.png" alt="histogram_goals_nogoals_vs_angle">
    <figcaption style="font-size: 12px;text-align: center;">Figure 1: Goals and no goals vs distance.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/histogram_goals_nogoals_vs_angle.png" alt="histogram_goals_nogoals_vs_angle">
    <figcaption style="font-size: 12px;text-align: center;">Figure 2: Goals and no goals vs angle.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/angle_vs_distance.png" alt="angle_vs_distance">
    <figcaption style="font-size: 12px;text-align: center;">Figure 3: angle vs distance.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/goal_rate_vs_distance.png" alt="goal_rate_vs_distance">
    <figcaption style="font-size: 12px;text-align: center;">Figure 4: Goal rate vs distance.</figcaption>
</figure>

All the figures above give us interesting information about shots and goals in the NHL. For example, figure 1 shows us that both goals and no goals happened more often closer to the net and that goals are much less frequent than goal. Figure 3 tells us that shots that are done farther from the net, generally are more aligned with it (smaller angle). Finally, if we analyze Figure 4, we can observe that when attacking players are very close to the opponent's net, the chance that they score is much higher which intuitively makes sense.

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/goal_rate_vs_angle.png" alt="goal_rate_vs_angle">
    <figcaption style="font-size: 12px;text-align: center;">Figure 5: Goal rate vs angle.</figcaption>
</figure>

From Figure 5 above, we can see that the goal rate is much higher when the shot is coming from the left and right side compared to when the shot comes from the center of the ice. This makes sense as goalies are much more vulnerable when shots come from the top of the circles (both left and right circles near the goalie).

Another interesting thing about Figure 5 is when we compare the goal rate from the left side to the right side. One reason why the goal rate is higher on the right side could be because the majority of NHL goalies have their glove on their left hand (maybe it's easier to stop shots with the glove in contrast to the blocker hand). Another reason might simply be because players shooting from the right side are much better than the ones shooting from the left side, and therefore have a higher goal rate.

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/histogram_goals.png" alt="Goals (empty net and non-empty net) vs distance">
    <figcaption style="font-size: 12px;text-align: center;">Figure 6: Goals (empty net and non-empty net) vs distance from net.</figcaption>
</figure>

Looking at the data from Figure 6, we can observe many interesting facts. Firstly, the vast majority of goals are on non-empty net which is logical since goalies are in the net most of the time. Secondly, we can observe that most of the goals are being scored within 60 feet from the net, which is inside the opponents' half of the rink. These two observations are aligned with our domain knowledge, and it makes perfect sense that the further you are from the opponents' net, the harder it is to score when there is a goalie in front of the net. With that said, the goals that were made from a distance of 150 feet when there was a goalie sound a bit unlikely.

We can observe in Figure 6 that the goals scored on a non-empty net from a distance of 150-170 feet are quite high. It could be that it has been originally misclassified as "non-empty net goals" as opposed to "empty-net goals". Another reason could be that these goals were scored by the other team that was then misclassified.


### Question 3

#### Results
For our baseline, we trained a Logistric Regression model using only the *distance* feature that we have previously extracted from the raw data, and it gave us a **90.59%** accuracy when we ran it on our validation dataset. We also generated the following confusion matrix to have a better look at our model's results:

| Target/Prediction | **Class 0 (not goal)** | **Class 1 (goal)** |
| :-------: | :-------: | :-------: |
| **Class 0 (not goal)** | 70748 | 0 |
| **Class 1 (goal)** | 7344 | 0 |

This confusion matrix clearly shows us that there is a major issue with our predictions. We are only getting high accuracy performance because the majority of our data points are classified as a *not goal*. By always predicting *not goal* our model does a pretty good job if we only look at the overall accuracy.


#### Analysis
From Figure 7 below, the main thing we can observe is that shots that have a higher probability represents a much greater proportion of the total goals scored compared to shots with lower probabilities. Another important aspect is how this proportion metric is different for our different models. Even though the model trained on the distance feature and the model trained on the angle feature are better than the random baseline, the model that we trained on both features (distance and angle) gave us better results. Meaning it is much better at predicting the probability that a shot would turn to be a goal.


<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/cumulative_sum_goal_baseline.png" alt="cumulative_sum_goal_baseline">
    <figcaption style="font-size: 12px;text-align: center;">Figure 7: Logistic Regression: Goal proportion.</figcaption>
</figure>

The results shown in Figure 8 is also about shot probabilities. It shows us that our trained models perform much better that the random classifier at predicting the shot probability. As in our previous analysis, our model that was trained on both features (distance and angle) does give us better results that models trained on the features separately.

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/goal_rate_curve_baseline.png" alt="goal_rate_curve_baseline">
    <figcaption style="font-size: 12px;text-align: center;">Figure 8: Logistic Regression: Goal rate.</figcaption>
</figure>

In order to have a deeper analysis of the behavior of our binary classifiers, using our results we generated a receiver operating characteristic curve (ROC). As we can see in Figure 9 above, the random classifier gives a perfect diagonal as expected. We can also observe that our model trained on both of our features gives the better curve compared to our models that were trained separately on the features. Our ROC score is also much higher (*area=0.68*) when we trained our model on both features.

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/roc_curve_baseline.png" alt="roc_curve_baseline">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Logistic Regression: ROC curve.</figcaption>
</figure>

Given the calibration curve shown in Figure 10, we can easily see that our trained models did learn some valuable representations of our data. Comparing all our current models, the model that was trained on both features (distance and angle) has the closest calibration values to the *perfectly* calibrated model. Again, as mentioned before, it confirms that overall this model is the model that gives us the best results so far.

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/calibration_curve_baseline.png" alt="calibration_curve_baseline">
    <figcaption style="font-size: 12px;text-align: center;">Figure 10: Logistic Regression: Calibration curve.</figcaption>
</figure>


#### Links to our models

1. [Logistic Regression on distance and angle](https://www.comet.ml/jaihon/ift6758-project/88c175fd9d3c4892acf334fcfdb4a6d0)
2. [Logistic Regression on distance](https://www.comet.ml/jaihon/ift6758-project/6997fdfbdc76426db60408591e58ac5a)
3. [Logistic Regression on angle](https://www.comet.ml/jaihon/ift6758-project/934baca85c9448c997d8d0727845db65)



### Question 4

We added below a list of all of the features that we created, and we listed each feature by both the column name
in the dataframe and a simple explanation. For the novel features, we describe what they are.
At the end, we added a link to the experiment which stores the filtered DataFrame.


| Feature      | Description |
| ----------- | ----------- |
| current_time_seconds | total sum of seconds elapsed in the game |
| period | period of the game during which the shot happened |
| coordinate_x | coordinates x  of the shot |
| coordinate_y | coordinates y  of the shot |
| distance_net | distance from the shot to the net |
| angle_net | angle between the shot and the net |
| shot_type | type of Shot (Wrist, Slap, Backhand, etc...) |
| previous_event_type | type of the last event |
| previous_event_x_coord | coordinates x of the last event |
| previous_event_y_coord | coordinates y of the last event |
| shot_last_event_delta | time elapsed since the last event |
| shot_last_event_distance | distance calculated from the last event |
| Rebound | Rebound of the last event (True if shot, otherwise False) |
| Change_in_shot_angle | change in the shot angle if the shot is a rebound |
| Speed | defined as the distance from the previous event, divided by the time since the previous event |
| time_since_pp_started |  time in seconds since the penalty started |
| current_friendly_on_ice | Number of friendly players on ice|
| current_opposite_on_ice | Number of opposite players on ice|


In the bonus question, we added a few more features like the time since the penalty started and the number of friendly and opposite players on ice. To compute the time since the penalty started, we started
by generating all types of events in our game, by evaluating, at each event, if there was a
penalty and by checking on which side the team was. We then built a tidy event object that gave the time and coordinates details
relative to the previous event. Finally, we got the current event time and subtracted
the starting time of the penalty from the current time to have the time since the penalty started (two types of penalties generated).
To get the number of friendly players on ice and the number of opposite players on ice, we first checked the side of the team to figure out who is friendly and who is not and
then subtracted the number of players lost depending on the type of the
penalty from 5.


link to the experiment which stores the filtered DataFrame artifact
(https://www.comet.ml/jaihon/ift6758-project/fae888ad53de4d1aa940a67b96d106ab?assetId=e46feef96edc4bf8afe7c676f05c192b&assetPath=dataframes&experiment-tab=assets)
[wpg_v_wsh_2017021065.csv]

### Question 5
> The code for this section can be found in xgboost_models.py

The first XGBoost model was trained on the distance and angle from the net features just like the regression in
section 3. We trained the model on approximately 75% of the training data from seasons 2015 to 2018 (inclusive). Since
the data is very unbalanced, we made sure that the splitting into training/validation kept the proportion of goal and
no goal by using a stratified option. We also standardized both features to have them in a comparable range centered
around 0. The resulting model performed somewhat better than the regression models as shown on figures 11 to 14. This model is
represented by the blue curve in the figures. If we compare this curve and the ones from figures 7 to 10, we first notice
that the area under the curve of the ROC curve is slightly larger than the one from the regression models with a value of 0.71
compared to the highest one of 0.70 in the regressions. For the goal rate and cumulative sum of goals in function with
the shot probability model percentile, the curves are fairly similar for the xgboost model and the regression model trained
on distance and angle. They have the same shape and in the case of the goal rate one, they are in the same range also.
The real difference comes from the calibration curve where we see that the probabilities from the xgboost model ranges
from 0 to about 0.85 as opposed to the regression models which go only up to 0.25. This tells us that the xgboost models
is better calibrated and that the probabilities are more telling and "accurate" than the ones from the regression models
as it tends to be closer to the perfectly calibrated curve.
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/roc_curve_xgboost.png" alt="roc_curve_xgboost">
    <figcaption style="font-size: 12px;text-align: center;">Figure 11: XGBoost: ROC curve.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/goal_rate_xgboost.png" alt="goal_rate_xgboost">
    <figcaption style="font-size: 12px;text-align: center;">Figure 12: XGBoost: goal rate.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/cumulative_goals_xgboost.png" alt="cumulative_sum_xgboost">
    <figcaption style="font-size: 12px;text-align: center;">Figure 13: XGBoost: cumulative sum.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/calibration_xgboost.png" alt="calibration_xgboost">
    <figcaption style="font-size: 12px;text-align: center;">Figure 14: XGBoost: calibration curve.</figcaption>
</figure>

For the next xgboost model, we started by standardizing all the numerical features and remove the rows which had nan values.
For the particular case of the *Speed* column, we had to impute a new value for the rows which had infinity values. To do so,
we simply changed the infinity with the maximum value (beside infinity) in the column. We then transformed all our categorical
data into one-hot encoding. This is only after doing this that
we did some hyperparameter optimization. Before tuning anything, we started by finding the relations between
different hyperparameters and metrics. These are shown on figures 15, 16 and 17 where we evaluated our model on
accuracy, precision and F1 score for a range of values on the number of estimators (trees), the maximum depth of each tree
and the l2 regularisation coefficient lambda. We found that the accuracy stayed constant around 0.907 for all parameters.
For both the number of estimators and the maximum depth, we noticed that increasing them decreased the precision, but
increased the F1 score. This probably meant that the recall was increasing while the precision was decreasing. Since we
care both about precision and recall, and even though generally speaking a higher F1 score is better, it is difficult
to say if increasing the number of estimators and the maximum depth is really better. Also, the precision decreases more
rapidly than the F1 score increases. For the regularisation coefficient on figure 17, it seems that the higher the coefficient,
the better since the precision increases while the F1 score and accuracy stays pretty much constant.

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/n_estimators_vs_metrics_xgboost.png" alt="number_of_estimators_vs_metrics">
    <figcaption style="font-size: 12px;text-align: center;">Figure 15: Relation between the number of estimators and different metrics.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/max_depth_vs_metrics_xgboost.png" alt="max_depth_vs_metrics">
    <figcaption style="font-size: 12px;text-align: center;">Figure 16: Relation between the max depth of a tree and different metrics.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/reg_lambda_vs_metrics_xgboost.png" alt="reg_lambda_vs_metrics">
    <figcaption style="font-size: 12px;text-align: center;">Figure 17: Relation between the regularisation coefficient lambda (l2) and different metrics.</figcaption>
</figure>

Once we finished exploring some hyperparameters, we did a randomized search over the hyperparameters space focusing
on the same parameters as before as well as the learning rate. We did not do a grid search because it would have taken
too long to search all the parameters we wanted to try, but also because the hyperparameters do not have such a big influence
on the results as was shown on the figures above. Since the accuracy is almost constant for all parameters, we decided to
focus on the ROC area under the curve. The best model was selected with the highest ROC AUC. The resulting model was
found to have the following hyperparameters:

|      Hyperparameter     | Value |
|:-----------------------:|:-----:|
|     # of estimators     |  300  |
|      Maximum depth      |   5   |
| Regularisation &lambda; |   0   |
|      Learning rate      |  0.03 |

It is the orange curve on the figures 11 to 14. Even if this is the *best* model, the one right below in terms of performance
was very close to it with a ROC AUC of 0.75 compared to the model presented here which has 0.76.

As for our last experience, we tried some feature selection. We tried two different methods: lasso and mutual information.
For the lasso one, we evaluated a lasso linear model on our training dataset and selected the features that had the highest
coefficient of importance, i.e. they were higher than the median. The features selected are shown in red on figure 18 which
presents the importance coefficient according to the lasso model for all of our features. The features that are in parentheses
are the categorical data that was encoded using one-hot vectors. We can see that some values in categorical data seem to be
"useless" like the type of period in which the shot happens (overtime vs regular). Interestingly, we noticed that knowing the side of the
team that is shooting (right vs left) seems to be more important when it is left. Not too surprisingly, some type of shots seem to
have a bigger impact like *deflected* or *tip-in* or *wrap-around* compared to wrist shot. If we recall from the previous milestone,
we concluded that *deflected* and *tip-in* were among the most dangerous shots which seems to agree with this. These are some of
the interesting observations that we found from this figure.

<figure style="display: block;margin-left: auto; margin-right: auto;width:100%;height:100%;">
    <img src="/public/importance_feature_lasso_xgboost.png" alt="importance_feature_xgboost">
    <figcaption style="font-size: 12px;text-align: center;">Figure 18: Importance of each feature according to lasso.</figcaption>
</figure>

Anyway, once we selected the features in red, we trained our model using a randomized search again on the same hyperparameters as before.
We found a model that was performing similarly as our best one trained on all features as we can see by comparing the green (lasso trained one) and orange
curves on figures 11 to 14.

For the mutual information model, we did a randomized search on the same hyperparameters as before, but with the addition of
choosing a number of features according to the mutual information. Our best model using this trained on 15 features. A comparison
of the chosen features between the lasso model and mutual information score is shown in the following table where only
the features selected by either one of them is presented:

|           Feature           | Lasso | Mutual |
|:---------------------------:|:-----:|:------:|
|         coordinate_x        |   x   |    x   |
|         coordinate_y        |       |    x   |
|         distance_net        |   x   |    x   |
|          angle_net          |       |    x   |
|    time_since_pp_started    |       |    x   |
| previous_event_time_seconds |   x   |        |
|   current_friendly_on_ice   |       |    x   |
|   current_opposite_on_ice   |   x   |    x   |
|    shot_last_event_delta    |   x   |    x   |
|           Rebound           |   x   |        |
|            Speed            |   x   |    x   |
|             away            |       |    x   |
|             home            |       |    x   |
|           Backhand          |   x   |        |
|          Deflected          |   x   |        |
|          Slap Shot          |   x   |        |
|          Snap Shot          |   x   |        |
|            Tip-In           |   x   |        |
|         Wrap-around         |   x   |        |
|          Wrist Shot         |       |    x   |
|              1              |   x   |        |
|              3              |   x   |        |
|              4              |   x   |        |
|           REGULAR           |       |    x   |
|             left            |   x   |    x   |
|            right            |       |    x   |
|           FACEOFF           |   x   |        |
|           GIVEAWAY          |   x   |        |
|             HIT             |   x   |        |
|           TAKEAWAY          |   x   |        |

*the numbers 1,3,4 are the period.

From this table, we can observe that both method seem to agree on some features like the coordinate x, the distance or the
speed. However, the interesting features are the one selected by the mutual information, but not the lasso since the lasso took
more. Among those are the types of shots, lasso take all of the types except the wrist shot, but mutual information does the complete
opposite and select only the wrist shot. Mutual information also takes the period type regular into account whereas lasso
completely ignore them.

The model resulting from training with mutual information is shown in red in the figures 11 to 14. We can see that it does not
perform as well as the models trained on all features and on the lasso selected features on figures 11 to 13 where the curves are
closer to the default XGBoost model.

We want to note that all of the curves on figure 14 start of pretty linear which is good because it follows the perfectly
calibrated model.

After inspecting figures 11 to 14, we concluded that the best model was the one trained on all features because it has the
highest ROC AUC, but also because its calibration curve seems to be the most linear one. Indeed, we can see that the orange
curve varies less than the others after around 0.5 probabilty.

The links to the different experiments shown in this section can be found here:
1. [XGBoost trained on distance and angle with default hyperparameters](https://www.comet.ml/jaihon/ift6758-project/20c76cb9d81541b5ae0d2b320e59f59f)
2. [XGBoost with hyperparameter tuning and trained on all features](https://www.comet.ml/jaihon/ift6758-project/05c804186d274b0c8955cbb14b1a66b3)
3. [XGBoost with hyperparameter tuning and trained on subset of features selected by Lasso](https://www.comet.ml/jaihon/ift6758-project/74f19b8137e14336ba1e49c198dfd3e6)
