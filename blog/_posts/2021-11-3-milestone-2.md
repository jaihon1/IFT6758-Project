---
layout: post
title: IFT6758 Milestone 2
---

### Question 2
> The code for this section can be found in project/ift6758/visualizations/plot_feature_engineer_1.py

Using our work from the previous milestone, we have extended our features by adding the following ones presented in the table below:

| Feature      | Description |
| ----------- | ----------- |
| distance_net | The shot/goal distance from the net |
| angle_net | The shot/goal angle from the net |
| is_goal | Whether or not the shot was a goal or not |
| empty_net | Whether or not the shot/goal was at an empty net |

From these new features, we created different figures to analyse them:

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/histogram_goals_nogoals_vs_distance.png" alt="histogram_goals_nogoals_vs_angle">
    <figcaption style="font-size: 15px;text-align: center;">Figure 1: Goals and no goals vs distance.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/histogram_goals_nogoals_vs_angle.png" alt="histogram_goals_nogoals_vs_angle">
    <figcaption style="font-size: 15px;text-align: center;">Figure 2: Goals and no goals vs angle.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/angle_vs_distance.png" alt="angle_vs_distance">
    <figcaption style="font-size: 15px;text-align: center;">Figure 3: angle vs distance.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/goal_rate_vs_distance.png" alt="goal_rate_vs_distance">
    <figcaption style="font-size: 15px;text-align: center;">Figure 4: Goal rate vs distance.</figcaption>
</figure>
<div style="text-align: justify"> All the figures above give us interesting information about shots and goals in the NHL. For example, figure 1 shows us that both goals and no goals happened more often closer to the net and that goals (in blue) are much less frequent than no goal (in orange). Figure 2, on the other hand, shows us that most shots (goal or not) are usually more aligned with the net since the angle is centered around 0. As for figure 3, it tells us that shots that are done farther from the net, generally have less of an angle with it (smaller angle). It also tells us that the two features are somewhat correlated with each other. Finally, if we analyse Figure 4, we can observe that when attacking players are very close to the opponent's net, the chance that they score is much higher which intuitively makes sense. However, it is surprising that the goal rate is so high for the farthest distances. This might be explained by a lower number of shots from this distance which could make the ratio of goal higher. This could also be due to a higher rate of empty net. It is sensible to think that it is easier to make a goal when a the net is empty, therefore, we evaluated the ratio of empty net each bins of distance and found that the farther the goal was made, the higher the ratio of empty net is (by a factor of about 10 between closer and farther bins).
 </div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/goal_rate_vs_angle.png" alt="goal_rate_vs_angle">
    <figcaption style="font-size: 15px;text-align: center;">Figure 5: Goal rate vs angle.</figcaption>
</figure>
<div style="text-align: justify">
From Figure 5 above, we can see that the goal rate is much higher when the shot is coming from the left and right side compared to when the shot comes from the center of the ice. This makes sense as goalies are much more vulnerable when shots come from the top of the circles (both left and right circles near the goalie).
<br>
<br>
Another interesting thing about Figure 5 is when we compare the goal rate from the left side to the right side. One reason why the goal rate is higher on the right side could be because the majority of NHL goalies have their glove on their left hand (maybe it's easier to stop shots with the glove in contrast to the blocker hand). Another reason might simply be because players shooting from the right side are much better than the ones shooting from the left side and, therefore, have a higher goal rate.
</div>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/histogram_goals.png" alt="Goals (empty net and non-empty net) vs distance">
    <figcaption style="font-size: 15px;text-align: center;">Figure 6: Goals (empty net and non-empty net) vs distance from net.</figcaption>
</figure>

<div style="text-align: justify">
Looking at the data from Figure 6, we can observe many interesting facts. Firstly, the vast majority of goals are on non-empty net which is logical since goalies are in the net most of the time. Secondly, we can observe that most of the goals are being scored within 60 feet from the net, which is inside the opponents' half of the rink. These two observations are aligned with our domain knowledge, and it makes perfect sense that the further you are from the opponents' net, the harder it is to score when there is a goalie in front of the net. With that said, the goals that were made from a distance of 150 feet when there was a goalie sound a bit unlikely.
<br>
<br>
We can observe in Figure 6 that the goals scored on a non-empty net from a distance of 150-170 feet are quite high. It could be that it has been originally misclassified as "non-empty net goals" as opposed to "empty-net goals". Another reason could be that these goals were scored by the other team that was then misclassified.
<br>
<br>
<i>Note about the empty_net feature: Even if we did the feature, the rest of the project does not use it because it would be a leaky feature as was discussed in piazza.</i>
</div>

### Question 3
> The code for this section is in features/baseline_models.py

#### Preparation of the data
<div style="text-align: justify">
After removing the test set from our data, we split the dataset into training and validation set using a stratified strategy, meaning that we kept
the same proportion of the classes into both sets. We kept 80% of the data for training and 20% for validation. We did not shuffle the
data, which means that the validation set is mostly composed of the later seasons (2018). We also removed all of the rows that had nan values in either <i>distance_net</i> or
<i>angle_net</i> columns.
</div>

#### Results
<div style="text-align: justify">
For our baseline, we trained a Logistic Regression model using only the <i>distance</i> feature that we have previously extracted from the raw data, and it gave us a <b>90.59</b> accuracy when we ran it on our validation set. We also generated the following confusion matrix to have a better look at our model's results:
</div>

| Target/Prediction | **Class 0 (not goal)** | **Class 1 (goal)** |
| :-------: | :-------: | :-------: |
| **Class 0 (not goal)** | 70748 | 0 |
| **Class 1 (goal)** | 7344 | 0 |

<div style="text-align: justify">
This confusion matrix clearly shows us that there is a major issue with our predictions. We are only getting high accuracy performance because the majority of our data points are classified as a <b>not goal</b>. By always predicting <b>not goal</b> our model does a pretty good job if we only look at the overall accuracy. This is due to the fact that our dataset is extremely unbalanced with 99% of the data as a non goal and only 1% as goals.
<br>
<br>
Following this, we tried 2 other models: one with the angle feature and the other with both angle and distance. These did not give us better results as we shall see in the next section.
</div>

#### Analysis
<div style="text-align: justify">
To give us an idea of the performance of our baselines, we added a curve to all of the following figures of a random model which randomly decides a probability between 0 and 1 for the example.
<br>
<br>
Figure 7 shows the ROC curve for all of the baseline models which is the receiver operating characteristic curve (ROC). A ROC curve plots the true positive rate against the false positive rate for different threshold of probabilities. For example, if we were to select a threshold of 0.3, we would compute how many true positive and false positive our models gives when setting 0.3 as its threshold for deciding the label. This would be one data point on the curve. The whole ROC curve is then generated for many threshold between 0 and 1. An ideal model would have a curve that tends to be closer to the upper left corner of the plot. The AUC also presented in the legend is the area under the curve of the ROC. The higher it is, the better. <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic">ROC Wikipedia</a>
<br>
<br>
As we can see in Figure 7 below, the random classifier gives a perfect diagonal as expected. We can also observe that our model trained on both of our features gives the best curve. Our ROC score is also higher (<b>area=0.70</b>) when we trained our model on both features compared to training solely on angle. However, the difference isn't big with only distance. This means that the logistic regression does not learn much from the angle feature.
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/roc_curve_baseline.png" alt="roc_curve_baseline">
    <figcaption style="font-size: 15px;text-align: center;">Figure 7: Logistic Regression: ROC curve.</figcaption>
</figure>

<div style="text-align: justify">
The results shown in Figure 8 is about shot probabilities. It shows the goal rate for different percentiles of the probability. The way to interpret this plot is the higher the percentile, the higher the goal rate should be since a higher percentile means a higher probability and a higher probability means a higher chance that the event is a goal. If our model was good, a high probability should mean a higher confidence that this is in fact a goal.
<br>
<br>
Knowing this, the plot shows us that our trained models perform much better that the random classifier at predicting the shot probability. As in our previous analysis, because of the shape of the curve which is higher at higher percentile, our model that was trained on both features (distance and angle) does give us better results than models trained on the features separately. Once again, the model trained solely on the angle feature performs poorly since the curve is somewhat constant.
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/goal_rate_curve_baseline.png" alt="goal_rate_curve_baseline">
    <figcaption style="font-size: 15px;text-align: center;">Figure 8: Logistic Regression: Goal rate.</figcaption>
</figure>

<div style="text-align: justify">
Figure 9 shows the cumulative sum of goal against the probability model percentile (the same one as for the last plot). For this plot, we should pay attention the slope of the curve. The slope should be very steep in the beginning and slowly get flatter near the lowest percentile. If we want to have confidence in our model, most of the goals should have a high probability and therefore be associated with a higher percentile and this is why we want a steep then gentle slope.
<br>
<br>
Fortunately, the main thing we can observe is that shots that have a higher probability represents a much greater proportion of the total goals scored compared to shots with lower probabilities. Once again, even though the model trained on the distance feature and the model trained on the angle feature are better than the random baseline, the model that we trained on both features (distance and angle) gave us better results. Meaning that it is much better at predicting the probability that a shot would turn out to be a goal.
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/cumulative_sum_goal_baseline.png" alt="cumulative_sum_goal_baseline">
    <figcaption style="font-size: 15px;text-align: center;">Figure 9: Logistic Regression: Goal proportion.</figcaption>
</figure>

<div style="text-align: justify">
Finally, the last plot shows us the calibration curve. A good calibration curve should be linear and should have a slope of 1. The interpretation is that if your model gives you a probability of 0.8 for a shot to be a goal, then 80% of the example that have a probability of 0.8 should be positives. <a href="https://scikit-learn.org/stable/modules/calibration.html#probability-calibration"> Explanation of calibration curve from scikit-learn </a>
<br>
<br>
Given the calibration curve shown in Figure 10, we can easily see that our trained models did learn some valuable representations of our data. Comparing all our current models, the model that was trained on both features (distance and angle) has the closest calibration values to the <b>perfectly</b> calibrated model. Again, as mentioned before, it confirms that overall this model is the model that gives us the best results so far. However, it is important to note that the curve do not go beyond 0.25 which is due to the fact that the probability predicted by our models are all below 0.25%. This means that our models are not very confident about their predictions.
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/calibration_curve_baseline.png" alt="calibration_curve_baseline">
    <figcaption style="font-size: 15px;text-align: center;">Figure 10: Logistic Regression: Calibration curve.</figcaption>
</figure>


#### Links to our models

1. [Logistic Regression on distance and angle](https://www.comet.ml/jaihon/ift6758-project/88c175fd9d3c4892acf334fcfdb4a6d0)
2. [Logistic Regression on distance](https://www.comet.ml/jaihon/ift6758-project/6997fdfbdc76426db60408591e58ac5a)
3. [Logistic Regression on angle](https://www.comet.ml/jaihon/ift6758-project/934baca85c9448c997d8d0727845db65)



### Question 4
> The code for this section is in project/ift6758/data/EventGenerator.py and project/ift6758/features/feat_eng2.py

<div style="text-align: justify">
We added below a list of all of the features that we created for this section, and we listed each feature by both the column name
in the dataframe and a simple explanation. For the novel features, we describe what they are.
At the end, we added a link to the experiment which stores the filtered DataFrame.
</div>

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
| previous_event_period | period of the game during which the last event happened |
| previous_event_period_time | total sum of seconds elapsed in the game for the last event |
| shot_last_event_delta | time elapsed since the last event |
| shot_last_event_distance | distance calculated from the last event |
| Rebound | Rebound of the last event (True if shot, otherwise False) |
| Change_in_shot_angle | change in the shot angle if the shot is a rebound |
| Speed | defined as the distance from the previous event, divided by the time since the previous event |
| time_since_pp_started |  time in seconds since the penalty started |
| current_friendly_on_ice | Number of friendly players on ice|
| current_opposite_on_ice | Number of opposite players on ice|

Some of these features above were mostly used to generate other features like previous_event_period.

<div style="text-align: justify">
In the bonus question, we added a few more features like the time since the penalty started and the number of friendly and opposite players on ice. For all of these, we started
by generating all types of events in our game. For each event, we evaluated if there was a
penalty and checked on which side the team was. We then built a tidy event object that gave the time and coordinates details
relative to the previous event. To compute the time since the penalty started, we got the current event time and subtracted
the starting time of the penalty from the current time (two types of penalties generated).
To get the number of friendly players on ice and the number of opposite players on ice, we first checked the side of the team to figure out who is friendly and who is not and
then subtracted the number of players lost depending on the type of the penalty from 5.
</div>


#### Link to the experiment which stores the filtered DataFrame artifact:
[wpg_v_wsh_2017021065.csv](https://www.comet.ml/jaihon/ift6758-project/fae888ad53de4d1aa940a67b96d106ab?assetId=e46feef96edc4bf8afe7c676f05c192b&assetPath=dataframes&experiment-tab=assets)

### Question 5
> The code for this section can be found in project/ift6758/models/xgboost_models.py
<div style="text-align: justify">
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
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/roc_curve_xgboost.png" alt="roc_curve_xgboost">
    <figcaption style="font-size: 15px;text-align: center;">Figure 11: XGBoost: ROC curve.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/goal_rate_xgboost.png" alt="goal_rate_xgboost">
    <figcaption style="font-size: 15px;text-align: center;">Figure 12: XGBoost: goal rate.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/cumulative_goals_xgboost.png" alt="cumulative_sum_xgboost">
    <figcaption style="font-size: 15px;text-align: center;">Figure 13: XGBoost: cumulative sum.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/calibration_xgboost.png" alt="calibration_xgboost">
    <figcaption style="font-size: 15px;text-align: center;">Figure 14: XGBoost: calibration curve.</figcaption>
</figure>

<div style="text-align: justify">
For the next xgboost model, we started by standardizing all the numerical features and remove the rows which had nan values.
For the particular case of the <i>Speed</i> column, we had to impute a new value for the rows which had infinity values. To do so,
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
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/n_estimators_vs_metrics_xgboost.png" alt="number_of_estimators_vs_metrics">
    <figcaption style="font-size: 15px;text-align: center;">Figure 15: Relation between the number of estimators and different metrics.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/max_depth_vs_metrics_xgboost.png" alt="max_depth_vs_metrics">
    <figcaption style="font-size: 15px;text-align: center;">Figure 16: Relation between the max depth of a tree and different metrics.</figcaption>
</figure>
<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/reg_lambda_vs_metrics_xgboost.png" alt="reg_lambda_vs_metrics">
    <figcaption style="font-size: 15px;text-align: center;">Figure 17: Relation between the regularisation coefficient lambda (l2) and different metrics.</figcaption>
</figure>

<div style="text-align: justify">
Once we finished exploring some hyperparameters, we did a randomized search over the hyperparameters space focusing
on the same parameters as before as well as the learning rate. We did not do a grid search because it would have taken
too long to search all the parameters we wanted to try, but also because the hyperparameters do not have such a big influence
on the results as was shown on the figures above. Since the accuracy is almost constant for all parameters, we decided to
focus on the ROC area under the curve. The best model was selected with the highest ROC AUC. The resulting model was
found to have the following hyperparameters:
</div>

|      Hyperparameter     | Value |
|:-----------------------:|:-----:|
|     # of estimators     |  300  |
|      Maximum depth      |   5   |
| Regularisation &lambda; |   0   |
|      Learning rate      |  0.03 |

<div style="text-align: justify">
It is the orange curve on figures 11 to 14. Even if this is the <i>best</i> model, the one right below in terms of performance
was very close to it with a ROC AUC of 0.75 compared to the model presented here which has 0.76.
<br>
<br>
As for our last experience, we tried some feature selection. We tried two different methods: lasso and mutual information.
For the lasso one, we evaluated a lasso linear model on our training dataset and selected the features that had the highest
coefficient of importance, i.e. they were higher than the median. The features selected are shown in red on figure 18 which
presents the importance coefficient according to the lasso model for all of our features. The features that are in parentheses
are the categorical data that was encoded using one-hot vectors. We can see that some values in categorical data seem to be
"useless" like the type of period in which the shot happens (overtime vs regular). Interestingly, we noticed that knowing the side of the
team that is shooting (right vs left) seems to be more important when it is left. Not too surprisingly, some type of shots seem to
have a bigger impact like <i>deflected</i> or <i>tip-in</i> or <i>wrap-around</i> compared to wrist shot. If we recall from the previous milestone,
we concluded that <i>deflected</i> and <i>tip-in</i> were among the most dangerous shots which seems to agree with this. These are some of
the interesting observations that we found from this figure.
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:100%;height:100%;">
    <img src="/public/milestone2/importance_feature_lasso_xgboost.png" alt="importance_feature_xgboost">
    <figcaption style="font-size: 15px;text-align: center;">Figure 18: Importance of each feature according to lasso.</figcaption>
</figure>

<div style="text-align: justify">
Anyway, once we selected the features in red, we trained our model using a randomized search again on the same hyperparameters as before.
We found a model that was performing similarly as our best one trained on all features as we can see by comparing the green (lasso trained one) and orange
curves on figures 11 to 14.
<br>
<br>
For the mutual information model, we did a randomized search on the same hyperparameters as before, but with the addition of
choosing a number of features according to the mutual information. Our best model using this trained on 15 features. A comparison
of the chosen features between the lasso model and mutual information score is shown in the following table where only
the features selected by either one of them is presented:
</div>

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

<div style="text-align: justify">
From this table, we can observe that both method seem to agree on some features like the coordinate x, the distance or the
speed. However, the interesting features are the one selected by the mutual information, but not the lasso since the lasso took
more. Among those are the types of shots, lasso take all of the types except the wrist shot, but mutual information does the complete
opposite and select only the wrist shot. Mutual information also takes the period type regular into account whereas lasso
completely ignore them.
<br>
<br>
The model resulting from training with mutual information is shown in red in the figures 11 to 14. We can see that it does not
perform as well as the models trained on all features and on the lasso selected features on figures 11 to 13 where the curves are
closer to the default XGBoost model.
<br>
<br>
We want to note that all of the curves on figure 14 start of pretty linear which is good because it follows the perfectly
calibrated model.
<br>
<br>
After inspecting figures 11 to 14, we concluded that the best model was the one trained on all features because it has the
highest ROC AUC, but also because its calibration curve seems to be the most linear one. Indeed, we can see that the orange
curve varies less than the others after around 0.5 probabilty.
</div>

The links to the different experiments shown in this section can be found here:
1. [XGBoost trained on distance and angle with default hyperparameters](https://www.comet.ml/jaihon/ift6758-project/20c76cb9d81541b5ae0d2b320e59f59f)
2. [XGBoost with hyperparameter tuning and trained on all features](https://www.comet.ml/jaihon/ift6758-project/05c804186d274b0c8955cbb14b1a66b3)
3. [XGBoost with hyperparameter tuning and trained on subset of features selected by Lasso](https://www.comet.ml/jaihon/ift6758-project/74f19b8137e14336ba1e49c198dfd3e6)

## Question 6: Best Shot
> The code for this section can be found in project/ift6758/models/best_shot.py, KNN_model.py, Optimised_best_shot.py

<div style="text-align: justify">
After using a logistic regression and XGBoost model, we decided to try other algorithms to find the best model
for our task of binary classification with our collected dataset. We have made many experiences in this milestone.
They are all available on comet_ml.
<br>
<br>
We decided to try K-Nearest Neighbors, Random Forest and a feed-forward neural network.
As already pointed out in section 2, our dataset is very unbalanced. Therefore, classification accuracy and its complement, the error
rate, might be a bad idea to use because it will be an unreliable measure of the model performance. We have have what is called an "Accuracy paradox"(3).
In that case, a good performance on the minority class (Goal) will be preferred over a good performance on both class.
In order to do so, alternative performance metrics, like precision, recall or the F-measure, may be required since reporting the classification accuracy may be misleading.
In the following section, we will explain how the features were processed for each model, how they were trained and which metrics were used.
All our models have been split into training (80%) and validation (20%) set using a stratified strategy and have been optimized using cross-validation
to find the best hyperparameters. The figures shown in this section have been obtained after evaluating our models on the validation set.
</div>

### Models Results and Analysis

#### 1. KNN

<div style="text-align: justify">
KNN was trained on all the features created in section 4. For the preprocessing, we started by changing all of the
categorical features into one-hot encoding.
We then dropped the rows that had nan values and removed the ones that had infinity values (like the Speed Column).
Once this was done, we split the dataset into training and validation set as specified above.
We trained our KNN using a GridSearch on different hyper-parameters: the number of neighbors and the weights used for
prediction(distance vs uniform). The best model found by the GridSearch used 8 neighbors and distance for weights.
However, even if the GridSearch did find a "best estimator", it was only able to reach an AUC of 0.63 which is lower
than the XGBoost model and our best regression model from section 3.
</div>

#### 2. Random Forest
<div style="text-align: justify">
The Random Forest had a similar preprocessing as the KNN, i.e. we used the same features with the one-hot encoding for the categorical features, etc.
We also trained the Random Forest using a GridSearch over 2 hyper-parameters: the criterion which is the function used to measure the quality of a split
in a tree and the number of estimators.
[<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor">Random Forest from Scikit-Learn</a>]
This time, the GridSearch finished with an AUC of 0.72 for the cross-validation results which is similar to what our XGBoost could do, but a little bit lower.
</div>

#### 3. Neural Network
<div style="text-align: justify">
For the neural network, different methods were used for the preprocessing. We tried first a model that did not use our
features created in the bonus part of section 4 (everything related to penalty). This means that our model had less features
than the others tried as of now (except for the linear regression of course). Still, for all neural networks models, we transformed the
categorical data into one-hot vectors just like we did we XGBoost, KNN and Random Forest.
However, we considered the feature current_friendly_on_ice and current_opposite_on_ice as categorical features as opposed to the other models
which supposed they were numerical and therefore were standardized which does not make much sense for a number of people on the ice that ranges from 1 to 5.
As the other models, we standardized the numerical features.
<br>
<br>
The other two models tried with the Neural Network differed on the used of dropouts. They both trained with all of the features created in section 4 (including the bonus),
but one of them used dropout and the other didn't.
For the training, we did some manual cross validation over different hyper-parameters like the learning rate, the coefficient for the Adam optimizer and the number of epochs.
All the hyper-parameters of the 3 models are the same, because those parameters had been optimized
previously. So mainly, the learning rate is 0.001, the Adam coefficient is 0.9, and we trained for 30 epochs.
So, to reiterate, the main difference is that the 'nn_no_bonus_feature' has no feature developed during the bonus part (section 4),
the 'nn_no_dropout' has no dropout and the "best_shot_nn_final" has all of the features and dropout.
<br>
<br>
Not presented here, but we did train a model without standardizing the numerical features, but found that the performance was better if standardization was done.
<br>
<br>
Here is a list of the features selected to train our neural network which we selected based on our domain knowledge:
</div>

##### Selected Features of the Neural Networks

| Feature     | Encoding |
| ----------- | ----------- |
| side | one-hot |
| shot_type | one-hot |
| period | one-hot |
| period_type | one-hot |
| coordinate_x | no encoding |
| coordinate_y | no encoding |
| distance_net | no encoding |
| angle_net | no encoding |
| previous_event_type | one-hot |
| previous_event_x_coord | no encoding |
| previous_event_y_coord | no encoding |
| previous_event_time_seconds | no encoding |
| time_since_pp_started | no encoding |
| current_time_seconds | no encoding |
| current_friendly_on_ice | one-hot |
| current_opposite_on_ice | one-hot |
| shot_last_event_delta | no encoding |
| shot_last_event_distance | no encoding |
| Change_in_shot_angle | no encoding |
| Speed | no encoding |
| Rebound | one-hot |

#### Threshold selection

<div style="text-align: justify">
Because the dataset was very unbalanced in nature, we decided to mainly use the F1 Score for all models as well as the AUC.
In addition, we also used a custom-made threshold technique to help us analyse the results of our models. Since our models' outputs had very small
probability values, we decided that the 0.5 threshold for binary prediction wasn't the way to go. Instead, for each model
we trained, we found a better threshold value that would give us the optimal F1 score at the end. To do so, we simply checked the performance of the F1 score
of our models on the training set for different threshold and took the one that gave the highest one.
</div>

#### Results and Analysis

<div style="text-align: justify">
The following figures present the different curves (ROC, goal rate, proportion and calibration) obtained on the models presented
in this section as well as their confusion matrices.
<br>
<br>
We can see right away on all figures that the Random Forest and KNN seems to strangely be extremely good at predicting goals. Indeed, our calculated AUC on graph is actually 0.94 for the KNN,
which tells us that there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. (1) This comes as
a surprise to us since the KNN performed poorly during training and the Random Forest was no better than the best XGBoost model.
This is very suspicious behaviour. We tried to investigate and see if we made a mistake, but did not find any obvious one. Even if they seem
to perform extremely well on the validation set, we doubt their actual performance generalizes well considering the training.
<br>
<br>
For the neural networks, the best AUC on graph is actually the NeuralNetwork with an AUC of 0.77 which corresponds to the
model trained with the bonus_features, the dropout and standardization. However, the performance was pretty equal
to the other two models (NeuralNetwork_no_bonus with AUC 0.75) trained with no bonus features, and NeuralNetwork_no_dropout (AUC=0.76)
trained with no dropout. This means that we could think that using the dropout technique is slightly helpful, but we can't be certain as the values and curves are very close to each other.
The same can be said about including or not the features from the bonus section. Thus, we can see that at a high level, our models had pretty much the same performances.
Nevertheless, we conclude that our best model for this section is the neural network trained using all the features developed in section 4 as well as dropout and leave the Random Forest and KNN model for the reasons mentioned before.
<br>
<i>For more information about the performance of our Neural Networks, we put the F1 score in a table in the annex.</i>
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:100%;height:100%;">
    <img src="/public/milestone2/confusion_matrix_val.png" alt="confusion_matrix_validation">
    <figcaption style="font-size: 15px;text-align: center;">Figure 19: Confusion matrices on the validation set.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/roc_curve_val.png" alt="roc_curve_validation">
    <figcaption style="font-size: 15px;text-align: center;">Figure 20: ROC curve on the validation set.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/goal_rate_percentile_val.png" alt="goal_rate_percentile_validation">
    <figcaption style="font-size: 15px;text-align: center;">Figure 21: Goal Rate on the validation set.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/proportion_percentile_val.png" alt="proportion_percentile_validation">
    <figcaption style="font-size: 15px;text-align: center;">Figure 22: Goal proportion on the validation set.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/calibration_val.png" alt="calibration_validation">
    <figcaption style="font-size: 15px;text-align: center;">Figure 23: Calibration on the validation set.</figcaption>
</figure>

##### Links to our models

1. [Neural Network - best_shot_nn_final](https://www.comet.ml/jaihon/ift6758-project/f02e46ac553944f7ba18060044d873e9?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
2. [Neural Network - nn_no_bonus_feature](https://www.comet.ml/jaihon/ift6758-project/f22281d6264d462685c13628a0dd7daa?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
3. [Neural Network - nn_no_dropout](https://www.comet.ml/jaihon/ift6758-project/b086d3049e1f47b7ae8aa569994983b4?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
4. [Random Forest](https://www.comet.ml/jaihon/ift6758-project/f4b6196482dc4e1c9c6ab32034bf2278)
5. [KNN](https://www.comet.ml/jaihon/ift6758-project/fc093cf3ac61416391c5f3fca4416117)

## Question 7: Evaluation
> The code for this section can be found in project/ift6758/models/evaluation.py

### Selected models

<div style="text-align: justify">
In order to test our models on the test dataset (regular and playoff games from 2019-2020 season), we have selected our best performing model (on the validation set) in the best-shot section, our three baseline models and our XGBoost model. Shown below are the models that we have selected for this section.
</div>


| **Models** | **Description** |
| :-------: | :-------: |
| LogisticRegression 1 | Model from baseline section. |
| LogisticRegression 2 | Model from baseline section. |
| LogisticRegression 3 | Model from baseline section. |
| XGBoost | Best resulting model from xgboost section.  |
| NeuralNetwork | Best resulting neural network from best-shot section. |

### Results and Analysis


#### Regular Season Games

<div style="text-align: justify">
For this section, we have selected as per requested, the 3 Logistic Regression models as well as the best XG_boost model and our overall best performing model (on the validation set), a Neural Network model. First, let us take a look at the overall performance of these models on the test set composed of the games from the regular season of 2019-2020. At first glance, we can clearly distinguish 3 models that stand out from the other 2 by taking a look at figure 25. They are, unsurprisingly, the Neural Network, XG Boost and Logistic Regression #3 models. As the other Logistic Regression models (1 and 2) were only using 1 feature each (angle_net, distance_net respectively), it is expected that they would perform relatively poorly based on such restricted information. It is interesting to note that although our two best models are XG Boost and the Neural Network model, the Logistic Regression #3 does not seem too far behind in terms of AUC score (0.76, 0.75, 0.69 respectively). It is important to note that, since the confidence interval of these measures are not displayed, we are not able to say with confidence that they perform similarly or not. Alghough intuitively, it does seem that XG boost and the Neural Network perform better. By looking at the confusion matrix (figure 24), one could think that since the recall of the XG boost and the Logistic Regression #3 model are essentially 0 (0 True positives), these models should have a similar AUC score as the 2 worst models (Log.Regres #1 and #2). This would be a wrong assumption though. As the default threshold for these models is 0.5, they do indeed all perform quite terribly at this threshold setting. Nevertheless, the ROC score enables the evaluation of the predictive performance of the model at different thresholds and it is clear that the Logistic Regression 3 and the XG Boost models perform significantly better than the two bottom tier models at a specific threshold. In fact, the XG boost model performs at similar levels than the neural network at a given threshold. If we compare the overall performance of all models between the test (figure 25) set and the validation set (figure 20), we clearly see very similar AUC scores. This reflects the rather robust generalisation of our models, at least on similar data (in terms of game types). By looking at the goal rate figure (figure 26) we are able to evaluate the variation of goal proportion in fonction of the percentile of the model probability. For instance: Is the highest probability percentile associated with increased goal / goal+shot (we will refer to this as :ratio)? In a way, this reflects the certainty of our models. We clearly dinstinguish 3 models that behave in a way that one would expect a decent model to perform. As such, our 3 top models have a higher ratio in the upper percentiles (100-80 for instance) compared to the low tier models. More importantly, this ratio decreases with the decrease of percentile, thus reflecting the fact that the smaller the probability is of scoring, the lower the ratio will be. Basically, our best models have less uncertainty than our low tier models. It is interesting to note that although the Logistic Regression #3 model performs relatively well, XG Boost and the Neural Network models seem to have less uncertainty, at least, at the higher percentiles (20%, 40% respectively). Another way to look at this is by looking at the cumulative goals in fonction of the shot probability percentile (Figure 27), where a good model would be a logarithmic curve, basically translating the fact that less and less goals will be scored at lower probability percentiles. Once again, these tendencies are very similar between our test set (regular season 2019-2020) and our training data. Lastly, the calibration graph (figure 28)enables one to determine if the predicted probabilities are linearly correlated with proportion of positive predictions and therefore reflects the confidence of the models predictions. A perfect model would have x = y as mentioned before. We can observe a positive correlation between x and y for our 2 top tier models (XG Boost and Neural Network), but not quite a perfect linear correlation. Concerning the remaining models, since their prediction distribution was centered around a very restricted interval, we did not obtain an interpretable visual. Except for the realisation that these models were incapable of producing a wide interval of predicted probabilities.
</div>
<figure style="display: block;margin-left: auto; margin-right: auto;width:100%;height:100%;">
    <img src="/public/milestone2/conf_matrix_regular.png" alt="conf_matrix_regular">
    <figcaption style="font-size: 15px;text-align: center;">Figure 24: Confusion Matrices Regular Season (2019-2020)</figcaption>
</figure>


<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/roc_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 25: ROC Regular Season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/goal_rate_vs_percentile_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 26: Goal rate Regular Season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/proportion_goal_percentile_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 27: Goal rate cumulative Regular Season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/calibration_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 28: Calibration results Regular Season (2019-2020)</figcaption>
</figure>


#### Playoff Games

<div style="text-align: justify">
We saw that our models mostly generalize very well on regular season games when we compared them to our validation set results. However, we can clearly see that our models perform poorly on playoff games. Throughout Figure 29 to Figure 33, we can see that our same models perform a little bit worse on playoff games compared to regular season games. This was expected since we have trained our models only on regular season games. Also, the playoff games are fundamentally different from the regular season games ("Anything can happen in the playoffs!"). The reason for this is that the playoff games are elimination games, which means that the teams and players always play with heavy pressure. In addition, the environment and the "energy" is different during the playoffs, which ultimately could lead to features that we simply don't have access to.
<br>
<br>
It would be interesting to see if adding external features such as a metric of the crowd energy (Noise volume) would help our models to generalize. Another interesting thing would be to train our models on playoff games instead of regular season games to see if it can generalize better on both regular season and playoff games.
</div>

<figure style="display: block;margin-left: auto; margin-right: auto;width:100%;height:100%;">
    <img src="/public/milestone2/conf_matrix_playoffs.png" alt="conf_matrix_playoffs">
    <figcaption style="font-size: 15px;text-align: center;">Figure 29: Confusion Matrices Playoffs season (2019-2020)</figcaption>
</figure>


<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/roc_playoffs.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 30: ROC Playoffs season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/goal_rate_vs_percentile_playoff.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 31: Goal rate Playoffs season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/proportion_goal_percentile_playoffs.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 32: Goal rate cumulative Playoffs season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:75%;height:75%;">
    <img src="/public/milestone2/calibration_curve_playoffs.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 15px;text-align: center;">Figure 33: Calibration results season (2019-2020)</figcaption>
</figure>

# Bibliography:

1. Bhandari, Aniruddha , "AUC-ROC Curve in Machine Learning Clearly Explained" <a href="https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/">https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/</a>, June 16, 2020
2. Takaya Saito, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets",  <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/">https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/</a>
3. Jason Brownlee, January 1, 2020 , "Failure of Classification Accuracy for Imbalanced Class Distributions", <a href="https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/">https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/</a>

# Annex
<table>
    <caption style="caption-side: bottom; font-size: small;">F1 score results for our Neural Network models</caption>
    <tr>
        <th scope="row">Model</th>
        <th scope="col">F1 Score (Class 0)</th>
        <th scope="col">F1 Score (Class 1)</th>
    </tr>
    <tr>
        <td>NeuralNetwork</td>
        <td>0.90</td>
        <td>0.32</td>
    </tr>
    <tr>
        <td>NeuralNetwork_no_dropout</td>
        <td>0.90</td>
        <td>0.31</td>
    </tr>
        <tr>
        <td>NeuralNetwork_no_bonus</td>
        <td>0.89</td>
        <td>0.31</td>
    </tr>
</table>
