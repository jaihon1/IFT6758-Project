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
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Logistic Regression: ROC rate.</figcaption>
</figure>

Given the calibration curve shown in Figure 10, we can easily see that our trained models did learn some valuable representations of our data. Comparing all our current models, the model that was trained on both features (distance and angle) has the closest calibration values to the *perfectly* calibrated model. Again, as mentioned before, it confirms that overall this model is the model that gives us the best results so far.

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/calibration_curve_baseline.png" alt="calibration_curve_baseline">
    <figcaption style="font-size: 12px;text-align: center;">Figure 10: Logistic Regression: Calibration cruve.</figcaption>
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


## Question 6: Best Shot

Brief intro on what and why we decide to use in this part...

### Model Results and Analysis

#### 1. KNN

#### 2. Neural Network

Using standardization techniques, we can see that our neural network model performs better than without.

Using very value of a small dropout technique to help prevent overfitting, we can see that our neural network model performs better with this regularization technique.

Including features developed in the bonus question (current_time_seconds, time_since_pp_started, current_friendly_on_ice, current_opposite_on_ice), we can see that our neural network model performs better.

##### Selected Features
Feature Selection technique: Domain knowledge
Using feature selection, we selected the following features to train our neual network models:

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


##### Results and Analysis

Because the dataset was very unbalanced in nature, we decide to mainly use the F1 Score. In addition, we also used a custom made threshold technique to help us analyse the results of our models. Since our model's outputs were very small probability values, we decided that the 0.5 threshold for binay prediction wasn't the way to go. Instead, for each model we trained, we found a better threshold value that would give us the optimal F1 score at the end.

<table>
    <caption style="caption-side: bottom; font-size: small;">F1 score results for our Neural Network models</caption>
    <tr>
        <th scope="row">Model</th>
        <th scope="col">F1 Score (Class 0)</th>
        <th scope="col">F1 Score (Class 1)</th>
    </tr>
    <tr>
        <td>best_shot_nn_final</td>
        <td>0.90</td>
        <td>0.32</td>
    </tr>
    <tr>
        <td>unnecessary_truss_2939</td>
        <td>0.90</td>
        <td>0.31</td>
    </tr>
        <tr>
        <td>separate_alfalfa_7886</td>
        <td>0.89</td>
        <td>0.31</td>
    </tr>
</table>

<table>
    <caption style="caption-side: bottom; font-size: small;">Confusion matrix results for best_shot_nn_final on validation set</caption>
    <tr>
        <th scope="row">Target/Prediction</th>
        <th scope="col">Class 0 (not goal)</th>
        <th scope="col">Class 1 (goal)</th>
    </tr>
    <tr>
        <th scope="row">Class 0 (not goal)</th>
        <td>46513</td>
        <td>7228</td>
    </tr>
    <tr>
        <th scope="row">Class 1 (goal)</th>
        <td>2930</td>
        <td>2413</td>
    </tr>
</table>

<table>
    <caption style="caption-side: bottom; font-size: small;">Confusion matrix results for unnecessary_truss_2939 on validation set</caption>
    <tr>
        <th scope="row">Target/Prediction</th>
        <th scope="col">Class 0 (not goal)</th>
        <th scope="col">Class 1 (goal)</th>
    </tr>
    <tr>
        <th scope="row">Class 0 (not goal)</th>
        <td>46712</td>
        <td>7029</td>
    </tr>
    <tr>
        <th scope="row">Class 1 (goal)</th>
        <td>3091</td>
        <td>2252</td>
    </tr>
</table>

<table>
    <caption style="caption-side: bottom; font-size: small;">Confusion matrix results for separate_alfalfa_7886 on validation set</caption>
    <tr>
        <th scope="row">Target/Prediction</th>
        <th scope="col">Class 0 (not goal)</th>
        <th scope="col">Class 1 (goal)</th>
    </tr>
    <tr>
        <th scope="row">Class 0 (not goal)</th>
        <td>45213</td>
        <td>8528</td>
    </tr>
    <tr>
        <th scope="row">Class 1 (goal)</th>
        <td>2814</td>
        <td>2529</td>
    </tr>
</table>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/best_shot_curves/best_shot_nn_roc.png" alt="best_shot_nn_roc">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Neural Network: ROC.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/best_shot_curves/best_shot_nn_goal_rate.png" alt="best_shot_nn_goal_rate">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Neural Network: Goal rate.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/best_shot_curves/best_shot_nn_cumulative.png" alt="best_shot_nn_cumulative">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Neural Network: Goal rate cumulative.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/best_shot_curves/best_shot_nn_calib.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Neural Network: Calibration.</figcaption>
</figure>

##### Links to our models

1. [Neural Network - best_shot_nn_final](https://www.comet.ml/jaihon/ift6758-project/f02e46ac553944f7ba18060044d873e9?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
2. [Neural Network - unnecessary_truss_2939](https://www.comet.ml/jaihon/ift6758-project/f22281d6264d462685c13628a0dd7daa?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
3. [Neural Network - separate_alfalfa_7886](https://www.comet.ml/jaihon/ift6758-project/b086d3049e1f47b7ae8aa569994983b4?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)



#### 3. Random Forest






## Question 7: Evaluation

For this section, we have selected as per requested, the 3 Logistic Regression models as well as the XG_boost model and our overall best performing model (on the validation set), a Neural Network model. First, let us take a look at the overall performance of these models on the test set composed of the games from the regular season of 2019-2020. At first glance, we can clearly distinguish 3 models that stand out from the other 2 by taking a look figure X!!!!!!!!!!!!!!!!!. They are, unsurprisingly, the Neural Network, XG Boost and Logistic Regression 3 models. As the other Logistic Regression models (1 and 2) were only using 1 feature each (angle_net, distance_net respectively), it is expected that they would perform relatively poorly based on such restricted information. It is intersting to note that although our two best models are XG Boost and the Neural Network model, the Logistic Regression 3 does not seem too far behind in terms of AUC score (0.76, 0.75, 0.69 respectively). It is important to note since the confidence interval of these measures are not displayed, we are not able to say with confidence that they perform similarly or not. Alghough intuitively, it does seem that XG boost and the Neural Network perform better. By looking at the confusion matrix (figure X!!!!!!!!!!!!!!!!!), one could think that since the recall of the XG boost and the Logistic Regression 3 model are essentially 0 (0 True positives), these models should have a similar AUC score as the 2 worst models (Log.Regres 1 and 2). This would be a wrong assumption though. As the default threshold for these models is 0.5, they do indeed all perform quite terribly at this threshold setting. Nevertheless, the ROC score enables the evaluation of the predictive performance of the model at different thresholds and it is clear that the Logistic Regression 3 and the XG Boost models perform significantly better that the two bottom tier models at a specific threshold. In fact, the XG boost model performs at similar levels than the neural network at a given threshold. If we compare the overall performance of all models between the test (figure X!!!!!!!!!!!!!) set and the validation set (figure X!!!!!!!!!!!!!!!), we clearly see very similar AUC scores. This reflects the rather robust generalisation of our models, at least on similar data (in terms of game types). By looking at the goal rate figure (figure X!!!!!!!!!!!) we are able to evaluate the variation of goal proportion in fonction of the percentile of the model probability. For instance: Is the highest probability percentile associated with increased goal / goal+shot (we will refer to this as :ratio)? In a way, this reflects the certainty of our models. We clearly dinstinguish 3 models that behave in a way that one would expect a decent model to perform. As such, our 3 top models have a higher ratio in the upper percentiles (100-80 for instance) compared to the low tier models. More importantly, this ratio decreases with the decrease of percentile, thus reflecting the fact that the smaller the probability is of scoring, the lower the ratio will be. Basically, our best models have less uncertainty than our low tier models. It is interesting to note that although the Logistic Regression 3 model performs relatively well, XG Boost and the Neural Network models seem to have less uncertainty, at least, at the higher percentiles (20%, 40% respectively). Another way to look at this is by looking at the cumulative goals in fonction of the shot probability percentile (Figure X!!!!!!!!!!!), where a good model would be a logarithmic curve, basically translating the fact that less and less goals will be scored at lower probability percentiles. Once again, these tendancies are very similar between our test set (regular season 2019-2020) and our training data. Lastly, the calibration graph (figure X!!!!!!!!!!!!!)enables one to determine if the predicted probabilities are linearly correlated with proportion of positive predictions and therefore reflects the confidence of the models predictions. A perfect model would have x = y. We can observe a positive correlation between x and y for our 2 top tier models (XG Boost and Neural Network), but not quite a perfect linear correlation. Concerning the remaining models, since their prediction distribution was centered around a very restricted interval, we did not obtain an interpretable visual. Except for the realisation that these models were incapable of producing a wide interval of predicted probabilities.

### Selected models

In order to test our models on the test dataset (regular and playoff games from 2019-2020 season), we have selected our best performing model (on the validation set) in the best-shot section, our three baseline models and our XGBoost model. Shown below are the models that we have selected for this section.

| **Models** | **Description** |
| :-------: | :-------: |
| LogisticRegression 1 | Model from baseline section. |
| LogisticRegression 2 | Model from baseline section. |
| LogisticRegression 3 | Model from baseline section. |
| XGBoost | Best resulting model from xgboost section.  |
| NeuralNetwork | Best resulting neural network from best-shot section. |

### Results and Analysis


#### Regular Season Games

<figure style="display: block;margin-left: auto; margin-right: auto;width:100%;height:100%;">
    <img src="/public/conf_matrix_regular.png" alt="conf_matrix_regular">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Confusion Matrices Regular Season (2019-2020)</figcaption>
</figure>


<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/roc_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: ROC Regular Season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/goal_rate_vs_percentile_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Goal rate Regular Season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/proportion_goal_percentile_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Goal rate cumulative Regular Season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/calibration_regular.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Calibration results Regular Season (2019-2020)</figcaption>
</figure>


#### Playoff Games

We saw that our models moslty generalize very well on regular season games when we compared them to our validation set results. However, we can clearly see that our models perform poorly on playoff games. Throughout Figure X to Figure Y!!! we can see that our same models perform a little bit worse on playoff games compared to regular season games. This was expected since we have only trained our models only on regular season games. Alos, since the playoff games are fundamentally different from the regular season games ("Anything can happen in the playoffs!"). The reason for this is that the playoff games are elimination games, which means that the teams and players always play with heavy pressure. In addition, the environemnt and the "energy" is different during the playoffs, which ultimately could features that we simply don't have access to.

It will be interesting to see if by adding external features such as a metric of the will help generalizing our models. Another interesting thing would be to train our models on playoff games instead of regular season games to see if it can generalize better both regular season and playoff games.


<figure style="display: block;margin-left: auto; margin-right: auto;width:100%;height:100%;">
    <img src="/public/conf_matrix_playoffs.png" alt="conf_matrix_playoffs">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Confusion Matrices Playoffs season (2019-2020)</figcaption>
</figure>


<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/roc_playoffs.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: ROC Playoffs season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/goal_rate_vs_percentile_playoff.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Goal rate Playoffs season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/proportion_goal_percentile_playoffs.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Goal rate cumulative Playoffs season (2019-2020)</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/calibration_curve_playoffs.png" alt="best_shot_nn_calib">
    <figcaption style="font-size: 12px;text-align: center;">Figure 9: Calibration results season (2019-2020)</figcaption>
</figure>


