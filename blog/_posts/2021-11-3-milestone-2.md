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
For our baseline, we trained a Logistic Regression model using only the *distance* feature that we have previously extracted from the raw data, and it gave us a **90.59%** accuracy when we ran it on our validation dataset. We also generated the following confusion matrix to have a better look at our model's results:

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
To get the number of friendly players on ice and the number of opposite players on ice, we first checked the side of the team to figure out
who is friendly and who is not and then subtracted the number of players lost depending on the type of the penalty from 5.


link to the experiment which stores the filtered DataFrame artifact
[wpg_v_wsh_2017021065.csv](https://www.comet.ml/jaihon/ift6758-project/fae888ad53de4d1aa940a67b96d106ab?assetId=e46feef96edc4bf8afe7c676f05c192b&assetPath=dataframes&experiment-tab=assets)



## Question 6: Best Shot

<div style="text-align: justify">
After using a logistic regression and XGBoost model, we decided to try other algorithms to find the best model
for our task of binary classification with our collected dataset. We have made many experiences in this milestone.
They are all available on comet_ml.
<br>
We decided to try K-Nearest Neighbors, Random Forest and a feed-forward neural network.
As already pointed out in section 2, our dataset is very unbalanced. Therefore, classification accuracy and its complement, the error
rate, might be a bad idea to use because it will be an unreliable measure of the model performance. We have have what is called an "Accuracy paradox"(5). In that case, a good performance on the minority class (Goal) will be preferred over a good performance on both class.
In order to do so, alternative performance metrics, like precision, recall or the F-measure, may be required since reporting the classification accuracy may be misleading.
In the following section, we will explain how the features were processed for each model, how they were trained and which metrics were used. All our models have been split into training (80%) and validation (20%) set using a stratified strategy and have been optimized using cross-validation to find the best hyperparameters. The figures shown in this section have been obtained after evaluating our models on the validation set.
</div>

### Models

#### 1. KNN

<div style="text-align: justify">
KNN was trained on all the features created in section 4. For the preprocessing, we started by changing all of the categorical features into one-hot encoding. We then dropped the rows that had nan values and
removed the ones that had infinity values (like the Speed Column). Once this was done, we split the dataset into training and validation set as specified above.
We trained our KNN using a GridSearch on different hyperparameters: the number of neighbors and the weights used (distance vs uniform). The best model found by the GridSearch used 8 neighbors and distance for weights.
However, even if the GridSearch did find a "best estimator", it was only able to reach an AUC of 0.63 which is lower than the XGBoost model.
</div>

#### 2. Random Forest
<div style="text-align: justify">
The Random Forest had a similar preprocessing as the KNN, i.e. we used the same features with the one-hot encoding for the categorical features, etc.
We also trained the Random Forest using a GridSearch over 2 hyperparameters: the criterion which is the function used to measure the quality of a split in a tree and the number of estimators. [<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor">Random Forest from Scikit-Learn</a>]
Once again, the model seemed to perform poorly as it only had AUC of 0.63 for the cross-validation results of the GridSearch.
</div>

#### 3. Neural Network
<div style="text-align: justify">
For the neural network, different methods were used for the preprocessing. We tried first a model that did not use our
features created in the bonus part of section 4 (everything related to penalty). This mean that our model had less features
than the others tried as of now. Still, for all neural networks models, we transformed the categorical data into one-hot vectors.
Moreover, we considered the feature current_friendly_on_ice and current_opposite_on_ice as categorical feature as opposed to the other models which supposed they were numerical and therefore were standardized which does not make much sense for a number of people on the ice that ranges from 1 to 5.
As the other models, we standardized the numerical features.
The other two models differed on the used of dropouts. They both trained with all of the features created in section 4 (including the bonus), but one of them used dropout and the other didn't.
For the training, we did some cross validation over different hyperparameters like the learning rate, the coefficient for the adam optimizer and the number of epochs. All the hyperparameters of the 3 models are the same, because those parameters had been optimized
previously. So mainly, the learning rate is 0.001, the Adam coefficient is 0.9, and we trained for 30 epochs. So, to reiterate, the main difference is that the 'nn_no_bonus_feature' has no feature developped during the bonus part (section 4),
the 'nn_no_dropout' has no dropout and the "best_shot_nn_final" has all of the features and dropout.

Not presented here, but we did train a model without standardizing the numerical feature, but found that the performance was better if standardization was done.
</div>

Here is a list of the features selected to train our neural network which we selected based on our domain knowledge:

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
Because the dataset was very unbalanced in nature, we decided to mainly use the F1 Score. In addition, we also used a
custom-made threshold technique to help us analyse the results of our models. Since our models' outputs were very small
probability values, we decided that the 0.5 threshold for binary prediction wasn't the way to go. Instead, for each model
we trained, we found a better threshold value that would give us the optimal F1 score at the end.
</div>

#### Results and Analysis

Using the dropout technique to help prevent over-fitting, we can see that our neural network model performs better with this regularization technique.

Including features developed in the bonus question (current_time_seconds, time_since_pp_started, current_friendly_on_ice, current_opposite_on_ice), we can see that our neural network model performs better.

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


##### Links to our models

1. [Neural Network - best_shot_nn_final](https://www.comet.ml/jaihon/ift6758-project/f02e46ac553944f7ba18060044d873e9?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
2. [Neural Network - nn_no_bonus_feature](https://www.comet.ml/jaihon/ift6758-project/f22281d6264d462685c13628a0dd7daa?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)
3. [Neural Network - nn_no_dropout](https://www.comet.ml/jaihon/ift6758-project/b086d3049e1f47b7ae8aa569994983b4?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)



The best AUC on graph is actually the best_shot_nn_final with an AUC of 0.77 which corresponds to the
model trained with the bonus_features, the dropout and normalization. The performance was pretty equal
to the other two models (nn_no_bonus_feature with AUC 0.76) trained with no bonus features, and equal to nn+dropout (AUC=0.76)
trained with no dropout and pretty much the same nn_no_normalization (AUC 0.76) that was trained with no normalization.
So we can see that at a high level,our models had pretty much the same performances.



The first curve is a ROC curve that explains the results of our KNN. Area Under the Curve” (AUC) of “Receiver Characteristic Operator” (ROC)
is a plot of True Positive Rate vs False Positive Rate.
The AUC-ROC curve helps us visualize how well our machine learning classifier is performing.(2)
Quite surprisingly, our calculated AUC on graph is actually 0.94 for the KNN, (0.5<AUC<1) which tells us that there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values.  This comes as a surprised for us because the results that we got on the gridsearch were less than optimal. We had AUC (of ROC) of about 0.63 for pretty much all our cross-validation trials.

![roc_curve.png](/public/roc_curve.png)

The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of
the ROC curve.(2) The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. (2)
This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.
ROC curves can present an overly optimistic view of an algorithm’s performance if there is a large skew in the class distribution.

![cum_sum.png](/public/cum_sum.png)
![goal_.rate](/public/cum_sum.png)

It is evident from the plot that the AUC for the RandomForest ROC curve is higher than that for the KNN ROC curve.
Therefore, we can say that logistic regression did a better job of classifying the positive class in the dataset.
Building a random forest starts by generating a high number of individual decision trees.
Random forest models are accurate and non-linear models and robust to over-fitting. (4)

### Question 7: Best Shot



In conclusion, the XGboost and the Neural Network were the two best models with overall same performance.


Bibliography:

1. Jason Brownlee , "How to Use ROC Curves and Precision-Recall Curves for Classification in Python", "https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/", January 13, 2021
2. Bhandari, Aniruddha , "AUC-ROC Curve in Machine Learning Clearly Explained" "https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/", June 16, 2020
3. Takaya Saito, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets",  "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/"
4. Author Derrick Mwiti, October 26th, 2021; Random forest models are accurate and non-linear models and robust to over-fitting' 'https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why'
5. Jason Brownlee, January 1, 2020 , "Failure of Classification Accuracy for Imbalanced Class Distributions", "https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/"
