---
layout: post
title: IFT6758 Milestone 2
---

## Question 2

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



## Question 4

list of all of the features that you created for this section. List each feature by both the column name in your dataframe AND a simple human-readable explanation
Game seconds:
Game period:
Coordinates:
Shot distance:
Shot angle:
Shot type:
Empty net:
Last event type:
Coordinates of the last event (x, y, separate columns):
Time from the last event:
Distance from the last event:
Rebound (bool):
Change in shot angle:
Speed:
Time since the power-play started (seconds):
Number of friendly non-goalie skaters on the ice:
Number of opposing non-goalie skaters on the ice

