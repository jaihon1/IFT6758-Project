---
layout: post
title: IFT6758 Milestone 2
---

## Question 2

Using our work from the previous milestone, we have extended our features by adding the following ones:

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
    <img src="/public/goal_rate_vs_distance.png" alt="goal_rate_vs_distance">
    <figcaption style="font-size: 12px;text-align: center;">Figure 3: Goal rate vs distance.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/goal_rate_vs_angle.png" alt="goal_rate_vs_angle">
    <figcaption style="font-size: 12px;text-align: center;">Figure 4: Goal rate vs angle.</figcaption>
</figure>

<figure style="display: block;margin-left: auto; margin-right: auto;width:50%;height:50%;">
    <img src="/public/histogram_goals.png" alt="Goals (empty net and non-empty net) vs distance">
    <figcaption style="font-size: 12px;text-align: center;">Figure 5: Goals (empty net and non-empty net) vs distance from net.</figcaption>
</figure>

Looking at the data from Figure 5, we can observe many interesting facts. Firstly, the vast majority of goals are on non-empty net which is logical since goalies are in the net most of the time. Secondly, we can observe that most of the goals are being scored within 60 feet from the net, which is inside the opponents' half of the rink. These two observations are aligned with our domain knowledge, and it makes perfect sense that the further you are from the opponents' net, the harder it is to score when there is a goalie in front of the net. With that said, the goals that were made from a distance of 150 feet when there was a goalie sound a bit unlikely.

Question 4

Game seconds: total sum of seconds elapsed in the game
Game period: date of the game
Coordinates: coordinates(x, y) of the shot
Shot distance:  distance from the shot to the net
Shot angle: angle from between the shot and the net
Shot type: type of Shot (Wrist, Slap, Backhand, etc...)
Empty net:
Last event type:
Coordinates of the last event: coordinates(x, y) of the last event
Time from the last event: time elapsed from the last event
Distance from the last event: distance calculated from the last event
Rebound (bool): True if the last event was also a shot, otherwise False
Change in shot angle: only include if the shot is a rebound, otherwise 0
Speed: defined as the distance from the previous event, divided by the time since the previous event.
Time since the power-play started (seconds): time since the penalty started
Number of friendly non-goalie skaters on the ice: Number of the team skaters on the ice
Number of opposing non-goalie skaters on the ice: Number of the opposing skaters on the ice



