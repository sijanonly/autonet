#### autonet
##### It is a Master project under Machine learning on title "Dynamic network creation for time series".
=====================================================================

Ideas
============
1. Hyperparameter search space is large for time series data and tuning it is a complex task.
2. Using reinforcement learning to find the optimal combinations of the parameters.

Submissions:

1. [Simple Optimization Test](https://github.com/sijanonly/autonet/blob/master/submissions/1.0-simple-optimization-test.ipynb)
2. [Q learning and agent grid layout](https://github.com/sijanonly/autonet/blob/master/submissions/2.0-q-learning-and-agent-grid.ipynb)

   #### setup
      - discrete reward function
      - constant exploration over each episodes
     
   #### observations:
   1. Agent stuck on one side and q-values keep on increasing, which is a sign of local minima solution.

3. [Q learning with continuous reward](https://github.com/sijanonly/autonet/blob/master/submissions/3.0-q-learning-and-countinous-reward-function.ipynb)
   
   #### setup
      - continous reward function
      - dynamic exploration with decaying epsilon values over episode
      - Q values as heatmap to check/discover the agent movement pattern
   
   #### observations:
      - with continous reward, convergence is faster

      ![Agent](images/agent.gif)