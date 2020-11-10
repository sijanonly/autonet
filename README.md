
## autonet : Neural Architecture Search using Reinforcement Learning for time series data

##### MSc. project (2020), Technical University of Kaiserslautern.

===========================================================================

  

Main Idea

============

1. Hyperparameter search space is large and tuning model for an optimal selection is a complex task.

2. Using reinforcement learning, we will find an optimal policy once the training is complete. Using this policy, we can construct new architecture or use policy as a component to other tasks, maybe ?

 
## Requirements

  

To install requirements:

  

```setup

pip install -r requirements.txt

```

  

>📋 Create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and run the `pip install` command as mentioned above.

>📋 Download the dataset (A5M.csv) from following links and put in `data` folder:
a. [Internet Traffic Time Series](http://www3.dsi.uminho.pt/pcortez/data/itraffic.html)

  
  

## Training

  

To train the model, using following commands:

  

#### Default

  
  

```

python main.py --entropy --shuffle

```

#### Enable Entropy

  

```

python main.py --entropy --shuffle

```

  

### Enable data points shuffle for child network

  
  

```

python main.py --shuffle

```

  

## Experiments

  
  

#### Hypothesis 1 : The RL agent will be able to find the model parameters in 2d-grid surface.

  

**Each cell in 2d-grid represents the error value

  

1.  [Simple Optimization Test](https://github.com/sijanonly/autonet/blob/master/submissions/1.0-simple-optimization-test.ipynb)

2.  [Q learning agent with discrete reward function](https://github.com/sijanonly/autonet/blob/master/submissions/2.0-q-learning-and-agent-grid.ipynb)

  

#### setup

- discrete reward function

- constant exploration over each episodes

#### observations:

1. Agent stuck on one side and q-values keep on increasing, which is a sign of local minima solution.

  
3.  [Q learning with continuous reward function](https://github.com/sijanonly/autonet/blob/master/submissions/3.0-q-learning-and-countinous-reward-function.ipynb)

#### setup

- continous reward function

- dynamic exploration with decaying epsilon values over episode

- Q values as heatmap to check/discover the agent movement pattern

#### observations:

- with continous reward, convergence is faster

  

![Agent](images/agent.gif)
