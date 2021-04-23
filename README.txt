Hello esteemed grader, code for this project can be found here: https://github.com/bjs250/CS7641-PS4

It's written to be compatible with Python Version 3.8.0, and you can install the libraries to run it in your virtual
environment from reqs.txt

Most of the figures used in the report are in the figures subdir
The params subdir contains pickled data structures for some of the longer runs

The code is quite disorganized, but the two main entry points are frozenlake.py and forest.py.
In there you'll find functions with toggles for individual experiments of VI, PI, or QL on the given env

Most of the plotting utility code is found in plotting.py for the frozen lake and forest_plot.py for the Forest

I stole a ton of code from Github in regards to the Frozen Lake, in particular implementations of Value Iteration, Policy Iteration, and Q-Learning
The Forest, on the other hand, was pretty straightforward from mdptoolbox-hiive (https://pypi.org/project/mdptoolbox-hiive/)

References:

Value Iteration
https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
https://reinforcement-learning4.fun/2019/06/24/create-frozen-lake-random-maps/
https://github.com/waqasqammar/MDP-with-Value-Iteration-and-Policy-Iteration/blob/master/MDP_with_PI_VI.ipynb

Policy Iteration
https://github.com/waqasqammar/MDP-with-Value-Iteration-and-Policy-Iteration/blob/master/MDP_with_PI_VI.ipynb (again)
https://marcinbogdanski.github.io/rl-sketchpad/RL_An_Introduction_2018/0403_Policy_Iteration.html

Q-Learning
https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
https://www.eecs.tufts.edu/~mguama01/post/q-learning/

General Resources
https://cs188ai.fandom.com/wiki/Value_Iteration
https://cs188ai.fandom.com/wiki/Policy_Iteration