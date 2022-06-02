# Guaranteed Neural Network Control

This library implements different methods to prove specification of controllers of dynamical systems based on Neural Networks. We provide a DDPG implementation using TensorFlow that allows to train some agents on problem environments that should have a similar structure to OpenAI's Gym env structure. Feed-forward neural networks that model the controller and are implemented on TensorFlow with only monotonous activation functions can also be used to study the reachability set of the dynamical system. We provide example of both applications on concrete problems.

<!-- Mountain Car Reachability Proof | Pendulum Stability Proof -->
<p align="center">
  <img src="https://user-images.githubusercontent.com/23098906/169909287-104d6a8f-4e3a-40d9-9c1f-e803846fb33c.png" />
</p>

## Examples
### Pendulum
<p align="center">
<img width="300" height="300" src="./plots/pendulum.gif">
<img src="./plots/pendulum_stability.gif">
</p>

### Double integrator
<p align="center">
<img width="300" height="300" src="./plots/double_integrator.gif">
<img src="./plots/double_stability.gif">
</p>

## Context

We consider a discrete-time closed-loop system that is operated by a deterministic controller which outputs the action at every iteration knowing the observation state of that system. Recent advances in Reinforcement Learning allow to represent such controller only using feed-forward neural networks and can solve complex problems with good generalization capabilities. The goal is therefore to guarantee the stability of these systems when the controller is a neural network. The code is provided with the article:.

## Usage guide

We implement an Interval object that allows to usefully manipulate intervals for the context of reachability analysis and also do basic arithmetic of intervals operations (summing, multiplying, dividing, taking the sinus of intervals...).

Here is a quick example of how to train a DDPG and analyze the reachability sets on a Gym problem.

Training the DDPG agent:

``` python
from guaranteed_control.ddpg.ddpg import DDPG
from guaranteed_control.ddpg.training import train, play
from guaranteed_control.intervals.interval import Interval
from guaranteed_control.problems.mountain_car import MountainCar

env = MountainCar()
input_interval = Interval(interval=[[-0.6, -0.4], [-0.07, 0.07]])
agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], upper_bounds=env.action_space.high, n_layer1=16, n_layer2=16, batch_size=16, noise_std=0.4, epsilon_s=0.05, lambda_smooth=0.2, D_s=10)
agent.start_training(env, agent, input_interval=input_interval, n_episodes=200)
```

Running the closed loop reachability analysis on 50 iterations

```python
from guaranteed_control.closed_loop.reachability.reach import interval_approximation
from guaranteed_control.problems.dynamics import F_car

epsilon = 0.1
specification_interval = None
state_interval = Interval(interval=[[-0.48, -0.4795], [1, 1.001]])
state_intervals = interval_approximation(50, agent.actor, F_car, state_interval, None, epsilon, epsilon_actions=0.5, plot_jumps=1, plot=True, threshold=0.1, verbose=1)
```

More detailed explanations for each module are provided in every module inside the guaranteed_control folder.

## Parameter Tuning

The parameters passed to the interval_approximation function should be chosen depending on the system considered and the specifications it was trained to achieve. $\epsilon$ is the mistake that we want the neural network interval approximation not to exceed between the prediction and the simulated interval. The bigger it is, the bigger the over-approximation errors are, but the faster the algorithm should run. $\epsilon_{actions}$ is the upper bound the size of the predicted action interval should take in order to not divide the state intervals. More detailed explanations are given in the article.

The difficulty comes from the fact that $\epsilon$ and $\epsilon_actions$ are in competition: Imposing a big $\epsilon$ and a small $\epsilon_{actions}$ will just lead to an unnecessarily long run because the interval approximation will keep on dividing the state interval until it becomes small enough for reachMLP to not make a big enough mistake on the interval since the error is proportionate to the size of the interval.

A good practice calibration would be to start with a small $\epsilon$ and make the value $\epsilon_{actions}$ bigger but small enough to not apply big steps on the interval to avoid propagating different scenarios of the state interval in every component of it (ie. creating as many scenarios as necessary inside every iteration).

For more documentation on these parameters and how to use the corresponding function, go the module "sys_reachability" inside the guaranteed_control folder, or check the projects report pdf. 

## Citations

```bibtex
@misc{ddpg,
  doi = {10.48550/ARXIV.1509.02971},
  
  url = {https://arxiv.org/abs/1509.02971},
  
  author = {Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Continuous control with deep reinforcement learning},
  
  publisher = {arXiv},
  
  year = {2015},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@misc{crowson2022,
    author  = {Katherine Crowson},
    url     = {https://twitter.com/rivershavewings}
}
```

```bibtex
@misc{nn_reach,
  doi = {10.48550/ARXIV.2004.12273},
  
  url = {https://arxiv.org/abs/2004.12273},
  
  author = {Xiang, Weiming and Tran, Hoang-Dung and Yang, Xiaodong and Johnson, Taylor T.},
  
  keywords = {Systems and Control (eess.SY), Optimization and Control (math.OC), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Mathematics, FOS: Mathematics},
  
  title = {Reachable Set Estimation for Neural Network Control Systems: A Simulation-Guided Approach},
  
  publisher = {arXiv},
  
  year = {2020},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@book{system_reach,
  title={Interval Reachability Analysis: Bounding Trajectories of Uncertain Systems with Boxes for Control and Verification},
  author={Meyer, P.J. and Devonport, A. and Arcak, M.},
  isbn={9783030651107},
  series={SpringerBriefs in Electrical and Computer Engineering},
  url={https://books.google.fr/books?id=YG8WEAAAQBAJ},
  year={2021},
  publisher={Springer International Publishing}
}
```

```bibtex
@inproceedings{Tu2022MaxViTMV,
    title   = {MaxViT: Multi-Axis Vision Transformer},
    author  = {Zhengzhong Tu and Hossein Talebi and Han Zhang and Feng Yang and Peyman Milanfar and Alan Conrad Bovik and Yinxiao Li},
    year    = {2022},
    url     = {https://arxiv.org/abs/2204.01697}
}
```

```bibtex
@misc{gym,
  doi = {10.48550/ARXIV.1606.01540},
  
  url = {https://arxiv.org/abs/1606.01540},
  
  author = {Brockman, Greg and Cheung, Vicki and Pettersson, Ludwig and Schneider, Jonas and Schulman, John and Tang, Jie and Zaremba, Wojciech},
  
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {OpenAI Gym},
  
  publisher = {arXiv},
  
  year = {2016},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@inproceedings{method_b,
author = {Lecomte, Thierry and Servat, Thierry and Pouzancre, Guilhem},
year = {2007},
month = {08},
pages = {},
title = {Formal Methods in Safety-Critical Railway Systems}
}
```

```bibtex
@article{safety_interpret,
  author    = {Anthony Corso and
               Mykel J. Kochenderfer},
  title     = {Interpretable Safety Validation for Autonomous Vehicles},
  journal   = {CoRR},
  volume    = {abs/2004.06805},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.06805},
  eprinttype = {arXiv},
  eprint    = {2004.06805},
  timestamp = {Tue, 21 Apr 2020 16:51:52 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-06805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{smooth_ddpg,
  author    = {Qianli Shen and
               Yan Li and
               Haoming Jiang and
               Zhaoran Wang and
               Tuo Zhao},
  title     = {Deep Reinforcement Learning with Smooth Policy},
  journal   = {CoRR},
  volume    = {abs/2003.09534},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.09534},
  eprinttype = {arXiv},
  eprint    = {2003.09534},
  timestamp = {Thu, 29 Jul 2021 12:23:01 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2003-09534.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@inproceedings{mpc_toolbox,
  title={Hybrid Toolbox for MATLAB - User's Guide},
  author={Alberto Bemporad},
  year={2003}
}
```

```bibtex
@article{lyapunov,
author = {Adnane Saoud},
title = {Stability analysis using LMIs for systems with neural network controllers}
}
```