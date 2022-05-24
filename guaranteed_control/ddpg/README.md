
# DDPG Agent

The DDPG agent is based on DeepMind's paper: . It consists in two actor networks and two critic networks used to output actions through one actor network, while the critics give a value that is related to how good the actions are to increase the rewards on the long term. We also add smooth regularization conditions for the actions of the actors in order to make the different states of the system be more continuous as described in ... .