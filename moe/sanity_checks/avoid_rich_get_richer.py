"""
Created 05 February 2024
Kristoffer Nesland, kristoffernesland@gmail.com

TODO: seems like there is a problem with one of the experts getting all the weights, https://chat.openai.com/share/5756feee-ee0e-4fd0-a26a-ecbbe26e3dbd
TODO: perhaps one can take the mean of the weighting for each expert in the entire batch and penalize based on the distance from 1/n_experts?

TODO: could also try the mixture of Gaussians cost function.

EDIT: looks like the mixture of Gaussians cost function helped (?)
"""
