# VSF
compute velocity structure functions for both 2D (projected position + line of sight velocity) and 3D data.

VSF_obs_share.py is an example for M87. It computes the 2D first order VSF using the projected plane of the sky positions and the line of sight velocity.

It was first used in Li+2020: https://ui.adsabs.harvard.edu/abs/2020ApJ...889L...1L/abstract
Since it uses projected data, there are non trivial projection effects. For non volume filling data analyzed in Li+2020, projection typically steepens the VSF. More discussions and references can be found in the paper and also in https://ui.adsabs.harvard.edu/abs/2023FrASS..1038613G/abstract


VSF_sim_share.py is an example of 3D VSF calculation for simulated data. 

Sorry they are not super well commented, but hopefully they are reasonably straightforward...
