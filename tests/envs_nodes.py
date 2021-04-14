import numpy as np
import ecole

scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 1000}
info_func = {"nb_nodes": ecole.reward.NNodes().cumsum(), "time": ecole.reward.SolvingTime().cumsum()}

branch_env = ecole.environment.Branching(
    observation_function=ecole.observation.Pseudocosts(),
    information_function=info_func,
    scip_params=scip_parameters)
    
scip_parameters.update({'branching/pscost/priority': 536870911})

conf_env = ecole.environment.Configuring(
    information_function=info_func,
    scip_params=scip_parameters)

generator = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500, add_item_prob=0.7)

for instance in generator:
    
    _, _, _, _, _ = conf_env.reset(instance)
    _, _, _, _, info = conf_env.step({})

    print(f"CONFIGURING: nb nodes    {int(info['nb_nodes']): >4d}  |  time {info['time']: >6.2f}")

    observation, action_set, _, done, info = branch_env.reset(instance)

    while not done:
        scores = observation
        action = action_set[scores[action_set].argmax()]
        observation, action_set, _, done, info = branch_env.step(action)

    print(f"BRANCHNG:    nb nodes    {int(info['nb_nodes']): >4d}  |  time {info['time']: >6.2f}")
