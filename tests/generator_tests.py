import ecole

if __name__ == "__main__":
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    default_env = ecole.environment.Branching(
        observation_function=ecole.observation.Pseudocosts(),
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)
    default_env = ecole.environment.Configuring(
        observation_function=None,
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)

    generators = {
        'setcover': ecole.instance.SetCoverGenerator(
            n_rows=500, n_cols=1000, density=0.05),
        'cauctions': ecole.instance.CombinatorialAuctionGenerator(
            n_items=100, n_bids=500, add_item_prob=0.7),
        'indset': ecole.instance.IndependentSetGenerator(
            n_nodes=500, graph_type="barabasi_albert", affinity=4),
        'facilities': ecole.instance.CapacitatedFacilityLocationGenerator(
            n_customers=100, n_facilities=100, continuous_assignment = True)
        }

    for problem_type, generator in generators.items():
        print(f'    Problem: {problem_type}')
        for instance_count, instance in zip(range(10), generator):
            default_env.reset(instance)
            _, _, _, _, info = default_env.step({})

            print(f"SCIP nb nodes  {int(info['nb_nodes']): >4d}  | SCIP time {info['time']: >6.2f} ")
