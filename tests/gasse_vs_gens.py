import ecole
from pathlib import Path

if __name__ == "__main__":
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}
    # branching/relpscost/priority = 10000
    default_env = ecole.environment.Branching(
        observation_function=ecole.observation.StrongBranchingScores(),
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)

    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=500, graph_type="barabasi_albert", affinity=4)

    train_files = [
        [str(path) for path in Path(
        f'branch2learn/data/instances/cauctions/train'
        ).glob('instance_*.lp')],
        [str(path) for path in Path(
        f'branch2learn/data/instances/cauctions/train'
        ).glob('instance_*.lp')],
        [str(path) for path in Path(
        f'branch2learn/data/instances/cauctions/train'
        ).glob('instance_*.lp')],
        [str(path) for path in Path(
        f'branch2learn/data/instances/cauctions/train'
        ).glob('instance_*.lp')]

    for i, instance in zip(range(10), generator):
        """observation, action_set, _, done, info = default_env.reset(instance)
        
        while not done:
            action = action_set[observation[action_set].argmax()]
            observation, action_set, _, done, info = default_env.step(action)

        print(f"ecole nb nodes  {int(info['nb_nodes']): >4d}  | ecole time {info['time']: >6.2f} ")
        """
        print(train_files[i])
        observation, action_set, _, done, info = default_env.reset(train_files[i])
    
        while not done:
            action = action_set[observation[action_set].argmax()]
            observation, action_set, _, done, info = default_env.step(action)

        print(f"gasse nb nodes  {int(info['nb_nodes']): >4d}  | gasse time {info['time']: >6.2f} ")
