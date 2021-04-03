import ecole
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    TIME_LIMIT = 60
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': TIME_LIMIT}
    # branching/relpscost/priority = 10000
    default_env = ecole.environment.Branching(
        observation_function=ecole.observation.Pseudocosts(),
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)

    generator = ecole.instance.IndependentSetGenerator(
        n_nodes=500, graph_type="barabasi_albert", affinity=4)

    generators = {
        'cauctions': ecole.instance.CombinatorialAuctionGenerator(
            n_items=100, n_bids=500, add_item_prob=0.7),  
        'facilities': ecole.instance.CapacitatedFacilityLocationGenerator(
            n_customers=100, n_facilities=100),
        'indset': ecole.instance.IndependentSetGenerator(
            n_nodes=500, graph_type="barabasi_albert", affinity=4),
        'setcover': ecole.instance.SetCoverGenerator(
            n_rows=500, n_cols=1000, density=0.05),
        
        }

    train_files = {
        #'cauctions': [str(path) for path in Path(
        #    f'branch2learn/data/instances/cauctions/train'
        #    ).glob('instance_*.lp')],
        'facilities': [str(path) for path in Path(
           f'branch2learn/data/instances/facilities/test'
            ).glob('instance_*.lp')],
        'indset': [str(path) for path in Path(
            f'branch2learn/data/instances/indset/test'
            ).glob('instance_*.lp')],
        #'setcover': [str(path) for path in Path(
        #    f'branch2learn/data/instances/setcover/train'
        #    ).glob('instance_*.lp')]
    }
    
    generator = generators['facilities']
    problem_files = train_files['facilities']
    e_node_list,e_time_list,g_node_list,g_time_list = [],[],[],[]
    e_done, g_done = 0, 0
    for i, gen_instance, problem_file in zip(range(5), generator, problem_files):
        print(i)
        observation, action_set, _, done, info = default_env.reset(gen_instance)
        
        while not done:
            action = action_set[observation[action_set].argmax()]
            observation, action_set, _, done, info = default_env.step(action)
        
        
        e_node_list.append(info['nb_nodes'])
        e_time_list.append(info['time'])
        if info['time'] < 0.99*TIME_LIMIT:
            e_done += done

        observation, action_set, _, done, info = default_env.reset(problem_file)
    
        while not done:
            action = action_set[observation[action_set].argmax()]
            observation, action_set, _, done, info = default_env.step(action)

        g_node_list.append(info['nb_nodes'])
        g_time_list.append(info['time'])
        if info['time'] < 0.99*TIME_LIMIT:
            g_done += done

    print('ecole')
    print(f" nb nodes    {int(np.mean(e_node_list)): >4d}  |  time {np.mean(e_time_list): >6.2f}, done {e_done}")
    print('gasse')
    print(f" nb nodes    {int(np.mean(g_node_list)): >4d}  |  time {np.mean(g_time_list): >6.2f}, done {g_done}")

    print('End of evaluation.')
