# Snake
For the 'Snake' case study, first **read the readme** inside the 'snake' case study.
Then, install the requirements.txt file.
  
## Reproducing Results  
  
To reproduce the results we got in the paper, simply run compare_models.py and choose the option you want to see.  
To see the comparison with the Traffic Grammar, run compare_training.py.  
  
The full trees with the original grammar is in the directory Tree/trees/orig_trees  
The full trees with the Traffic grammar is in the directory Tree/trees/traffic_trees  
  
To see a comparison with VIPER, you must run viper_Snake/pong/compare_viper.py. But first we must train VIPER run viper_Snake/pong/main.py (this may take a while). You can also use the existing VIPER tree, but need to test it and create the metrics for the comparison.  

To get similar results, do the following:  
### Pre Training  
We need to create two files with undesirable and desirable state-action pairs (in the format that we will feed the decison tree with - i.e. with features included)  
Now, we need to train our tree on those files.  
  
### Training  
After that, we run agent_1.py with parameters train_or_test, reward_modifier and run_number, e.g.:  
`path_to_env path_to_agent_1.py -t 0 -re reward_modifer -ru run_number`  
Just remember to change the path from the old tree to the new tree we just trained.
  
### Testing  
After training, we run testing_main.py, with parameters run_number, e.g.:  
`path_to_env path_to_agent_1 -t 0 -re reward_modifer -ru run_number`

## Files
Now, we will explain what each file and folder contains:

 - agent_1.py - has the code for the snake-playing agent. We modified it to run our approach.
 - batch_writer.py - if one wants to run the training in batches on some cluster, running this will give the commands to enter when running on the cluster.
 - snake_env.py - the snake environment. Minor to no changes made from original file.
 - Folder - model
	 - All the different models from previous runs. Mostly irrelevant.
 - Folder - Tree
	 - Folder - Properties
		 - Folder - Property 1
			 - Here are files relevant for the undesirable property we chose.
			 - des.txt -  the data file which contains extracted features for the regular grammar for desirable state-action pairs, 
			 - des_traffic.txt -  the data file which contains extracted features for the Traffic-Control grammar for desirable state-action pairs, 
			 - dict.pkl - dictionary of {depth : {metric : value}} for the trees trained on the original grammar.
			 - dict_traffic.pkl - dictionary of {depth : {metric : value}} for the trees trained on the Traffic-Control grammar.
			 - new_tree.pkl - the final decision tree, used for training the model.
			 - undes.txt - the data file which contains extracted features for the regular grammar for undesirable state-action pairs.
			 - undes_traffic.txt -  the data file which contains extracted features for the Traffic-Control grammar for desirable state-action pairs.
			 - The rest of the files are irrelevant.
	- Folder - trees:
		- Folder - trees_new - pngs and dot files for the trees for each depth
		- Folder - traffic_trees_new - pngs and dot files for the trees trained on Traffic-Control grammar for each depth
		- Folder - orig_trees - obsolete
		- Folder - traffic_trees - obsolete
	- compare_models.py - run to get the graphs we displayed in the paper.
	- compare_training.py - comparing between the different metrics between trees trained over regular grammar, and trees trained over Traffic-Control grammar.
	- create_features.py - obsolete
	- features.py - contains classes relevant to the Snake grammar
	- traffic_features.py - contains classes relevant to the Traffic-Control grammar for snake.
	- tree.py - This contains all the relevant functions to training the trees and showing them. Download graphviz for this to work properly.
	- utils_for_tree.py - utility functions for tree.py
- Folder - trees:
	- Folder - trees_new - pngs and dot files for the trees for each depth
	- Folder - traffic_trees_new - pngs and dot files for the trees trained on Traffic-Control grammar for each depth
- Folder - viper_Snake - contains all the files for the comparison with VIPER
	- Folder - core
		- dt.py - contains the decision tree policy of VIPER (with our additions)
		- rl.py - has the main RL logic of VIPER (with our additions)
	- Folder - pong
		- compare_viper.py - run to get the graphs for VIPER comparison.
		- dqn.py - contains the DQN policy of VIPER (with our additions)
		- final_student.svg - the svg file of the tree we got with VIPER
		- main.py - the main to run for running VIPER
	- Folder - snake_reduced - has the VIPER reduction tree for the Snake network, also has the log files and the relevant graphs for VIPER comparison.
	- Folder - util - has log.py file, we did not touch it
 - obsolete files:
	 - analyze_graph.py
	 - decision_tree.py
	 - features.txt
	 - featureset.py
	 - plot_script.py
	 - ruleset.py
	 - train_data.txt
