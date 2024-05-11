# General
This is the repository for the code written for the paper 'On Reducing Undesirable Behavior in Deep-
Reinforcement-Learning-Based Software', published in FSE 2024.

Authored by Ophir Carmel and Guy Katz.
  
In the paper, there were 3 different case studies presented:  
- Aurora (PCC-RL-master)  
- TrafficControl  
- Snake  

The code with our alterations (implementations of our approach) for each one of the case studies, is in each of the corresponding folders.

In the 'Graphs' folder are all the relevant graphs for our paper.
  
For Aurora, a python3.7 environment is *required*.  
For TrafficControl, a python3.7 environment is *required*.  
For Snake, a python3.9 environment is *required*.  

Therefore, if you want to use these case studies, we recommend copying the relevant folder to a different project, and running each case study separately (as to not need to switch interpreters between runs).

Each case study has a different requirements.txt - install the relevant one for each of the case studies.

We recommend running the snake-rl case study first, as it is the easiest to install (does not require additional installations) and is the easiest to run.

**In each of our case studies, we took already existing code and added to it. Therefore, most of the code (which is unrelated to training the decision trees) is the code from the original project, and most of our additions to the original code are clearly highlighted (our important additions were put between two comments that say - 'START ADDITION' and 'END ADDITION') .**

## Graphs
In this folder, there are multiple sub-folders:
 - **Convergence** - has the graphs for the reward over time for each of our case studies.
 - **Grammar Comparison** - has two sub-folders (for the Traffic-Control case study and the Snake case study), which compares different metrics when training the trees on each one of the case studies with both of their grammars.
 - **Ratio** - has graphs of the undesirable behavior ratio over different reward-modifiers for each of the case studies.
 - **Reward** - has graphs of the average reward over different reward-modifiers for each of the case studies.
 - **Trees** - has the trees used for training the different case studies.
 - **VIPER comparison** - has the graphs comparing VIPER with the case studies Snake and Traffic-Control.

## General Running Tips
When running each one of the case studies, you might want to comment out all the places which use wandb if you do not want to use it yourself (except Aurora).

If you encounter issues with the imports of internal modules, try marking the relevant directories as sources root, or change the import path.

Remember to use different python version and download all relevant packages. For Traffic Control specifically, you must download an external application, and for Aurora you might encounter issues with the stable-baselines package.

If you encounter any other issues, feel free to contact me (the author) of the paper.

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
We need to create two files with undesirable and desirable state-action pairs (in the format that we will feed the decision tree with - i.e. with features included)  
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



# Traffic-Control
For the 'Traffic-Control' case study, first **read the original readme** inside the 'Traffic-Control' folder.
Then, install the requirements.txt file.
To run Traffic Control, you should install SUMO (the original readme explains how), and also graphviz.

## Reproducing Results  
  
To reproduce the results we got in the paper, simply run compare_models.py and choose the option you want to see. This will display the graphs shown in the paper, and some more.  

To see the comparison with the Snake Grammar, run compare_training.py.  
  
The full trees with the original grammar is in the directory Tree/Trees  
The full trees with the Snake grammar is in the directory Tree/Trees_snake  
The final tree we used can be found in Tree/Properties/Property 0/finalized_tree.pkl.  
  
To see a comparison with VIPER, you must run test_multiple.py, and change the flag in the code to "VIPER" (this may take a while). if you want to train another VIPER tree, run TLCS/viper_TrafficControl/pong/main.py (this may take a while).  
  
  
To get similar results, do the following:  
### Pre Training  
We need to create two files with undesirable and desirable state-action pairs (in the format that we will feed the decision tree with - i.e. with features included)  
Now, we need to train our tree on those files.  
  
### Training  
After that, we run training_main.py, with parameters reward_modifier and run_number, e.g.:  
```  
path_to_env path_to_training_main -re reward_modifer -ru run_number`  
```  
  
### Testing  
After training, we run testing_multiple.py, with parameter run_number, e.g.:  
```  
path_to_env path_to_testing_main -ru run_number  
```
## Files
 - batch_writer.py - if one wants to run the training in batches on some cluster, running this will give the commands to enter when running on the cluster.
 - Readme.md - original readme. You should read this.
 - Folder - TLCS:
	 - compare_models.py -  run to get the graphs we displayed in the paper, of the undesirable behavior ratio and average reward, and also the time overhead.
	 - consts.py - constants used for the Traffic-Control case study
	 - generator.py - generates traffic. Original Traffic-Control logic, we did not touch this file.
	 - memory.py - handles memory. Original Traffic-Control logic, we did not touch this file.
	 - model.py - handles the model. Original Traffic-Control logic, we did not touch this file.
	 - test_multiple.py - get undesirable behavior ratio and reward modifiers for a single run. This file is based on the testing_main.py file, and we modified it all over.
	 - testing_simulation.py - the test simulations. Our changes are highlighted.
	 - testing_settings.ini - the ini file for the testing.
	 - traffic_ratio_comparison.svg - the undesirable behavior ratio graph shown in the paper.
	 - traffic_reward_comparison.svg - the average reward graph shown in the paper
	 - training_main.py - the training main file. We train new runs with by running this with some reward modifier as an argument.
	 - training_settings.ini - the training settings
	 - training_simulation.py - the training simulation file. Our changes are highlighted.
	 - utils.py - utility functions for saving the models.
	 - utils_for_trees.py - utility functions for classifying and labeling state-action pairs.
	 - visualization.py - saves data and plots. Original Traffic-Control logic, we did not touch this file.
	 - Folder - intersection - do not touch, has xmls relevant to the intersection modeled.
	 - Folder - models - where the saved models are. Has a bunch of trained and saved models for archive.
	 - Folder - Tree:
		 - compare_training.py - compare_training.py - comparing between the different metrics between trees trained over regular grammar (Traffic-Control), and trees trained over Snake grammar.
		 - create_features - a way for us to create the undesirable and desirable xlsx, from a file of state-action traces. Currently obsolete.
		 - features.py - the features for the Traffic-Control grammar
		 - snake_features.py - the features for the Snake grammar
		 - tree.py - this file contains most of the relevant functions for training, displaying and saving the decision trees, trained on the different grammars.
		 - Folder - compare_trees - this file contains the graphs of the comparison between trees trained with the original grammar and snake grammar, for each metric we chose.
		 - Folder - Properties
				- Folder - Property 1
					- Here are files relevant for the undesirable property we chose.
					- des.txt - the data file which contains extracted features for the regular grammar for desirable state-action pairs. 
					- des_snake.txt - the data file which contains extracted features for the Snake grammar for desirable state-action pairs.
					 - perf_dict.pkl - dictionary of {depth : {metric : value}} for the trees trained on the original grammar.
					 - perf_dict_snake.pkl - dictionary of {depth : {metric : value}} for the trees trained on the Snake grammar.
					 - finalized_tree.pkl - the final decision tree, used for training the model.
					 - snake_tree.pkl - the final tree trained on Snake grammar
					 - undes.txt - the data file which contains extracted features for the regular grammar for undesirable state-action pairs.
					 - undes_snake.txt -  the data file which contains extracted features for the Snake grammar for desirable state-action pairs.
					 - The rest of the files are irrelevant.
		 - Folder - Trees - has all the dot files and the pngs of the tree trained on the original grammar for all depths.
		 - Folder - Trees_snake - has all the dot files and the pngs of the tree trained on the Snake grammar for all depths.
	 - Folder - viper_TrafficControl - contains all the files for the comparison with VIPER
		- Folder - core
			- Folder - intersection - contains all relevant xmls for the Traffic-Control simulation to work properly.
			- dt.py - contains the decision tree policy of VIPER (with our additions)
			- rl.py - has the main RL logic of VIPER (with our additions)
		- Folder - pong
			- dqn.py - contains the DQN policy of VIPER (with our additions)
			- final_student_new - the dot file of the tree we got with VIPER
			- final_student_new.svg - the svg file of the tree we got with VIPER
			- main.py - the main to run for running VIPER
			- all the rest of the files - either irrelevant, or saved pkl files for faster running
		- Folder - traffic_control - has the VIPER reduction tree for the Traffic Control network, also has the log files and the relevant graphs for VIPER comparison.
		- Folder - util - has log.py file, we did not touch it

# PCC-RL  (Aurora)
Reinforcement learning resources for the Performance-oriented Congestion Control  
project.  
This README is about how to produce the results found in the paper, and create similar models.  
  
## Overview  
This repo contains the gym environment required for training reinforcement  
learning models used in the PCC project  
  
## Reproducing Results  
  
To reproduce the results we got in the paper, simply run graph_runs.py and choose the option you want to see.  
  
The tree we used is less_than_1_pnt_2_treefile inside src/gym/create_trees/trees

To get similar results, do the following:  
### Pre Training  
We need to create two files with undesirable and desirable state-action pairs (in the format that we will feed the decision tree with - this means the file should contain the extracted features)  
Now, we need to train our tree on those files.  
  
## Training  
To run training only, go to ./src/gym/, install any missing requirements for  
stable_solve.py and run that script. You need to enter the path the 'gym' directory where it is written in the code (in stable_solve.py and network_sim.py). You must run this with wandb for it to log the undesirable behavior ratio and average reward correctly.
The arguments appear as comments inside the file but should look like:  
  
```  
python3.7 stable_solve.py 0 0 run_number 0 ruleset_name featureset_name tree_name "Message" reward_modifier.  
```  
  
To get the same model we used in the paper "On Reducing Undesirable Behavior in Deep Reinforcement Learning Models", change the following:  
- ruleset_name -> less_than_1_pnt_2  
- featureset_name -> all_with_res  
- tree_name -> less_than_1_pnt_2  

And enter run_number, message, and reward_modifier as you wish.

## Testing Models  
  
To test the models, just change the line to:  
  
```  
python3.7 stable_solve.py 0 1 run_number 0 ruleset_name featureset_name tree_name "Message" reward_modifier.  
```  
  
## Notes  
If running on linux, change the second argument (there are 2 zeros, the second zero) to a 1.


## Files
- batch_writer.py - if you want to run the training in batches on some cluster, running this will give the commands to enter when running on the cluster (using sbatch).
- graph_runs.py - run this file to get the graphs we displayed in the paper for Aurora, and more graphs (like convergence).
- Folder - wandb_runs - we used wandb to log all the runs, these are loaded runs for graph_runs.py to use.
- Folder - src
	- Folder - common - original Aurora code, we did not touch
	- Folder - utd_plugins - original Aurora code, we did not touch
	- Folder - gym - the main folder with the relevant code for training the Aurora model
		- eval_aurora.py - this file contains the function with which we test the model. We call this function from stable_solve.py (if the argument we sent was '1' and not '0' in the train or test argument).
		- graph_run.py - original Aurora code, we did not touch
		- stable_solve.py - run this to train the model. Our changes are highlighted.
		- network_sim.py - logic of the network simulation. Here we added most of our approach logic. Our changes are highlighted.
		- Folder - runs - where models are saved by run_number
		- Folder - online - original Aurora code, we did not touch
	- Folder - create_trees.py
		- Folder - featuresets - has different grammars. The only relevant file is all_with_res_featureset.txt, which features define the regular grammar.
		- Folder - model - obsolete
		- Folder - new_model - obsolete
		- Folder - more_trees - obsolete
		- Folder - rules_files - in this folder are subfolders which correspond to different undesirable behavior. Currently, there is one subfolder, which has undesirable (neg.txt) and desirable (pos.txt) state-action pair examples, and they are used for constructing the trees.
		- Folder - rulesets - in this folder are different .txt files, each one describes a different undesirable behavior. The only relevant one is less_than_1_pnt_2_ruleset, which is the undesirable behavior we measure in this case study.
		- Folder - trees - in this folder are the different trees which correspond to different undesirable behavior (all trained with the regular grammar). The only relevant tree is less_than_1_pnt_2_treefile, which is the tree we use for the retraining of our model.
		- aurora_tree - this file is the .dot file describing the final tree we used for training in this case study. Read this with graphviz.
		- aurora_tree.png - the tree we used for training in this case study.
		- create_data_and_labels_from_rules.py - the file we use for creating the files which we train our decision tree over.
		- decision_tree.py - regular decision trees with some extra nice features.
		- eval_network_restart_from_clean_v1.py - obsolete
		- featureset.py - has all the functions relevant to extracting the features defined in our grammar.
		- generate_srs.py - obsolete
		- main.py - run this to train the tree on the data
		- ruleset.py - has a set of rules which describe undesirable behavior. Using the rules we can determine whether a state-action pair is desirable or not (useful for tree training, and evaluating our final model).
        - samples.txt - traces of states from runs