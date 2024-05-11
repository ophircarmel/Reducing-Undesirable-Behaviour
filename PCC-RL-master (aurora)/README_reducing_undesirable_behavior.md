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
		- Folder - rules_files - in this folder are subfolders which correspond to different undesirable behavior. Currently there is one subfolder, which has undesirable (neg.txt) and desirable (pos.txt) state-action pair examples, and they are used for constructing the trees.
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