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
We need to create two files with undesirable and desirable state-action pairs (in the format that we will feed the decison tree with - i.e. with features included)  
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
