# Training a model for the resource allocation problem

**usage**: train.py [-h] [--hpsearch] [--baseline] [--algo ALGO]
                [--stage1steps STAGE1STEPS] [--stage2steps STAGE2STEPS] [--l]
                [--ad]
                configuration

**positional arguments**

  * *configuration*     The name of the rap configuration to solve. See
                        resources/rap_configurations.py for configurations.
                        Example: simple_problem_A

**optional arguments**

  * *--ad*              Use A2D2 decomposition. If neither this flag nor baseline is set,
                        DLCA will be used.

  * *--algo ALGO*       Which algorithm to use: A2C or PPO. Default: A2C

  * *--baseline*        Train model without decomposition. If neither this flag nor ad is 
                        set, DLCA will be used.

  * *-h, --help*        show help message and exit
  
  * *--hpsearch*        Train with hyperparameter search
  
  * *--l*               Load stage 1 models from previous training run. Only works for
                        Dean-Lin decomposition. Models must be saved in models folder.
     
  * *--stage1steps STAGE1STEPS*
                        How many training steps in stage 1. Default: 50000
  
  * *--stage2steps STAGE2STEPS*
                        How many training steps in stage 2. Default: 50000
                        
  #### Result ####
  
  Models will be stored in the models folder. Learning curve data and evaluation scores will be
  stored in the data folder.
  
# Training a model for gridwold using Abbad-Daoui decomposition

**usage**: maze_trainer.py [-h] [--i I] maze

**positional arguments**
  * *maze*    The name of the maze configuration to solve. See maze_trainer.py
              for configurations. Example: 4-rooms-4-areas

**optional arguments**:
  * *-h, --help*  show help message and exit
  * *--i I*       How many models to train. The output will be an image showing
              the average of all policies.
 
#### Result ####
An map of the maze will be stored in the img folder. The starting point will be shown in black 
and the goal in green. The agent's policy will be drawn in the form of arrows. If multiple
models were trained the average will be drawn, such that actions taken by more policies will
be more opaque. If there are multiple SCCs the corresponding policies will have different colors.
   