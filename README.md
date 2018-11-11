## Action Representation
Chunhui LIu & Kenan Deng Project for CMU 16811 2018 FALL

3D skeleton points derived from the human body can provide valuable and comprehensive representations of human actions. Modeling the trajectories of real-world skeleton points can help us understanding and generating human actions.

This project focus on representing trajectories of 3D points efficiently using approximation and interpolation approaches. Given real-world action coordinates, we try to represent the action trajectories using a combination of basis functions. 
There are following questions that will be possibly concerned. For example, How to generate new trajectories on new animation models given real-world data? Given a human skeleton data, is there a way to classify those action? How to blend the different actions together to create new animations? 

We would like to try different approximation methods on each point trajectory of the skeleton animations data. Also, we would like to cluster the existing skeleton data and derive a sensible form to represent each action using these techniques. Further, we plan to use such representation to generate new combination actions which consist of multiple actions. Also, through this representation power, we plan to use this as an action descriptor to do recognition task on the action. 

# DATASET:
MSR-action-3D
todo: update data format in this repo

# project outline
* definition of problem 
* solutions: 
  1. baseline （x,y,z) based : approximation using basic function (L2 dis) 
  2. (u,v) based:  approximation using basic function (L2 dis) 
  3. (u,v) based: lie group based analysis
* experiments:
  1. representations of actions, maybe combination 
  2. classification

# MILETONE
1. 11/5  - 11/12 
* LCH: Pre-process Data  
* DKN: go through code 
2. 11/12 - 11/19 
* LCH: implement solutions 
* DKN: implement solutions 
3. 11/19 - 11/26 
* experiments
4. 11/26 - 12/03
