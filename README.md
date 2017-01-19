# Outbrain Click Prediction challenge solution

- The goal of the competition is to predict which ad will be clicked
- See https://www.kaggle.com/c/outbrain-click-prediction for more 
- This is a part of the 13th place solution to the challenge


## Overview:

The part of the solution is a combination of 5 models:

- SVM and FTRL on basic features:
  - event features: user id, document id, platform_id, day and hour and geo
  - adv features: ad document id, campaign, advertizer id
- XGB and ET on MTV (Mean Target Value) features:
  - all categorical features that previous model used
  - document features like publisher, source, top category, topic and entity
  - interaction between these featuers
- FFM with the following features:
  - all categorical features from the above, except categories, topics and entities
  - XGB leaves from the previous step (see slide 7 from [this presentation](http://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf) for the description of the idea)

To get the 13th positions, models from [diaman](https://www.kaggle.com/dselivanov) should also be added 

## Files description




The files should be run in the above order
