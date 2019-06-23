# Gaze Estimation

This repository is the culmination of a semester-long internship at ISR, mentored by Phd candidate João Avelino (@joao-avelino). The finnal objective is to improve a social robot's skill of understanding Humans, through one of our most revealing features, the gaze.

To achieve our goal, representative videos of people looking at objects (in Sims 4, given its ease of use) were recorded. From each frame was extracted information regarding the head's position and where the person was looking at (ground truth) - this data was then used to train a support vector machine, creating a classifier able to estimate the coordinates of the object in a 2D space with high accuracy (~90%). Runing it in real time in Vizzy (social robot) it [the robot] is supposed to follow the person's visual vocus of attention.
