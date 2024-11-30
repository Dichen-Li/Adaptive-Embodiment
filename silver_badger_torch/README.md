### Install
1. Run: ```pip install -r requirements.txt```

### Deployment run
1. Source the ros setup.bash
2. Source the hb40 install setup.bash
3. Go into ```nn_ros_controller.py``` and make sure ```self.is_real_robot = True``` (line 23). If you want to test in the MAB simulator, set it to False. If your robot is not the TUDa robot make sure that ```self.is_tuda_robot = False``` (line 24) is set => Else the spine axis is inverted.
4. Run ```python3 nn_ros_controller.py```
5. Wait until the consol prints: ```"Robot ready. Using device: CPU"```.
6. Press ```B``` (the right button) on your Xbox controller to start the NN policy. Press ```A``` (the down button) to stop it.
