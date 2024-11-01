class JointPositionAction:
    def __init__(self, env, scale, use_default_offset):
        self.env = env
        self.scale = scale
        self.use_default_offset = use_default_offset
        self.default_offset = self.env.robot.data.default_joint_pos

    def process_action(self, action):
        """
        Returns processed joint position action, given raw actions
        predicted by the model
        :param action: raw action from policy
        :return: target joint positions to be achieved by the robot
        """
        processed_action = action * self.scale
        if self.use_default_offset:
            processed_action += self.default_offset
        return processed_action
