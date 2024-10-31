import torch


class RandomCommands:
    """
    Generates random velocity commands
    It returns one command regardless of number of parallelized envs
    """
    def __init__(self, env, x_vel_range, y_vel_range, yaw_vel_range):
        self.env = env
        self.min_x_velocity = x_vel_range[0]
        self.max_x_velocity = x_vel_range[1]
        self.min_y_velocity = y_vel_range[0]
        self.max_y_velocity = y_vel_range[1]
        self.min_yaw_velocity = yaw_vel_range[0]
        self.max_yaw_velocity = yaw_vel_range[1]

    def get_next_command(self):
        goal_x_velocity = torch.rand(self.env.num_envs, 1).uniform_(self.min_x_velocity, self.max_x_velocity).to(self.env.device)
        goal_y_velocity = torch.rand(self.env.num_envs, 1).uniform_(self.min_y_velocity, self.max_y_velocity).to(self.env.device)
        goal_yaw_velocity = torch.rand(self.env.num_envs, 1).uniform_(self.min_yaw_velocity, self.max_yaw_velocity).to(self.env.device)

        return goal_x_velocity, goal_y_velocity, goal_yaw_velocity
