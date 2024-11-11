import sapien as sapien
from sapien.utils import Viewer
import numpy as np
import argparse


def main(urdf_path):
    scene = sapien.Scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency

    # Add some lights to observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()  # Create a viewer (window)

    # Set up the camera view
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_path)
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    while not viewer.closed:  # Press key q to quit
        scene.step()  # Simulate the world
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a URDF into the Sapien scene.")
    parser.add_argument("urdf_path", type=str, help="Path to the URDF file.")
    args = parser.parse_args()

    main(args.urdf_path)
