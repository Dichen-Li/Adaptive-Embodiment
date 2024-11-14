import pickle

import flax.serialization


def save_trainstate(trainstate, filepath):
    """Serialize and save a Flax TrainState object."""
    # Convert TrainState to a state dictionary
    state_dict = flax.serialization.to_state_dict(trainstate)

    # Save the state dictionary using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(state_dict, f)


def restore_trainstate(trainstate_class, filepath):
    """Load and restore a Flax TrainState object from a file."""
    # Load the state dictionary from the file
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)

    # Restore the TrainState object using the state dictionary
    restored_trainstate = flax.serialization.from_state_dict(trainstate_class, state_dict)
    return restored_trainstate


class AverageMeter:
    """Keeps track of the running average of a metric (e.g., loss, accuracy)."""

    def __init__(self):
        """Initializes internal variables to keep track of sums, counts, and averages."""
        self.reset()

    def reset(self):
        """Resets the meter to initial state."""
        self.val = 0  # Current value
        self.avg = 0  # Average value
        self.sum = 0  # Sum of all values
        self.count = 0  # Total number of updates

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): New value to update the meter with.
            n (int): Weight or count associated with the value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
