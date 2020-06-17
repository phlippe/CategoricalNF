## Experiments

### Organization

Every experiment consists of (at least) the following files/folders:
* _train.py_: This file implements a training class that inherits the training template class from [general/train.py](../general/train.py). In there, the model and task object are created, and arguments for training are defined. For starting a training of the experiment, the train.py is executed (i.e. `python train.py --...`)
* _task.py_: Besides a training object, every experiment also provides a task object which implements the loss calculation and tensorboard logging for the specific experiment. Again, the task class inherits the template from [general/task.py](../general/task.py).
* _eval.py_: File for running the evaluation on the test set for a saved model: `python eval.py --checkpoint_path ...`.
* _model.py_: File for implementing the normalizing flow or other likelihood-based model to apply on the task. The file is usually named more specifically based on the network architecture, and multiple models can be implemented per experiment.
* _datasets/_: Folder that contains the files for loading the datasets of the expeirment. The actual data is, however, saved in a separate `data/` folder.
* _checkpoints/_: Folder to save models and training information in sub-folders of.
