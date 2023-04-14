import argparse
import numpy as np
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, MulticlassTarget, RegressionTarget
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate
from tqdm import tqdm


class Interface:
    def __init__(self):
        # parameters
        self.task_definition = None
        self.n_updates = int(1e3)
        self.evaluate_at = int(1e2)
        self.kernel_size = 9
        self.n_kernels = 32
        self.sample_n_sequences = int(1e4)
        self.learning_rate = 1e-4
        self.device = "cuda:0"
        self.rnd_seed = 0

        # dataset
        self.trainingset = None
        self.trainingset_eval = None
        self.validationset_eval = None
        self.testset_eval = None

        # other
        self.model = None

    # Parameters:
    #   n_updates, evaluate_at, kernel_size, n_kernels, sample_n_sequence, learning_rate, device, rnd_seed

    def parse_arguments(self, n_updates: int, evaluate_at: int, kernel_size: int, n_kernels: int, sample_n_sequences: int,
                        learning_rate: float, device: str, rnd_seed: int):
        pass

    def create_task_definitions(self):
        self.task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
            BinaryTarget(  # Add binary
                # classification task with sigmoid output function
                column_name='signal_disease',  # Column name of task in metadata file
                true_class_value='True',  # Entries with value '+' will be positive class, others will be negative class
                pos_weight=1.,  # We can up- or down-weight the positive class if the classes are imbalanced
            ),
        ]).to(device=self.device)

    def get_dataset(self):
        self.trainingset, self.trainingset_eval, self.validationset_eval, self.testset_eval = make_dataloaders(
            task_definition=self.task_definition,
            metadata_file="datasets/example_dataset/metadata.tsv",
            repertoiresdata_path="datasets/example_dataset/repertoires",
            metadata_file_id_column='ID',
            sequence_column='amino_acid',
            sequence_counts_column='templates',
            sample_n_sequences=self.sample_n_sequences,
            sequence_counts_scaling_fn=no_sequence_count_scaling
            # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
        )

    # Create network
    def create_network(self):
        # Create sequence embedding network (for CNN, kernel_size and n_kernels are important hyper-parameters)
        sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20 + 3, kernel_size=self.kernel_size,
                                                          n_kernels=self.n_kernels, n_layers=1)
        # Create attention network
        attention_network = AttentionNetwork(n_input_features=self.n_kernels, n_layers=2, n_units=32)
        # Create output network
        output_network = OutputNetwork(n_input_features=self.n_kernels,
                                       n_output_features=self.task_definition.get_n_output_features(), n_layers=1,
                                       n_units=32)
        # Combine networks to DeepRC network
        self.model = DeepRC(max_seq_len=30, sequence_embedding_network=sequence_embedding_network,
                            attention_network=attention_network,
                            output_network=output_network,
                            consider_seq_counts=False, n_input_features=20, add_positional_information=True,
                            sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
                            device=self.device).to(device=self.device)

    # Train DeepRC model
    def train(self):
        train(self.model, task_definition=self.task_definition, trainingset_dataloader=self.trainingset,
              trainingset_eval_dataloader=self.trainingset_eval, learning_rate=self.learning_rate,
              early_stopping_target_id='binary_target_1',  # Get model that performs best for this task
              validationset_eval_dataloader=self.validationset_eval, n_updates=self.n_updates, evaluate_at=self.evaluate_at,
              device=self.device, results_directory="results/singletask_cnn_interface"
              # Here our results and trained models will be stored
              )
        # Evaluate trained model on testset
        scores = evaluate(model=self.model, dataloader=self.testset_eval, task_definition=self.task_definition, device=self.device)
        print(f"Test scores:\n{scores}")

    # TODO: train model
    def fit(self):
        pass

    # TODO: predict and return scores to immuneML -
    def predict(self):
        # should return dict signal_disease : [true/false, true/false, ...]
        scores = evaluate(model=self.model, dataloader=self.testset_eval, task_definition=self.task_definition, device=self.device)

    def predict_proba(self):
        # should return dict signal_disease : dict false (), dict true ()

        pass

    # from training.py
    def evaluate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, task_definition: TaskDefinition,
                 show_progress: bool = True, device: torch.device = torch.device('cuda:0')) -> dict:
        """Compute DeepRC model scores on given dataset for tasks specified in `task_definition`

        Parameters
        ----------
        model: torch.nn.Module
             deeprc.architectures.DeepRC or similar model as PyTorch module
        dataloader: torch.utils.data.DataLoader
             Data loader for dataset to calculate scores on
        task_definition: TaskDefinition
            TaskDefinition object containing the tasks to train the DeepRC model on. See `deeprc/examples/` for examples.
        show_progress: bool
             Show progressbar?
        device: torch.device
             Device to use for computations. E.g. `torch.device('cuda:0')` or `torch.device('cpu')`.

        Returns
        ---------
        scores: dict
            Nested dictionary of format `{task_id: {score_id: score_value}}`, e.g.
            `{"binary_task_1": {"auc": 0.6, "bacc": 0.5, "f1": 0.2, "loss": 0.01}}`. The scores returned are computed using
            the .get_scores() methods of the individual target instances (e.g. `deeprc.task_definitions.BinaryTarget()`).
            See `deeprc/examples/` for examples.
        """
        with torch.no_grad():
            model.to(device=device)
            scoring_predictions = []

            for scoring_data in tqdm(dataloader, total=len(dataloader), desc="Evaluating model",
                                     disable=not show_progress):
                # Get samples as lists
                targets, inputs, sequence_lengths, counts_per_sequence, sample_ids = scoring_data

                # Apply attention-based sequence reduction and create minibatch
                targets, inputs, sequence_lengths, n_sequences = model.reduce_and_stack_minibatch(
                    targets, inputs, sequence_lengths, counts_per_sequence)

                # Compute predictions from reduced sequences
                raw_outputs = model(inputs_flat=inputs, sequence_lengths_flat=sequence_lengths,
                                    n_sequences_per_bag=n_sequences)

                # from taks_defintions.py - BinaryTarget()
                prediction = torch.sigmoid(raw_outputs).detach()
                scoring_predictions.append(prediction)

            scoring_predictions = torch.cat(scoring_predictions, dim=0).float().cpu().numpy()

        return scoring_predictions


if __name__ == '__main__':
    a = Interface()

    a.create_task_definitions()
    a.get_dataset()
    a.create_network()
    a.train()
