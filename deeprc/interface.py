import argparse
import numpy as np
import torch
from deeprc.task_definitions import TaskDefinition, BinaryTarget, MulticlassTarget, RegressionTarget
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, AttentionNetwork, OutputNetwork
from deeprc.training import train, evaluate


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
                column_name='binary_target_1',  # Column name of task in metadata file
                true_class_value='+',  # Entries with value '+' will be positive class, others will be negative class
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

        scores = evaluate(model=self.model, dataloader=self.testset_eval, task_definition=self.task_definition, device=self.device)
        print(f"Test scores:\n{scores}")

    # Evaluate trained model on testset


if __name__ == '__main__':
    a = Interface()

    a.create_task_definitions()
    a.get_dataset()
    a.create_network()
    a.train()
