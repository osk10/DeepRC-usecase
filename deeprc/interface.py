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

    def parse_arguments(self, n_updates: int, evaluate_at: int, kernel_size: int, n_kernels: int,
                        sample_n_sequences: int,
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

    def get_dataset(self, metadata_file: str, repertoiresdata_path: str, one_loader: bool = False):
        if not one_loader:
            self.trainingset, self.trainingset_eval, self.validationset_eval, self.testset_eval = make_dataloaders(
                task_definition=self.task_definition,
                metadata_file=metadata_file,
                repertoiresdata_path=repertoiresdata_path,
                metadata_file_id_column='ID',
                sequence_column='amino_acid',
                sequence_counts_column='templates',
                sample_n_sequences=self.sample_n_sequences,
                sequence_counts_scaling_fn=no_sequence_count_scaling
            )
        else:
            dataloader = make_dataloaders(
                task_definition=self.task_definition,
                metadata_file=metadata_file,
                repertoiresdata_path=repertoiresdata_path,
                metadata_file_id_column='ID',
                sequence_column='amino_acid',
                sequence_counts_column='templates',
                sample_n_sequences=self.sample_n_sequences,
                sequence_counts_scaling_fn=no_sequence_count_scaling,
                one_loader=True
            )
            return dataloader

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
        self.model = DeepRC(max_seq_len=60, sequence_embedding_network=sequence_embedding_network,
                            attention_network=attention_network,
                            output_network=output_network,
                            consider_seq_counts=False, n_input_features=20, add_positional_information=True,
                            sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
                            device=self.device).to(device=self.device)

    # Train DeepRC model
    def train(self):
        train(self.model, task_definition=self.task_definition, trainingset_dataloader=self.trainingset,
              trainingset_eval_dataloader=self.trainingset_eval, learning_rate=self.learning_rate,
              early_stopping_target_id='signal_disease',  # Get model that performs best for this task
              validationset_eval_dataloader=self.validationset_eval, n_updates=self.n_updates,
              evaluate_at=self.evaluate_at,
              device=self.device, results_directory="results/immuneml_may_singletask_cnn_interface"
              # Here our results and trained models will be stored
              )
        # Evaluate trained model on testset
        scores = evaluate(model=self.model, dataloader=self.testset_eval, task_definition=self.task_definition,
                          device=self.device)
        print(f"Test scores:\n{scores}")

    def fit(self, data):
        self.create_task_definitions()
        self.get_dataset(data["metadata_filepath"], data["dataset_filepath"])
        self.create_network()
        self.train()
        return {"status": "finished train"}

    def predict(self, data):
        dataset = self.get_dataset(data["metadata_filepath"], data["dataset_filepath"], one_loader=True)
        pred = self.get_predictions(self.model, dataset)

        return {"signal_disease": (pred > 0.5)}

    def predict_proba(self, data):
        dataset = self.get_dataset(data["metadata_filepath"], data["dataset_filepath"], one_loader=True)
        pred = self.get_predictions(self.model, dataset)

        return {"signal_disease": {True: pred, False: 1 - pred}}

    # from training.py
    @staticmethod
    def get_predictions(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                        show_progress: bool = True, device: torch.device = torch.device('cuda:0')) -> dict:
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

        return np.squeeze(scoring_predictions)


if __name__ == '__main__':
    pass
    #a = Interface()
    #data = {"metadata_filepath": "datasets/immuneml_testset2/metadata.tsv",
    #        "dataset_filepath": "datasets/immuneml_testset2/repertoires"}

    #a.fit(data)
    #a.predict()
    # a.predict_proba()
    # a.create_task_definitions()
    # a.get_dataset()
    # a.create_network()
    # a.train()
