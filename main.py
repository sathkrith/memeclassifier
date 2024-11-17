import importlib
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a classifier.")
    parser.add_argument("--task", type=str, required=True, help="Task to perform (e.g., 'multiclass_meme', 'binary_meme').")
    parser.add_argument("--train_folder", type=str, required=True, help="Path to the training data folder.")
    parser.add_argument("--dev_folder", type=str, required=True, help="Path to the validation data folder.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    args = parser.parse_args()

    # Dynamically import and run the train function
    try:
        task_module = importlib.import_module(f"tasks.{args.task}.model")
    except ModuleNotFoundError:
        raise ValueError(f"Task '{args.task}' not implemented.")

    task_module.train(args)


if __name__ == "__main__":
    main()
