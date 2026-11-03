# Backprop MLP Classifier (C# WinForms, .NET 8)

Multi-class classification using a hand-written 3-layer MLP (input / hidden / output) with backpropagation.
Only the provided classes are used for the network: `INeuron`, `HNeuron`, `ONeuron`, `NeuralNet`.

## Data description
- Dataset: Iris (data/iris.csv) - 150 rows, 4 numeric features (sepal/petal length and width), 3 classes.
- Last CSV column is the class label. (Replace with your own dataset and describe it here.)

## How to run
1. Open the project in Visual Studio 2022 (.NET 8, Windows Forms).
2. Press F5.
3. Click **Load CSV** and choose `data/iris.csv`.
4. Set hidden nodes, epochs, learning rates, then click **Train**. Loss is logged every N epochs and plotted.
5. **Evaluate** shows accuracy, precision, recall, F1 and the confusion matrix on the test set.
6. **Run Grid Search** tests hidden sizes {4, 8, 16} x learning rates {0.05, 0.2, 0.5} and saves `tuning_results.csv`.
7. **Export Log CSV** saves the loss-per-epoch log.

## Pipeline
- Rows with missing/invalid values are skipped.
- Shuffled train/test split (default 80/20, seeded).
- MinMax scaling to [0,1], fitted on the training set only and applied to both sets.
- Labels are one-hot encoded; prediction = output neuron with highest activation.

## Change to the provided engine
`LRPOUT` and `LRPIN` in `BackNeural.cs` changed from `private const` to `public` fields so learning rates can be tuned at runtime. No other engine logic was changed.

## Hyperparameter tuning (fill in from your grid results)
| Hidden | LR | Epochs | Train MSE | Test MSE | Test Acc |
|---|---|---|---|---|---|
| | | | | | |

Explain which configuration you chose and why (convergence speed, overfitting, stability).

## Results / analysis
(Add your final accuracy, confusion matrix, and which classes are confused.)
