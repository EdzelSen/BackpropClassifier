using System;
using System.Linq;
using System.Threading;

namespace Backprop
{
    /// <summary>Training loop, loss (MSE) calculation and classification metrics
    /// built on top of the given NeuralNet class.</summary>
    public class Trainer
    {
        private readonly NeuralNet _net;
        private readonly int _outputs;

        public Trainer(NeuralNet net, int outputs)
        {
            _net = net;
            _outputs = outputs;
        }

        private void Feed(double[] x)
        {
            for (int i = 0; i < x.Length; i++) _net.setInputs(i, x[i]);
        }

        /// <summary>Mean squared error over a dataset (forward pass only).</summary>
        public double Mse(Dataset ds)
        {
            double sum = 0;
            for (int s = 0; s < ds.X.Length; s++)
            {
                Feed(ds.X[s]);
                _net.run();
                for (int k = 0; k < _outputs; k++)
                {
                    double e = ds.Y[s][k] - _net.getOuputData(k);
                    sum += e * e;
                }
            }
            return sum / (ds.X.Length * _outputs);
        }

        /// <summary>Index of the output neuron with the highest activation.</summary>
        public int Predict(double[] x)
        {
            Feed(x);
            _net.run();
            int best = 0;
            for (int k = 1; k < _outputs; k++)
                if (_net.getOuputData(k) > _net.getOuputData(best)) best = k;
            return best;
        }

        /// <summary>Online (per-sample) backprop with shuffling every epoch.
        /// onLog(epoch, trainMse, testMse) fires on epoch 1 and every logEvery epochs.</summary>
        public void Train(Dataset train, Dataset test, int epochs, int logEvery,
                          Action<int, double, double> onLog, CancellationToken ct = default)
        {
            var rng = new Random(1);
            var order = Enumerable.Range(0, train.X.Length).ToArray();

            for (int epoch = 1; epoch <= epochs && !ct.IsCancellationRequested; epoch++)
            {
                order = order.OrderBy(_ => rng.Next()).ToArray();
                foreach (int s in order)
                {
                    Feed(train.X[s]);
                    for (int k = 0; k < _outputs; k++)
                        _net.setDesiredOutput(k, train.Y[s][k]);
                    _net.learn();   // forward pass + backprop weight update
                }
                if (epoch == 1 || epoch % logEvery == 0 || epoch == epochs)
                    onLog(epoch, Mse(train), Mse(test));
            }
        }

        /// <summary>Confusion matrix indexed [actual, predicted].</summary>
        public int[,] Confusion(Dataset ds)
        {
            var cm = new int[_outputs, _outputs];
            for (int s = 0; s < ds.X.Length; s++)
            {
                int actual = Array.IndexOf(ds.Y[s], 1.0);
                cm[actual, Predict(ds.X[s])]++;
            }
            return cm;
        }

        /// <summary>Accuracy plus per-class precision, recall and F1.</summary>
        public static (double acc, double[] precision, double[] recall, double[] f1) Metrics(int[,] cm)
        {
            int n = cm.GetLength(0);
            int total = 0, correct = 0;
            var p = new double[n];
            var r = new double[n];
            var f = new double[n];
            for (int c = 0; c < n; c++)
            {
                int tp = cm[c, c], rowSum = 0, colSum = 0;
                for (int j = 0; j < n; j++)
                {
                    rowSum += cm[c, j];
                    colSum += cm[j, c];
                    total += cm[c, j];
                }
                correct += tp;
                p[c] = colSum == 0 ? 0 : (double)tp / colSum;
                r[c] = rowSum == 0 ? 0 : (double)tp / rowSum;
                f[c] = (p[c] + r[c]) == 0 ? 0 : 2 * p[c] * r[c] / (p[c] + r[c]);
            }
            return (total == 0 ? 0 : (double)correct / total, p, r, f);
        }
    }
}
