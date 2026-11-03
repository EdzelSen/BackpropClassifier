using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Backprop
{
    /// <summary>
    /// UI for loading a CSV, training the backprop network, viewing the loss curve,
    /// evaluating on the test set, and running a hyperparameter grid search.
    /// The whole UI is built in code, so no designer work is needed.
    /// </summary>
    public class MainForm : Form
    {
        // ---- controls ----
        private readonly Button btnLoad = new Button { Text = "Load CSV", AutoSize = true };
        private readonly Button btnTrain = new Button { Text = "Train", AutoSize = true };
        private readonly Button btnStop = new Button { Text = "Stop", AutoSize = true, Enabled = false };
        private readonly Button btnEval = new Button { Text = "Evaluate", AutoSize = true };
        private readonly Button btnGrid = new Button { Text = "Run Grid Search", AutoSize = true };
        private readonly Button btnExport = new Button { Text = "Export Log CSV", AutoSize = true };
        private readonly NumericUpDown numHidden = Num(1, 512, 8, 0, 1);
        private readonly NumericUpDown numEpochs = Num(1, 1000000, 500, 0, 100);
        private readonly NumericUpDown numLrOut = Num(0.001m, 5m, 0.2m, 3, 0.05m);
        private readonly NumericUpDown numLrIn = Num(0.001m, 5m, 0.15m, 3, 0.05m);
        private readonly NumericUpDown numLogEvery = Num(1, 100000, 50, 0, 10);
        private readonly NumericUpDown numTrainPct = Num(50, 95, 80, 0, 5);
        private readonly TextBox txtLog = new TextBox
        {
            Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical,
            Dock = DockStyle.Fill, Font = new Font("Consolas", 9f), WordWrap = false
        };
        private readonly Panel chart = new Panel { Dock = DockStyle.Fill, BackColor = Color.White };

        // ---- state ----
        private Dataset? _all, _train, _test;
        private NeuralNet? _net;
        private Trainer? _trainer;
        private readonly List<(int epoch, double train, double test)> _history = new();
        private CancellationTokenSource? _cts;
        private double[] _min = Array.Empty<double>(), _max = Array.Empty<double>();

        public MainForm()
        {
            Text = "Backprop MLP Classifier";
            Width = 1100;
            Height = 700;

            var top = new FlowLayoutPanel { Dock = DockStyle.Top, AutoSize = true, Padding = new Padding(6) };
            top.Controls.AddRange(new Control[]
            {
                btnLoad, Lbl("Train %"), numTrainPct,
                Lbl("Hidden"), numHidden, Lbl("Epochs"), numEpochs,
                Lbl("LR out"), numLrOut, Lbl("LR in"), numLrIn, Lbl("Log every"), numLogEvery,
                btnTrain, btnStop, btnEval, btnGrid, btnExport
            });

            var split = new SplitContainer { Dock = DockStyle.Fill, SplitterDistance = 520 };
            split.Panel1.Controls.Add(txtLog);
            split.Panel2.Controls.Add(chart);

            Controls.Add(split);
            Controls.Add(top);

            btnLoad.Click += (s, e) => LoadCsv();
            btnTrain.Click += async (s, e) => await TrainAsync();
            btnStop.Click += (s, e) => _cts?.Cancel();
            btnEval.Click += (s, e) => Evaluate();
            btnGrid.Click += async (s, e) => await GridSearchAsync();
            btnExport.Click += (s, e) => ExportLog();
            chart.Paint += DrawChart;
            chart.Resize += (s, e) => chart.Invalidate();
        }

        // ---------- helpers ----------
        private static NumericUpDown Num(decimal min, decimal max, decimal val, int dec, decimal inc) =>
            new NumericUpDown
            {
                Minimum = min, Maximum = max, Value = val, DecimalPlaces = dec,
                Increment = inc, Width = 70
            };

        private static Label Lbl(string t) =>
            new Label { Text = t, AutoSize = true, Margin = new Padding(8, 7, 0, 0) };

        private void Log(string msg) => txtLog.AppendText(msg + "\r\n");

        private bool HasData()
        {
            if (_train == null || _test == null) { MessageBox.Show("Load a CSV first."); return false; }
            return true;
        }

        private void SetBusy(bool busy)
        {
            btnTrain.Enabled = btnGrid.Enabled = btnLoad.Enabled = !busy;
            btnStop.Enabled = busy;
        }

        // ---------- load + preprocess ----------
        private void LoadCsv()
        {
            using var dlg = new OpenFileDialog { Filter = "CSV files|*.csv|All files|*.*" };
            if (dlg.ShowDialog() != DialogResult.OK) return;

            try
            {
                _all = DataPipeline.Load(dlg.FileName, true, out int skipped);
                (_train, _test) = DataPipeline.Split(_all, (double)numTrainPct.Value / 100.0);
                (_min, _max) = DataPipeline.FitMinMax(_train.X);   // fit on TRAIN only
                DataPipeline.Scale(_train.X, _min, _max);
                DataPipeline.Scale(_test.X, _min, _max);

                Log($"Loaded {Path.GetFileName(dlg.FileName)}: {_all.X.Length} rows, " +
                    $"{_all.X[0].Length} features, {_all.Classes.Count} classes ({string.Join(", ", _all.Classes)})");
                Log($"Skipped {skipped} invalid rows | train {_train.X.Length} / test {_test.X.Length} | MinMax scaled to [0,1]");
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not load CSV: " + ex.Message);
            }
        }

        // ---------- training ----------
        private NeuralNet BuildNet(int hidden, double lrOut, double lrIn)
        {
            var net = new NeuralNet(_train!.X[0].Length, hidden, _train.Classes.Count);
            net.LRPOUT = lrOut;
            net.LRPIN = lrIn;
            return net;
        }

        private async Task TrainAsync()
        {
            if (!HasData()) return;

            int hidden = (int)numHidden.Value, epochs = (int)numEpochs.Value, every = (int)numLogEvery.Value;
            _net = BuildNet(hidden, (double)numLrOut.Value, (double)numLrIn.Value);
            _trainer = new Trainer(_net, _train!.Classes.Count);
            _history.Clear();
            _cts = new CancellationTokenSource();
            var token = _cts.Token;
            var trainer = _trainer;

            Log($"\r\n--- Training: hidden={hidden}, LRout={numLrOut.Value}, LRin={numLrIn.Value}, epochs={epochs} ---");
            SetBusy(true);
            try
            {
                await Task.Run(() => trainer.Train(_train, _test!, epochs, every, (ep, tr, te) =>
                {
                    BeginInvoke(new Action(() =>
                    {
                        _history.Add((ep, tr, te));
                        Log($"Epoch {ep,6} | train MSE {tr:F6} | test MSE {te:F6}");
                        chart.Invalidate();
                    }));
                }, token));
            }
            finally { SetBusy(false); }

            Evaluate();
        }

        // ---------- evaluation ----------
        private void Evaluate()
        {
            if (_trainer == null || _test == null) { MessageBox.Show("Train a model first."); return; }

            var cm = _trainer.Confusion(_test);
            var (acc, p, r, f1) = Trainer.Metrics(cm);
            int n = _test.Classes.Count;

            Log($"\r\n=== Test evaluation ({_test.X.Length} samples) ===");
            Log($"Accuracy: {acc:P1} | Test MSE: {_trainer.Mse(_test):F6}");
            Log($"{"Class",-16}{"Precision",10}{"Recall",10}{"F1",10}");
            for (int c = 0; c < n; c++)
                Log($"{_test.Classes[c],-16}{p[c],10:F3}{r[c],10:F3}{f1[c],10:F3}");

            Log("Confusion matrix (rows = actual, cols = predicted):");
            Log("".PadRight(16) + string.Join("", _test.Classes.Select(c => c.Length > 9 ? c.Substring(0, 9).PadLeft(10) : c.PadLeft(10))));
            for (int a = 0; a < n; a++)
            {
                var sb = new StringBuilder(_test.Classes[a].PadRight(16));
                for (int q = 0; q < n; q++) sb.Append(cm[a, q].ToString().PadLeft(10));
                Log(sb.ToString());
            }
        }

        // ---------- grid search (hyperparameter tuning) ----------
        private async Task GridSearchAsync()
        {
            if (!HasData()) return;

            int[] hiddenSizes = { 4, 8, 16 };
            double[] learningRates = { 0.05, 0.2, 0.5 };
            int epochs = (int)numEpochs.Value;
            var train = _train!; var test = _test!;
            int outputs = train.Classes.Count;

            Log($"\r\n=== Grid search ({epochs} epochs each) ===");
            Log($"{"Hidden",6}{"LR(out=in)",12}{"TrainMSE",12}{"TestMSE",12}{"TestAcc",10}");
            SetBusy(true);

            var csv = new StringBuilder("hidden,lr,epochs,train_mse,test_mse,test_accuracy\r\n");
            try
            {
                foreach (int h in hiddenSizes)
                {
                    foreach (double lr in learningRates)
                    {
                        var result = await Task.Run(() =>
                        {
                            var net = BuildNet(h, lr, lr);
                            var t = new Trainer(net, outputs);
                            t.Train(train, test, epochs, epochs, (a, b, c) => { });
                            var (acc, _, _, _) = Trainer.Metrics(t.Confusion(test));
                            return (tr: t.Mse(train), te: t.Mse(test), acc);
                        });
                        Log($"{h,6}{lr,12:F2}{result.tr,12:F6}{result.te,12:F6}{result.acc,10:P1}");
                        csv.AppendLine($"{h},{lr},{epochs},{result.tr},{result.te},{result.acc}");
                    }
                }
            }
            finally { SetBusy(false); }

            string path = Path.Combine(AppContext.BaseDirectory, "tuning_results.csv");
            File.WriteAllText(path, csv.ToString());
            Log("Saved grid results to: " + path);
        }

        // ---------- export ----------
        private void ExportLog()
        {
            if (_history.Count == 0) { MessageBox.Show("Nothing to export. Train first."); return; }
            using var dlg = new SaveFileDialog { Filter = "CSV|*.csv", FileName = "training_log.csv" };
            if (dlg.ShowDialog() != DialogResult.OK) return;

            var lines = new List<string> { "epoch,train_mse,test_mse" };
            lines.AddRange(_history.Select(h => $"{h.epoch},{h.train},{h.test}"));
            File.WriteAllLines(dlg.FileName, lines);
            Log("Exported log to " + dlg.FileName);
        }

        // ---------- loss chart (custom drawn, no NuGet needed) ----------
        private void DrawChart(object? sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(Color.White);
            var area = new Rectangle(55, 25, chart.Width - 80, chart.Height - 60);
            if (area.Width < 50 || area.Height < 50) return;

            g.DrawRectangle(Pens.Gray, area);
            g.DrawString("Loss (MSE) vs Epoch", Font, Brushes.Black, 55, 5);
            if (_history.Count < 2) return;

            double maxLoss = _history.Max(h => Math.Max(h.train, h.test));
            int maxEpoch = _history.Max(h => h.epoch);
            if (maxLoss <= 0) maxLoss = 1;

            PointF Pt(int epoch, double loss) => new PointF(
                area.Left + (float)epoch / maxEpoch * area.Width,
                area.Bottom - (float)(loss / maxLoss) * area.Height);

            using var trainPen = new Pen(Color.RoyalBlue, 2);
            using var testPen = new Pen(Color.OrangeRed, 2);
            g.DrawLines(trainPen, _history.Select(h => Pt(h.epoch, h.train)).ToArray());
            g.DrawLines(testPen, _history.Select(h => Pt(h.epoch, h.test)).ToArray());

            g.DrawString(maxLoss.ToString("F4"), Font, Brushes.Black, 5, area.Top - 5);
            g.DrawString("0", Font, Brushes.Black, 35, area.Bottom - 8);
            g.DrawString(maxEpoch.ToString(), Font, Brushes.Black, area.Right - 25, area.Bottom + 4);
            g.DrawString("Train", Font, Brushes.RoyalBlue, area.Right - 110, area.Top + 5);
            g.DrawString("Test", Font, Brushes.OrangeRed, area.Right - 60, area.Top + 5);
        }
    }
}
