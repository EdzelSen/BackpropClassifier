using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace Backprop
{
    /// <summary>Holds features (X) and one-hot encoded labels (Y).</summary>
    public class Dataset
    {
        public double[][] X = Array.Empty<double[]>();
        public double[][] Y = Array.Empty<double[]>();
        public List<string> Classes = new List<string>();
    }

    /// <summary>CSV parsing, MinMax scaling and train/test splitting.</summary>
    public static class DataPipeline
    {
        /// <summary>
        /// Loads a CSV where the LAST column is the class label and all other
        /// columns are numeric features. Rows with missing/invalid numbers are skipped.
        /// </summary>
        public static Dataset Load(string path, bool hasHeader, out int skippedRows)
        {
            var rows = new List<double[]>();
            var labels = new List<string>();
            skippedRows = 0;

            foreach (var line in File.ReadLines(path).Skip(hasHeader ? 1 : 0))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(',');
                if (parts.Length < 2) { skippedRows++; continue; }

                var features = new double[parts.Length - 1];
                bool valid = true;
                for (int i = 0; i < features.Length; i++)
                {
                    if (!double.TryParse(parts[i], NumberStyles.Float,
                                         CultureInfo.InvariantCulture, out features[i]))
                    {
                        valid = false;
                        break;
                    }
                }
                string label = parts[parts.Length - 1].Trim();
                if (!valid || label.Length == 0) { skippedRows++; continue; }

                rows.Add(features);
                labels.Add(label);
            }

            var ds = new Dataset { Classes = labels.Distinct().OrderBy(c => c).ToList() };
            ds.X = rows.ToArray();
            ds.Y = labels.Select(l =>
            {
                var oneHot = new double[ds.Classes.Count];
                oneHot[ds.Classes.IndexOf(l)] = 1.0;
                return oneHot;
            }).ToArray();
            return ds;
        }

        /// <summary>Finds per-feature min and max. Fit on TRAIN data only to avoid leakage.</summary>
        public static (double[] min, double[] max) FitMinMax(double[][] train)
        {
            int n = train[0].Length;
            var min = new double[n];
            var max = new double[n];
            for (int j = 0; j < n; j++)
            {
                min[j] = train.Min(r => r[j]);
                max[j] = train.Max(r => r[j]);
            }
            return (min, max);
        }

        /// <summary>Scales features in place to [0,1] using the given min/max.</summary>
        public static void Scale(double[][] data, double[] min, double[] max)
        {
            foreach (var row in data)
                for (int j = 0; j < row.Length; j++)
                    row[j] = max[j] == min[j] ? 0.0 : (row[j] - min[j]) / (max[j] - min[j]);
        }

        /// <summary>Shuffled split into train/test (e.g. trainRatio = 0.8).</summary>
        public static (Dataset train, Dataset test) Split(Dataset ds, double trainRatio, int seed = 42)
        {
            var rng = new Random(seed);
            var idx = Enumerable.Range(0, ds.X.Length).OrderBy(_ => rng.Next()).ToArray();
            int cut = (int)(idx.Length * trainRatio);

            Dataset Pick(IEnumerable<int> ids) => new Dataset
            {
                X = ids.Select(i => (double[])ds.X[i].Clone()).ToArray(),
                Y = ids.Select(i => ds.Y[i]).ToArray(),
                Classes = ds.Classes
            };
            return (Pick(idx.Take(cut)), Pick(idx.Skip(cut)));
        }
    }
}
