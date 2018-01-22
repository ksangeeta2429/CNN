using System;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using ConvNet.Core;
using System.IO;
using System.Linq;
using ConvNet.Core.Layers;

namespace ConvNet.Test
{
    public class Weights
    {
        public Weights() { }
        public List<List<Slice>> Conv1 { get; set; }
        public List<List<Slice>> Conv2 { get; set; }
        public List<double[]> Dense1 { get; set; }
        public List<double[]> Dense2 { get; set; }
        public double[] Bias_Conv1 { get; set; }
        public double[] Bias_Conv2 { get; set; }
        public double[] Bias_Dense1 { get; set; }
        public double[] Bias_Dense2 { get; set; }
    }
    class CNN
    {
        public static List<int> predictions = new List<int>();
        public static void Main()
        {
            List<List<Slice>> Test = getDataTensor();
            //printDimensions(Test);

            Weights wt = readJSON();
            /*Model net = new Model();

            //Prepare parameters for Layers
            net.AddLayer(new ConvLayer(wt.Conv1, wt.Bias_Conv1, new Dictionary<string, object> { ["filter_count"] = 32, ["kernel_size"] = new Tuple<int, int>(1, 8), ["strides"] = new Tuple<int, int>(1, 4), ["activation"] = "relu" }));
            net.AddLayer(new ConvLayer(wt.Conv2, wt.Bias_Conv2, new Dictionary<string, object> { ["filter_count"] = 32, ["kernel_size"] = new Tuple<int, int>(1, 3), ["strides"] = new Tuple<int, int>(1, 1), ["activation"] = "relu" }));
            net.AddLayer(new PoolLayer(new Dictionary<string, object> { ["kernel_size"] = new Tuple<int, int>(2, 2), ["strides"] = new Tuple<int, int>(1, 1)}));
            net.AddLayer(new FlattenLayer());
            net.AddLayer(new DenseLayer(wt.Dense, wt.Bias_Dense, new Dictionary<string, object> { ["neurons"] = 512, ["activation"]= "relu"}));
            net.AddLayer(new DenseLayer(wt.LastWt, new double[1] { 0 }, new Dictionary<string, object> { ["neurons"] = 1, ["activation"] = "sigmoid" }));

            var preds = net.Forward(Test);*/
        }

        public static List<List<Slice>> getDataTensor()
        {
            int lastCol = 512;
            Tuple<int, int> dimension = new Tuple<int, int>(16, 32);
            List<List<Slice>> tensor = new List<List<Slice>>();
            string csvData = File.ReadAllText("c:/users/sangeeta/desktop/ConvNet/cow_human.txt");
            foreach (string row in csvData.Split('\n'))
            {
                List<Slice> list = new List<Slice>();
                Slice s = new Slice(dimension.Item1, dimension.Item2);
                if (!string.IsNullOrEmpty(row))
                {
                    double[] values = new double[dimension.Item2];
                    int i = 0; //for Column count
                    int w = 0; // Column count for 2D data
                    foreach (string cell in row.Split(','))
                    {
                        if (i == lastCol)
                            predictions.Add(Convert.ToInt16(cell));
                        else
                        {
                            if(w < dimension.Item2)
                            {
                                values[w] = Convert.ToDouble(cell);
                            }
                            else
                            {
                                s.setValue(values);
                                w = 0;
                                values[w] = Convert.ToDouble(cell);
                            }
                            w++;
                        }
                        i++;
                    } //end of inner foreach
                }
                list.Add(s);
                tensor.Add(list);
            }//end of outer foreach
            return tensor;
        }

        public static void printDimensions(List<List<Slice>> Test)
        {
            Console.WriteLine(Test.Count);
            Console.WriteLine(Test[0].Count);
            Console.WriteLine(Test[0][0].Height);
            Console.WriteLine(Test[0][0].Width);
            Console.ReadLine();
        }

        public static Weights readJSON()
        {
            Weights wt = new Test.Weights();
            StreamReader r = new StreamReader("c:/users/sangeeta/desktop/ConvNet/weights.json");
            string json = r.ReadToEnd();

            JObject o = JObject.Parse(json);
            wt.Conv1 = formTensor((JArray)o["conv1"]);
            wt.Conv2 = formTensor((JArray)o["conv2"]);
            wt.Bias_Conv1 = ((JArray)o["bias_conv1"]).Select(jv => (double)jv).ToArray();
            wt.Bias_Conv2 = ((JArray)o["bias_conv2"]).Select(jv => (double)jv).ToArray();
            wt.Bias_Dense1 = ((JArray)o["bias_dense1"]).Select(jv => (double)jv).ToArray();
            wt.Bias_Dense2 = ((JArray)o["bias_dense2"]).Select(jv => (double)jv).ToArray();
            wt.Dense1 = new List<double[]>();
            wt.Dense2 = new List<double[]>();

            //Transpose the FC Weights and save
            for (int i=0; i < ((JArray)o["dense1"][0]).Count; i++)
            {
                double[] arr = new double[((JArray)o["dense1"]).Count];
                for(int j=0; j < ((JArray)o["dense1"]).Count; j++)
                {
                    arr[j] = (double)o["dense1"][j][i];
                }
                wt.Dense1.Add(arr);
            }

            for (int i = 0; i < ((JArray)o["dense2"][0]).Count; i++)
            {
                double[] arr = new double[((JArray)o["dense2"]).Count];
                for (int j = 0; j < ((JArray)o["dense2"]).Count; j++)
                {
                    arr[j] = (double)o["dense2"][j][i];
                }
                wt.Dense2.Add(arr);
            }

            Console.ReadKey();
            return wt;
        }

        public static List<List<Slice>> formTensor(JArray data)
        {
            List<List<Slice>> T = new List<List<Slice>>();
            //Array's shape: (f_H, f_W, n_C_Prev, n_C)
            //Convert to: (n_C, n_C_prev, f_H, f_W)
            int f_H = data.Count;
            int f_W = ((JArray)data[0]).Count;
            int n_C_prev = ((JArray)data[0][0]).Count;
            int n_C = ((JArray)data[0][0][0]).Count;
            for (int i=0; i<n_C; i++)
            {
                List<Slice> lst = new List<Slice>();
                for (int j=0; j<n_C_prev; j++)
                {
                    Slice s = new Core.Slice(f_H, f_W);
                    for(int h=0; h < f_H; h++)
                    {
                        double[] values = new double[f_W];
                        for(int w=0; w < f_W; w++)
                        {
                            values[w] = (double)data[h][w][j][i];
                        }
                        s.setValue(values);
                    }
                    lst.Add(s);
                }
                T.Add(lst);
            }
            return T;
        }
    }
}
