﻿using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using ConvNet.Core;
using System.IO;
using System.Text;
using System.Linq;

namespace ConvNet.Test
{
    public class Weights
    {
        public Weights() { }
        public List<List<Slice>> Conv1 { get; set; }
        public List<List<Slice>> Conv2 { get; set; }
        public double[] Dense { get; set; }
        public double[] Bias_Conv1 { get; set; }
        public double[] Bias_Conv2 { get; set; }
        public double[] Bias_Dense { get; set; }
        public double[] LastWt { get; set; }
    }
    class CNN
    {
        public static List<int> predictions = new List<int>();
        public static void Main()
        {
            //List<List<Slice>> Test = getDataTensor();
            //printDimensions(Test);

            Weights wt = readJson();
            /*Model net = new Model();
            net.AddLayer(new ConvLayer());
            net.AddLayer(new ConvLayer());
            net.AddLayer(new PoolLayer());
            net.AddLayer(new FlattenLayer());
            net.AddLayer(new DenseLayer());
            net.AddLayer(new DenseLayer());

            var preds = net.Forward(test);*/
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

        public static Weights readJson()
        {
            Weights wt = new Test.Weights();
            StreamReader r = new StreamReader("c:/users/sangeeta/desktop/ConvNet/weights.json");
            string json = r.ReadToEnd();

            JObject o = JObject.Parse(json);
            wt.Conv1 = formTensor((JArray)o["conv1"]);
            wt.Conv2 = formTensor((JArray)o["conv2"]);
            wt.Bias_Conv1 = ((JArray)o["bias1"]).Select(jv => (double)jv).ToArray();
            wt.Bias_Conv2 = ((JArray)o["bias2"]).Select(jv => (double)jv).ToArray();
            wt.Bias_Dense = ((JArray)o["bias3"]).Select(jv => (double)jv[0]).ToArray();
            wt.Dense = ((JArray)o["dense"]).Select(jv => (double)jv).ToArray();
            wt.LastWt[0] = (double)o["last"][0];

            Console.WriteLine(wt.Conv1[0][0].getValue(0, 7));
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