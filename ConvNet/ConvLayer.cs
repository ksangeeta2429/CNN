using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Core.Layers
{
    public class ConvLayer: BaseLayer
    {
        //Keras Equivalent: Conv2D(32, kernel_size=(1, 8), strides=(1, 4), input_shape=(16, 32, 1), padding='same', activation='relu')
        public ConvLayer(List<List<Slice>> kernelWts, double[] bias, Dictionary<string, object> data) : base(data)
        {
            this.Kernels = kernelWts;
            this.Bias = bias;
            this.FilterCount = Convert.ToInt32(data["filter_count"]);
            this.KernelSizes = (Tuple<int, int>) data["kernel_size"];
            this.Strides = (Tuple<int, int>)data["strides"];
            this.Pad = CalculatePad(KernelSizes);
            this.Activation = Convert.ToString(data["activation"]);
        }

        public string Activation { get; set; }

        public double[] Bias { get; private set; }

        public List<List<Slice>> Kernels { get; private set; }

        public Tuple<int,int> Pad { get; private set; }

        public Tuple<int,int> Strides { get; private set; }

        public Tuple<int, int> KernelSizes { get; private set; }

        public int FilterCount { get; }

        public override void Init(int batch, int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(batch, inputWidth, inputHeight, inputDepth);
            UpdateOutputSize();
        }

        internal void UpdateOutputSize()
        {
            this.OutputDepth = this.FilterCount;
            this.OutputHeight = Convert.ToInt16((this.InputHeight - KernelSizes.Item1 + (2 * Pad.Item1)) / Strides.Item1) + 1;
            this.OutputWidth = Convert.ToInt16((this.InputWidth - KernelSizes.Item2 + (2 * Pad.Item2)) / Strides.Item2) + 1;
        }

        public Tuple<int,int> CalculatePad(Tuple<int, int> KernelSizes)
        {
            Tuple<int, int> pad = Tuple.Create((KernelSizes.Item1-1)/2,(KernelSizes.Item2)/2);
            return pad;
        }

        public List<List<Slice>> zero_pad(List<List<Slice>> X)
        {
            for (int i = 0; i < X.Count; i++)
                for (int j = 0; j < X[0].Count; j++)
                {
                    X[i][j].resize(Pad);
                }
            return X;
        }

        public static double convolve_single(List<Slice> slice, List<Slice> Weight, double b)
        {
            double sum = 0;
            for (int j = 0; j < slice.Count; j++)
                for (int k = 0; k < slice[0].Height; k++)
                    for (int l = 0; l < slice[0].Width; l++)
                        sum = sum + slice[j].getValue(k, l) * Weight[j].getValue(k, l);

            return sum + b;
        } // end of convolve_single

        protected override List<List<Slice>> Forward(List<List<Slice>> input, bool isTraining = false)
        {
            List<List<Slice>> Z = new List<List<Slice>>();
            List<List<Slice>> A_prev_pad = zero_pad(input);
            for (int i = 0; i < BatchSize; i++)
            {
                List<Slice> slices = new List<Slice>();
                for (int c = 0; c < OutputDepth; c++)
                {
                    Slice s = new Slice(OutputHeight, OutputWidth);
                    for (int h = 0; h < OutputHeight; h++)
                    {
                        double[] values = new double[OutputWidth];
                        for (int w = 0; w < OutputWidth; w++)
                        {
                            List<Slice> a_slice_prev = new List<Slice>();
                            int vert_start = h * Strides.Item1;
                            int vert_end = vert_start + KernelSizes.Item1 - 1;
                            int horiz_start = w * Strides.Item2;
                            int horiz_end = horiz_start + KernelSizes.Item2 - 1;

                            for (int col = 0; col < A_prev_pad[i].Count; col++)
                            {
                                a_slice_prev.Add(A_prev_pad[i][col].getRegion(vert_start, vert_end, horiz_start, horiz_end));
                            }
                            if(this.Activation == "relu")
                                values[w] = convolve_single(a_slice_prev, Kernels[c], Bias[c])<=0?0: convolve_single(a_slice_prev, Kernels[c], Bias[c]);
                            if(this.Activation == "sigmoid")
                            {
                                double x = convolve_single(a_slice_prev, Kernels[c], Bias[c]);
                                values[w] = 1.0 / (1.0 + Math.Exp(-x));
                            }
                        }
                        s.setValue(values);
                    }//end of height i.e. slice
                    slices.Add(s);
                }//end of channels i.e. c
                Z.Add(slices);
            }//end of #data points i.e. m
            return Z;
        }//end of conv_forward function
    }
}
