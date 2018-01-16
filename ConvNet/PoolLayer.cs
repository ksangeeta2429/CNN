using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Core.Layers
{
    public class PoolLayer: BaseLayer
    {
        //Keras Equivalent: MaxPooling2D(pool_size=(2, 2))
        public PoolLayer(Dictionary<string, object> data) : base(data)
        {
            //padding is assumed to be "valid"
            this.KernelSizes = (Tuple<int, int>)data["kernel_size"];
            this.Strides = (Tuple<int, int>)data["strides"];
        }
        public Tuple<int, int> KernelSizes { get; set; }
        public Tuple<int, int> Strides { get; set;  }

        public override void Init(int batch, int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(batch, inputWidth, inputHeight, inputDepth);
            UpdateOutputSize();
        }

        private void UpdateOutputSize()
        {
            this.OutputDepth = this.InputDepth;
            this.OutputHeight = Convert.ToInt16(1 + (this.InputHeight - KernelSizes.Item1) / Strides.Item1);
            this.OutputWidth = Convert.ToInt16(1 + (this.InputWidth - KernelSizes.Item2) / Strides.Item2);
        }
        protected override List<List<Slice>> Forward(List<List<Slice>> input, bool isTraining=false)
        {
            List<List<Slice>> A = new List<List<Slice>>();
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
                            double[] a_slice_prev = new double[input[i].Count];
                            int vert_start = h * Strides.Item1;
                            int vert_end = vert_start + KernelSizes.Item1 - 1;
                            int horiz_start = w * Strides.Item2;
                            int horiz_end = horiz_start + KernelSizes.Item2 - 1;
                            //Console.WriteLine(vert_start + " " + vert_end + " " + horiz_start + " " + horiz_end);
                            for (int col = 0; col < input[i].Count; col++)
                            {
                                a_slice_prev[col] = input[i][col].getMaxFromRegion(vert_start, vert_end, horiz_start, horiz_end);
                            }
                            values[w] = a_slice_prev.Max();
                        }
                        s.setValue(values);
                    }//end of height i.e. slice
                    slices.Add(s);
                }//end of channels i.e. c
                A.Add(slices);
            }//end og #data points i.e. m
            return A;
        }

    }
}
