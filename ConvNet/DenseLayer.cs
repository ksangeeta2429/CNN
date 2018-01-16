using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Core.Layers
{
    public class DenseLayer: BaseLayer
    {
        public DenseLayer(double[] W, double[] bias, Dictionary<string, object> data): base(data)
        {
            this.Weights = W;
            this.Bias = bias;
            this.NeuronCount = Convert.ToInt16(data["neurons"]);
            this.Activation = Convert.ToString(data["activation"]);
            if (NeuronCount == 1)
                this.isLast = true;
            else
                this.isLast = false;
        }

        public double[] Weights { get; set;  }
        public double[] Bias { get; set;  }
        public int NeuronCount { get; set; }
        public string Activation { get; set; }
        public bool isLast { get; set; }

        public override void Init(int batch, int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(batch, inputWidth, inputHeight, inputDepth);
            UpdateOutputSize();
        }

        internal void UpdateOutputSize()
        {
            this.OutputWidth = 1;
            this.OutputHeight = 1;
            this.OutputDepth = this.NeuronCount;
        }
        protected override List<List<Slice>> Forward(List<List<Slice>> input, bool isTraining = false)
        {
            List<List<Slice>> Z = new List<List<Slice>>();
            double[] values = new double[1];

            for (int i = 0; i < BatchSize; i++)
            {
                List<Slice> list = new List<Slice>();
                for (int c = 0; c < OutputDepth; c++)
                {
                    Slice s = new Slice(1, 1);
                    double sum = 0.0;
                    for (int d = 0; d < InputDepth; d++)
                        sum = sum + Weights[c] * input[i][d].getValue(0, 0);

                    if (this.Activation == "relu")
                        values[0] = sum + Bias[c]  <= 0 ? 0 : sum + Bias[c];

                    else if (this.Activation == "sigmoid")
                    {
                        if(isLast)
                            values[0] = 1.0 / (1.0 + Math.Exp(-(sum))) > 0.5 ? 1 : 0;
                        else
                            values[0] = 1.0 / (1.0 + Math.Exp(-(sum + Bias[c])));
                    }
                    s.setValue(values);
                    list.Add(s);
                }
                Z.Add(list);
            }
            return Z;
        }//end of conv_forward function
    }
}
