using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Core.Layers
{
    public class FlattenLayer: BaseLayer
    {
        public FlattenLayer() { }
        public FlattenLayer(Dictionary<string, object> data): base(data)
        {

        }
        public override void Init(int batch, int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(batch, inputWidth, inputHeight, inputDepth);
            UpdateOutputSize();
        }

        private void UpdateOutputSize()
        {
            this.OutputDepth = this.InputWidth * this.InputHeight * this.InputDepth;
            this.OutputHeight = 1;
            this.OutputWidth = 1;
        }
        protected override List<List<Slice>> Forward(List<List<Slice>> input, bool isTraining = false)
        {
            List<List<Slice>> Z = new List<List<Slice>>();
            double[] values = new double[1];

            for (int i = 0; i < BatchSize; i++)
            {
                List<Slice> list = new List<Slice>();
                for (int c = 0; c < InputDepth; c++)
                {
                    for (int h = 0; h < InputHeight; h++)
                    {
                        for (int w = 0; w < InputWidth; w++)
                        {
                            Slice s = new Slice(1, 1);
                            values[0] = input[i][c].getValue(h, w);
                            s.setValue(values);
                            list.Add(s); //list.Count = OutputDepth
                        }
                    }
                }
                Z.Add(list);
            }
            return Z;
        }//end of conv_forward function
    }
}
