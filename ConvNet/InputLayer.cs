using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Core.Layers
{
    public class InputLayer: BaseLayer
    {
        public InputLayer(Dictionary<string, object> data) : base(data)
        {
            this.BatchSize = this.BatchSize;
            this.OutputWidth = this.InputWidth;
            this.OutputHeight = this.InputHeight;
            this.OutputDepth = this.InputDepth;
        }

        public InputLayer(int batch, int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(batch, inputWidth, inputHeight, inputDepth);
            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
            this.OutputDepth = inputDepth;
        }

        protected override List<List<Slice>> Forward(List<List<Slice>> input, bool isTraining = false)
        {
            return input;
        }
    }
}
