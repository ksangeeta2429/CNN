using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Core.Layers
{
    public abstract class BaseLayer
    {
        protected BaseLayer()
        {
        }

        protected BaseLayer(Dictionary<string, object> data)
        {
            this.BatchSize = Convert.ToInt16(data["batch_size"]);
            this.InputHeight = Convert.ToInt16(data["InputHeight"]);
            this.InputWidth = Convert.ToInt16(data["InputWidth"]);
            this.InputDepth = Convert.ToInt16(data["InputDepth"]);
            this.OutputHeight = Convert.ToInt16(data["OutputHeight"]);
            this.OutputWidth = Convert.ToInt16(data["OutputWidth"]);
            this.OutputDepth = Convert.ToInt16(data["OutputDepth"]);
        }

        public List<List<Slice>> Input { get; protected set; }

        public List<List<Slice>> Output { get; protected set; }

        public int OutputHeight { get; protected set; }

        public int OutputWidth { get; protected set; }

        public int OutputDepth { get; protected set; }

        public int InputHeight { get; private set; }

        public int InputWidth { get; private set; }

        public int InputDepth { get; private set; }

        public int BatchSize { get; set; }

        public BaseLayer Child { get; set; }

        public List<BaseLayer> Parents { get; set; } = new List<BaseLayer>();

        public virtual void Init(int batch, int inputWidth, int inputHeight, int inputDepth)
        {
            this.BatchSize = batch;
            this.InputWidth = inputWidth;
            this.InputHeight = inputHeight;
            this.InputDepth = inputDepth;
        }

        internal void ConnectTo(BaseLayer layer)
        {
            this.Child = layer;
            layer.Parents.Add(this);
            layer.Init(this.BatchSize, this.OutputWidth, this.OutputHeight, this.OutputDepth);
        }

        public virtual List<List<Slice>> ForwardPass(List<List<Slice>> input, bool isTraining = false)
        {
            this.Input = input;
            this.Output = Forward(input, isTraining);
            return this.Output;
        }

        //To be overridden in each Layer class
        protected abstract List<List<Slice>> Forward(List<List<Slice>> input, bool isTraining = false);
    }
}
