using System;
using System.Collections.Generic;
using System.Linq;
using ConvNet.Core.Layers;

namespace ConvNet.Core
{
    class Model
    {
        public List<BaseLayer> Layers = new List<BaseLayer>();

        public Model() { }

        public int[] Forward(List<List<Slice>> input, bool isTraining = false)
        {
            var output = this.Layers[0].ForwardPass(input, isTraining);

            for (var i = 1; i < this.Layers.Count; i++)
            {
                var layer = this.Layers[i];
                output = layer.ForwardPass(output, isTraining);
            }
            return Predictions(output);
        }

        public int[] Predictions(List<List<Slice>> output)
        {
            int[] result = new int[output.Count];
            for (int i = 0; i < output.Count; i++)
                result[i] = Convert.ToInt16(output[i][0].getValue(0,0));
            return result;
        }

        public void AddLayer(BaseLayer layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0, batch = 0;
            BaseLayer lastLayer = null;

            if (this.Layers.Count > 0)
            {
                batch = this.Layers[this.Layers.Count - 1].BatchSize;
                inputWidth = this.Layers[this.Layers.Count - 1].OutputWidth;
                inputHeight = this.Layers[this.Layers.Count - 1].OutputHeight;
                inputDepth = this.Layers[this.Layers.Count - 1].OutputDepth;
                lastLayer = this.Layers[this.Layers.Count - 1];
            }

            if (this.Layers.Count > 0)
            {
                layer.Init(batch, inputWidth, inputHeight, inputDepth);
            }
            this.Layers.Add(layer);
        }
    }
}
