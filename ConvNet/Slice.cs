using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Core
{
    public class Slice
    {
        List<double[]> rectangle = new List<double[]>();
        public Slice(int h, int w)
        {
            this.Height = h;
            this.Width = w;
        }

        public int Height { get; set; }
        public int Width { get; set; }

        public void setValue(double[] val)
        {
            rectangle.Add(val);
        }

        public double getValue(int row, int col)
        {
            return rectangle[row][col];
        }

        public List<double[]> getRectangle()
        {
            return rectangle;
        }

        public void printS()
        {
            for (int r = 0; r < Height; r++)
            {
                for (int c = 0; c < Width; c++)
                    Console.Write(rectangle[r][c] + " ");
                Console.WriteLine();
            }
        }

        public Slice getRegion(int v_start, int v_end, int h_start, int h_end)
        {
            List<double[]> slicedList = rectangle.GetRange(v_start, v_end - v_start);

            int r_H = v_end - v_start;
            int r_W = h_end - h_start;

            Slice region = new Slice(r_H, r_W);
            for (int c = 0; c < r_W && h_start < h_end; c++, h_start++)
            {
                double[] arr = slicedList[c];
                region.setValue(arr.Skip(h_start - 1).Take(r_W).ToArray());
            }
            return region;
        }

        public double getMaxFromRegion(int v_start, int v_end, int h_start, int h_end)
        {
            List<double[]> slicedList = rectangle.GetRange(v_start, v_end - v_start);
            double maxVal = 0;
            double val = 0;
            int r_H = v_end - v_start;
            int r_W = h_end - h_start;

            for (int c = 0; c < r_W && h_start < h_end; c++, h_start++)
            {
                double[] arr = slicedList[c];
                if (maxVal < arr.Skip(h_start - 1).Take(r_W).ToArray().Max())
                    maxVal = val;
            }
            return maxVal;
        }
        public void resize(Tuple<int, int> pad)
        {
            int old_H = this.Height;
            int old_W = this.Width;
            this.Height = Height + 2 * pad.Item1;
            this.Width = Width + 2 * pad.Item2;
            List<double[]> newR = new List<double[]>();

            for (int j = 0; j < Height; j++)
            {
                double[] arr = new double[Width];
                for (int k = 0; k < Width; k++)
                {
                    if (j < pad.Item1 || k < pad.Item2 || j > old_H + pad.Item1 - 1 || k > old_W + pad.Item2 - 1)
                        arr[k] = 0;
                    else
                        arr[k] = this.getValue(j - pad.Item1, k - pad.Item2);
                }
                newR.Add(arr);
            }
            rectangle = newR;
        }
    }
}
