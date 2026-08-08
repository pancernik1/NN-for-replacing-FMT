using System;
using System.CodeDom.Compiler;
using System.IO;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace NN.GPU
{
    public class mainHandler: IDisposable
    {
        private readonly GPUMidLayer midLayer;
      
        private weightsHandler wHandler = new weightsHandler();
        
        public int[] calcNetwork(int[] input,string fileName,int[] layerSizes)
        {
            var midLayer = new GpuMiddleLayer(layerSizes,null,false);
            for(int i =0;i<LayerSizes.Length;i++)
            {
                midLayer.SetWeights(i,wHandler.LoadWeights(midLayer._accelerator,fileName).weights[i],wHandler.LoadWeights(midLayer._accelerator,fileName).biases[i]);
            }
            return midLayer.Forward(input);
        }
        public void makeRandom(int[] input,int[] layerSizes)
        {
            var midLayer = new GpuMiddleLayer(layerSizes,null,false);
            midLayer.Forward(input);
        }
    }
    public class weightsHandler()
    {
        public void SaveWeights(MemoryBuffer1D<float, Stride1D.Dense>[] weights,MemoryBuffer1D<float, Stride1D.Dense>[] biases, string name)
        {
            using (var fs = new FileStream(name + "_weights.bin",FileMode.Create,FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(weights.Length);

                foreach(var buffer in weights)
                {
                    float[] hostData = buffer.GetAsArray1D();

                    bw.Write(hostData.Length);

                   byte[] byteData = new byte[hostData.Length * sizeof(float)];
                   Buffer.BlockCopy(hostData,0,byteData,0,byteData.Length);
                   bw.Write(byteData);
                }
            }
            using (var fs = new FileStream(name + "_biases.bin",FileMode.Create,FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(biases.Length);

                foreach(var buffer in biases)
                {
                    float[] hostData = buffer.GetAsArray1D();

                    bw.Write(hostData.Length);

                   byte[] byteData = new byte[hostData.Length * sizeof(float)];
                   Buffer.BlockCopy(hostData,0,byteData,0,byteData.Length);
                   bw.Write(byteData);
                }
            }
        }
        public (MemoryBuffer1D<float, Stride1D.Dense>[] weights,MemoryBuffer1D<float, Stride1D.Dense>[] biases) LoadWeights(Accelerator accelerator,string name)
        {
            var result_weights = new MemoryBuffer1D<float, Stride1D.Dense>[count];
            var result_biases = new MemoryBuffer1D<float, Stride1D.Dense>[count];
            using (var fs = new FileStream(name+"_weights.bin", FileMode.Open, FileAccess.Read))
            using (var br = new BinaryReader(fs))
            {
                int count = br.ReadInt32();
              
                for(int i = 0; i < count; i++)
                {
                    int len = br.ReadInt32();
                    byte[] byteData = br.ReadBytes(len * sizeof(float));

                    float[] data = new float[len];
                    Buffer.BlockCopy(byteData,0,data,byteData.Length);

                    var buffer = accelerator.Allocate1D<float>(len);
                    buffer.copyFromCPU(data);
                    result_weights[i] = buffer;
                }
                
            }
            using (var fs = new FileStream(name+"_biases.bin", FileMode.Open, FileAccess.Read))
            using (var br = new BinaryReader(fs))
            {
                int count = br.ReadInt32();
              
                for(int i = 0; i < count; i++)
                {
                    int len = br.ReadInt32();
                    byte[] byteData = br.ReadBytes(len * sizeof(float));

                    float[] data = new float[len];
                    Buffer.BlockCopy(byteData,0,data,byteData.Length);

                    var buffer = accelerator.Allocate1D<float>(len);
                    buffer.copyFromCPU(data);
                    result_biases[i] = buffer;
                }
                
            }
            return (result_weights,result_biases);
        }
    }
    public class GPUMidLayer : IDisposable
    {
        private weightsHandler wHandler;
        private readonly Context _context;
        public Accelerator _accelerator;
        private readonly int[] _layerSizes;
        private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _weights;
        private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _biases;
        private readonly Action<
        Index1D
        ,ArrayView1D<float, Stride1D.Dense>
        ,ArrayView1D<float, Stride1D.Dense>
        ,ArrayView1D<float, Stride1D.Dense>
        ,ArrayView1D<float, Stride1D.Dense>
        ,int
        ,byte> _layerKernel;
        private readonly Random _rng;

        public GPUMidLayer(int[] layerSizes, int? seed = null, bool preferCPU = false)
        {
            if(layerSizes == null || layerSizes.Length < 2)
            {
                throw new ArgumentException("Minimum 2 layers needed");
            }    
            foreach(var size in layerSizes)
            {
                if(size <= 0)
                {
                    throw new ArgumentException("Layer must have at least one node");
                }
            }

            _layerSizes = layerSizes;
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();

            _context = Context.CreateDefault();
            _accelerator = _context.GetPrefferedDevice(preferCPU).CreateAccelerator(_context);
        
            int layerCount = layerSizes.Length - 1;
            _weights = new MemoryBuffer1D<float, Stride1D.Dense>[layerCount];
            _biases = new MemoryBuffer1D<float, Stride1D.Dense>[layerCount];

            for(int i = 0; i< layerCount;i++)
            {
                int inSize = layerSizes[i];
                int outSize = layerSizes[i];

                _weights[i] = _accelerator.Allocate1D(InitWeights(inSize,outSize));
                _biases[i] = _accelerator.Allocate1D(new float[outSize]);
            }
            _layerKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                int,
                byte>(DenseLayerKernel);
        }

        private float[] InitWeights(int inSize,int outSize)
        {
            float Scale = MathF.Sqrt(2f / inSize);
            var w = new float[inSize*outSize];
            for(int i = 0; i<w.Length;i++)
            {
                    w[i] = (float)(_rng.NextDouble() * 2.0 - 1.0) * Scale;
                
            }
            return w;
        }
        private static void DenseLayerKernel(
            Index1D j,
            ArrayView1D<float, Stride1D.Dense> input,
            ArrayView1D<float, Stride1D.Dense> weights,
            ArrayView1D<float, Stride1D.Dense> biases,
            ArrayView1D<float, Stride1D.Dense> output,
            int inSize,
            byte applyRelu
        )
        {
            float sum = 0f;
            int baseIdx = j * inSize;

            for(int i = 0; i<inSize;i++)
            {
                sum += input[i] * weights[baseIdx + i];
            }

            sum += biases;
            output[j] = (applyRelu != 0 && sum < 0f) ? 0f : sum;
        }
        public int[] Forward(int[] input)
        {
            if(input.Length != _layerSizes[0])
            {
                throw new ArgumentException("input layer length's do not match");
            }
            var floatInput = new float[input.Length];
            for(int i =0; i<input.Length; i++)
            {
                floatInput[i] = input[i];
            }

            using var initialBuffer = _accelerator.Allocate1D(floatInput);
            MemoryBuffer1D<float, Stride1D.Dense> prevBuffer = initialBuffer;
            bool ownsPrev = false;

            for(int layer = 0; layer < _weights.Length; layer++)
            {
                int inSize = _layerSizes[layer];
                int outSize = _layerSizes[layer + 1];
                bool isLastLayer = layer == _weights.Length - 1;

                var outBuffer = _accelerator.Allocate1D<float>(outSize);

                _layerKernel(
                    outSize,
                    prevBuffer.View,
                    _weights[layer].View,
                    _biases.View,
                    outBuffer.View,
                    inSize,
                    (byte)(isLastLayer ? 0 : 1)

                );
                
                _accelerator.Synchronize();

                if(ownsPrev)
                {
                    prevBuffer.Dispose();
                }
                prevBuffer = outBuffer;
                ownsPrev =true;
            }
            var floatResult = prevBuffer.GetAsArray1D();
            if(ownsPrev)
            {
                prevBuffer.Dispose();
            }

            var result = new int[floatResult.Length];
            for(int i =0;i<floatResult.Length;i++)
            {
                result[i] = (int)MathF.Round(floatResult[i]);
            }
            return result;
        }
        public void SetWeights(int layerIndex,MemoryBuffer1D<float, Stride1D.Dense> weights, MemoryBuffer1D<float, Stride1D.Dense> biases)
        {
            _weights[layerIndex] = weights;
            _biases[layerIndex] = biases;
        }
        public void Dispose()
        {
            foreach(var w in _weights) w.Dispose();
            foreach(var b in _biases) b.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
        }
    }
}
