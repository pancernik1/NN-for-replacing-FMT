using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;

namespace NN.GPU
{
    public class MainHandler
    {
        private readonly WeightsHandler wHandler = new WeightsHandler();

        public int[] CalcNetwork(int[] input, string fileName, int[] layerSizes)
        {
            using var midLayer = new GPUMidLayer(layerSizes);

            var (weights, biases) = wHandler.LoadWeights(midLayer.Accelerator, fileName);

            int layerCount = layerSizes.Length - 1;
            for (int i = 0; i < layerCount; i++)
            {
                midLayer.SetWeights(i, weights[i], biases[i]);
            }

            return midLayer.Forward(input);
        }

        public void MakeRandom(int[] layerSizes)
        {
            using var midLayer = new GPUMidLayer(layerSizes);
            wHandler.SaveWeights(midLayer.Weights, midLayer.Biases, "rand");
        }

        public int[][] TrainResults(int[] input, int[] layerSizes, int processCount, string fileName, float mutationRange)
        {
            var results = new int[processCount][];

            Parallel.For(0, processCount, procIndex =>
            {
                using var midLayer = new GPUMidLayer(layerSizes);

                var (baseWeights, baseBiases) = wHandler.LoadWeights(midLayer.Accelerator, fileName);

                int layerCount = layerSizes.Length - 1;
                for (int layer = 0; layer < layerCount; layer++)
                {
                    float[] hostWeights = baseWeights[layer].GetAsArray1D();
                    float[] hostBiases = baseBiases[layer].GetAsArray1D();

                    Mutate(hostWeights, mutationRange);
                    Mutate(hostBiases, mutationRange);

                    baseWeights[layer].Dispose();
                    baseBiases[layer].Dispose();

                    var mutatedWeightsBuf = midLayer.Accelerator.Allocate1D(hostWeights);
                    var mutatedBiasesBuf = midLayer.Accelerator.Allocate1D(hostBiases);

                    midLayer.SetWeights(layer, mutatedWeightsBuf, mutatedBiasesBuf);
                }

                results[procIndex] = midLayer.Forward(input);
            });

            return results;
        }

        private static void Mutate(float[] values, float range)
        {
            for (int i = 0; i < values.Length; i++)
            {
                float delta = (float)(Random.Shared.NextDouble() * 2.0 - 1.0) * range;
                values[i] += delta;
            }
        }
    }

    public class WeightsHandler
    {
        public void SaveWeights(MemoryBuffer1D<float, Stride1D.Dense>[] weights, MemoryBuffer1D<float, Stride1D.Dense>[] biases, string name)
        {
            using (var fs = new FileStream(name + "_weights.bin", FileMode.Create, FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(weights.Length);

                foreach (var buffer in weights)
                {
                    float[] hostData = buffer.GetAsArray1D();
                    bw.Write(hostData.Length);

                    byte[] byteData = new byte[hostData.Length * sizeof(float)];
                    Buffer.BlockCopy(hostData, 0, byteData, 0, byteData.Length);
                    bw.Write(byteData);
                }
            }

            using (var fs = new FileStream(name + "_biases.bin", FileMode.Create, FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(biases.Length);

                foreach (var buffer in biases)
                {
                    float[] hostData = buffer.GetAsArray1D();
                    bw.Write(hostData.Length);

                    byte[] byteData = new byte[hostData.Length * sizeof(float)];
                    Buffer.BlockCopy(hostData, 0, byteData, 0, byteData.Length);
                    bw.Write(byteData);
                }
            }
        }

        public (MemoryBuffer1D<float, Stride1D.Dense>[] weights, MemoryBuffer1D<float, Stride1D.Dense>[] biases) LoadWeights(Accelerator accelerator, string name)
        {
            MemoryBuffer1D<float, Stride1D.Dense>[] resultWeights;
            MemoryBuffer1D<float, Stride1D.Dense>[] resultBiases;

            using (var fs = new FileStream(name + "_weights.bin", FileMode.Open, FileAccess.Read))
            using (var br = new BinaryReader(fs))
            {
                int count = br.ReadInt32();
                resultWeights = new MemoryBuffer1D<float, Stride1D.Dense>[count];

                for (int i = 0; i < count; i++)
                {
                    int len = br.ReadInt32();
                    byte[] byteData = br.ReadBytes(len * sizeof(float));

                    float[] data = new float[len];
                    Buffer.BlockCopy(byteData, 0, data, 0, byteData.Length);

                    var buffer = accelerator.Allocate1D<float>(len);
                    buffer.CopyFromCPU(data);
                    resultWeights[i] = buffer;
                }
            }

            using (var fs = new FileStream(name + "_biases.bin", FileMode.Open, FileAccess.Read))
            using (var br = new BinaryReader(fs))
            {
                int count = br.ReadInt32();
                resultBiases = new MemoryBuffer1D<float, Stride1D.Dense>[count];

                for (int i = 0; i < count; i++)
                {
                    int len = br.ReadInt32();
                    byte[] byteData = br.ReadBytes(len * sizeof(float));

                    float[] data = new float[len];
                    Buffer.BlockCopy(byteData, 0, data, 0, byteData.Length);

                    var buffer = accelerator.Allocate1D<float>(len);
                    buffer.CopyFromCPU(data);
                    resultBiases[i] = buffer;
                }
            }

            return (resultWeights, resultBiases);
        }
    }

    public class GPUMidLayer : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly int[] _layerSizes;
        private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _weights;
        private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _biases;
        private readonly Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int,
            byte> _layerKernel;
        private readonly Random _rng;

        public Accelerator Accelerator => _accelerator;
        public MemoryBuffer1D<float, Stride1D.Dense>[] Weights => _weights;
        public MemoryBuffer1D<float, Stride1D.Dense>[] Biases => _biases;

        public GPUMidLayer(int[] layerSizes, int? seed = null, bool preferCPU = false)
        {
            if (layerSizes == null || layerSizes.Length < 2)
            {
                throw new ArgumentException("Minimum 2 layers needed");
            }
            foreach (var size in layerSizes)
            {
                if (size <= 0)
                {
                    throw new ArgumentException("Layer must have at least one node");
                }
            }

            _layerSizes = layerSizes;
            _rng = seed.HasValue ? new Random(seed.Value) : new Random();

            _context = Context.CreateDefault();
            _accelerator = _context.GetPreferredDevice(preferCPU).CreateAccelerator(_context);

            int layerCount = layerSizes.Length - 1;
            _weights = new MemoryBuffer1D<float, Stride1D.Dense>[layerCount];
            _biases = new MemoryBuffer1D<float, Stride1D.Dense>[layerCount];

            for (int i = 0; i < layerCount; i++)
            {
                int inSize = layerSizes[i];
                int outSize = layerSizes[i + 1];

                _weights[i] = _accelerator.Allocate1D(InitWeights(inSize, outSize));
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

        private float[] InitWeights(int inSize, int outSize)
        {
            float scale = MathF.Sqrt(2f / inSize);
            var w = new float[inSize * outSize];
            for (int i = 0; i < w.Length; i++)
            {
                w[i] = (float)(_rng.NextDouble() * 2.0 - 1.0) * scale;
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
            byte applyRelu)
        {
            float sum = 0f;
            int baseIdx = j * inSize;

            for (int i = 0; i < inSize; i++)
            {
                sum += input[i] * weights[baseIdx + i];
            }

            sum += biases[j];

            output[j] = (applyRelu != 0 && sum < 0f) ? 0f : sum;
        }

        public int[] Forward(int[] input)
        {
            if (input.Length != _layerSizes[0])
            {
                throw new ArgumentException("input layer length's do not match");
            }

            var floatInput = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                floatInput[i] = input[i];
            }

            using var initialBuffer = _accelerator.Allocate1D(floatInput);
            MemoryBuffer1D<float, Stride1D.Dense> prevBuffer = initialBuffer;
            bool ownsPrev = false;

            for (int layer = 0; layer < _weights.Length; layer++)
            {
                int inSize = _layerSizes[layer];
                int outSize = _layerSizes[layer + 1];
                bool isLastLayer = layer == _weights.Length - 1;

                var outBuffer = _accelerator.Allocate1D<float>(outSize);

                _layerKernel(
                    outSize,
                    prevBuffer.View,
                    _weights[layer].View,
                    _biases[layer].View,
                    outBuffer.View,
                    inSize,
                    (byte)(isLastLayer ? 0 : 1));

                _accelerator.Synchronize();

                if (ownsPrev)
                {
                    prevBuffer.Dispose();
                }
                prevBuffer = outBuffer;
                ownsPrev = true;
            }

            var floatResult = prevBuffer.GetAsArray1D();
            if (ownsPrev)
            {
                prevBuffer.Dispose();
            }

            var result = new int[floatResult.Length];
            for (int i = 0; i < floatResult.Length; i++)
            {
                result[i] = (int)MathF.Round(floatResult[i]);
            }
            return result;
        }

        public void SetWeights(int layerIndex, MemoryBuffer1D<float, Stride1D.Dense> weights, MemoryBuffer1D<float, Stride1D.Dense> biases)
        {
            _weights[layerIndex]?.Dispose();
            _biases[layerIndex]?.Dispose();

            _weights[layerIndex] = weights;
            _biases[layerIndex] = biases;
        }

        public void Dispose()
        {
            foreach (var w in _weights) w.Dispose();
            foreach (var b in _biases) b.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
        }
    }

    internal static class Program
    {
        private static void Main()
        {
            Console.WriteLine("Type in layer sizes with commas between them (ex. 4,8,8,3):");
            int[] layerSizes = ParseIntArray(Console.ReadLine());

            Console.WriteLine($"Input ({layerSizes[0]} with commas between them:");
            int[] input = ParseIntArray(Console.ReadLine());

            Console.WriteLine("Input filename for weights and biases:");
            string fileName = (Console.ReadLine() ?? string.Empty).Trim();

            Console.WriteLine("Input amount of parallel networks for training");
            int processCount = ParseIntOrDefault(Console.ReadLine(), 1);

            Console.WriteLine("Input mutation range (reccomended: 0.1):");
            float mutationRange = ParseFloatOrDefault(Console.ReadLine(), 0.1f);

            var handler = new MainHandler();

            if (!File.Exists(fileName + "_weights.bin"))
            {
                Console.WriteLine($"File '{fileName}_weights.bin' doesn't exist, generating random weights under that name...");
                var tempHandler = new WeightsHandler();
                using var tempLayer = new GPUMidLayer(layerSizes);
                tempHandler.SaveWeights(tempLayer.Weights, tempLayer.Biases, fileName);
            }

            int[][] results = handler.TrainResults(input, layerSizes, processCount, fileName, mutationRange);

            for (int i = 0; i < results.Length; i++)
            {
                Console.WriteLine($"Proces {i}: [{string.Join(", ", results[i])}]");
            }
        }

        private static int[] ParseIntArray(string? line)
        {
            if (string.IsNullOrWhiteSpace(line))
                throw new ArgumentException("Input cannot be empty");

            return line.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                        .Select(int.Parse)
                        .ToArray();
        }

        private static int ParseIntOrDefault(string? line, int fallback)
        {
            return int.TryParse(line, out int value) ? value : fallback;
        }

        private static float ParseFloatOrDefault(string? line, float fallback)
        {
            return float.TryParse(line, NumberStyles.Float, CultureInfo.InvariantCulture, out float value)
                ? value
                : fallback;
        }
    }
}