package nn

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"math/rand"
	"os"
	"time"
)

// NetworkOptions is for exporting network information to JSON
type NetworkOptions struct {
	I, O   int
	H      []int
	Learn  float64
	WPaths []string
	BPaths []string
}

// layer is a layer of the network
type layer struct {
	weights mat.Matrix
	biases  mat.Matrix
}

// newLayer Creates a new layer
func newLayer(layerSize, inputSize int, random bool) layer {
	if random {
		return layer{
			weights: mat.NewDense(layerSize, inputSize, randomArray(layerSize*inputSize, -1, 1)),
			biases:  mat.NewDense(layerSize, 1, randomArray(layerSize, -1, 1)),
		}
	}

	return layer{
		weights: mat.NewDense(layerSize, inputSize, nil),
		biases:  mat.NewDense(layerSize, 1, nil),
	}
}

// Network contains the whole neural network
type Network struct {
	i, o, h   int
	hidden    []int
	layers    []layer
	learnRate float64
}

// NewNetwork Creates a new Network
func NewNetwork(inputs, outputs int, hidden []int, learn float64, random bool) Network {
	layers := make([]layer, len(hidden)+1)

	for i := 0; i < len(hidden)+1; i++ {
		if i == 0 {
			layers[i] = newLayer(hidden[i], inputs, random)
			continue
		}

		if i == len(hidden) {
			layers[i] = newLayer(outputs, hidden[i-1], random)
			continue
		}

		layers[i] = newLayer(hidden[i], hidden[i-1], random)
	}

	return Network{
		i:         inputs,
		h:         len(layers),
		o:         outputs,
		hidden:    hidden,
		layers:    layers,
		learnRate: learn,
	}
}

// Calc evaluates a given input into the network
func (n Network) Calc(data []float64) []float64 {
	if len(data) != n.i {
		panic(errInvalidDataSize)
	}

	inputs := mat.NewDense(n.i, 1, data)

	var activation mat.Matrix

	for i := 0; i < n.h; i++ {
		if i == 0 {
			activation = fun(sigmoid, add(dot(n.layers[i].weights, inputs), n.layers[i].biases))
			continue
		}

		activation = fun(sigmoid, add(dot(n.layers[i].weights, activation), n.layers[i].biases))
	}

	r, _ := activation.Dims()
	res := make([]float64, r)

	for i := 0; i < r; i++ {
		res[i] = activation.At(i, 0)
	}

	return res
}

// backpropagate performs a small change on the network based on given data
func (n *Network) backpropagate(inputData []float64, expectedData []float64) {
	if len(inputData) != n.i || len(expectedData) != n.o {
		panic(errInvalidDataSize)
	}

	input := mat.NewDense(n.i, 1, inputData)
	expected := mat.NewDense(n.o, 1, expectedData)

	var (
		activations = make([]mat.Matrix, n.h)
		zs          = make([]mat.Matrix, n.h)
	)

	for i := 0; i < n.h; i++ {
		if i == 0 {
			zs[i] = add(dot(n.layers[i].weights, input), n.layers[i].biases)
			activations[i] = fun(sigmoid, zs[i])
			continue
		}

		zs[i] = add(dot(n.layers[i].weights, activations[i-1]), n.layers[i].biases)
		activations[i] = fun(sigmoid, zs[i])
	}

	layerErrors := sub(expected, activations[n.h-1])

	for i := n.h - 1; i >= 0; i-- {
		if i != n.h-1 {
			layerErrors = dot(n.layers[i+1].weights.T(), layerErrors)
		}

		n.layers[i].biases = add(n.layers[i].biases,
			scl(2*n.learnRate,
				mul(
					layerErrors,
					fun(dSigmoid, zs[i]))))

		if i == 0 {
			n.layers[i].weights = add(n.layers[i].weights,
				scl(n.learnRate,
					dot(mul(
						layerErrors,
						fun(dSigmoid, zs[i])),
						input.T())))
			continue
		}

		n.layers[i].weights = add(n.layers[i].weights,
			scl(n.learnRate,
				dot(mul(
					layerErrors,
					fun(dSigmoid, zs[i])),
					activations[i-1].T())))
	}
}

// Train repeatedly performs backpropagation. Will print information on the performance of the network
func (n *Network) Train(inputs, expected [][]float64, epochs int) {
	if len(inputs) != len(expected) {
		panic(errInvalidDataSize)
	}

	fmt.Printf("Began training for %d epochs...\n", epochs)

	start := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		counter := time.Now()
		avgCost := 0.0

		for i := 0; i < len(inputs); i++ {
			n.backpropagate(inputs[i], expected[i])
			avgCost += totalCost(expected[i], n.Calc(inputs[i]))
		}

		avgCost /= float64(len(inputs))

		fmt.Printf("  + Completed epoch %d of %d in %dms with an average cost of %.5f,\n",
			epoch+1, epochs, time.Since(counter).Milliseconds(), avgCost)
	}

	delta := time.Since(start).Milliseconds()

	fmt.Printf("Trained for %d epochs in %dms with an average of %dms per epoch.\n",
		epochs, delta, delta/int64(epochs))
}

func (n *Network) Perturb(strength float64) {
	rand.Seed(time.Now().Unix())

	for i := 0; i < n.h; i++ {
		wr, wc := n.layers[i].weights.Dims()
		br, bc := n.layers[i].biases.Dims()

		n.layers[i].weights = add(n.layers[i].weights, mat.NewDense(wr, wc, randomArray(wr*wc, -1*strength, 1*strength)))
		n.layers[i].biases = add(n.layers[i].biases, mat.NewDense(br, bc, randomArray(br*bc, -1*strength, 1*strength)))
	}
}

func (n *Network) Copy() (m Network) {
	m = Network{
		i:         n.i,
		o:         n.o,
		h:         n.h,
		hidden:    make([]int, len(n.hidden)),
		layers:    make([]layer, len(n.layers)),
		learnRate: n.learnRate,
	}

	copy(m.hidden, n.hidden)
	copy(m.layers, n.layers)

	return m
}

// Save will compress the network and then save it as a file to be used later.
func (n Network) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	zipper := zip.NewWriter(file)

	meta, err := zipper.Create("meta.json")

	opts := NetworkOptions{
		I:      n.i,
		O:      n.o,
		H:      n.hidden,
		Learn:  n.learnRate,
		WPaths: make([]string, n.h),
		BPaths: make([]string, n.h),
	}

	for i := 0; i < n.h; i++ {
		opts.WPaths[i] = fmt.Sprintf("%dw.bin", i)
		opts.BPaths[i] = fmt.Sprintf("%db.bin", i)
	}

	metaJson, err := json.Marshal(opts)

	_, err = meta.Write(metaJson)
	if err != nil {
		return err
	}

	for i := 0; i < n.h; i++ {
		w, wErr := zipper.Create(fmt.Sprintf("%dw.bin", i))
		if wErr != nil {
			return wErr
		}

		wb, wErr := n.layers[i].weights.(*mat.Dense).MarshalBinary()
		if wErr != nil {
			return wErr
		}

		_, wErr = w.Write(wb)
		if wErr != nil {
			return wErr
		}

		b, bErr := zipper.Create(fmt.Sprintf("%db.bin", i))
		if bErr != nil {
			return bErr
		}

		bb, bErr := n.layers[i].biases.(*mat.Dense).MarshalBinary()
		if bErr != nil {
			return bErr
		}

		_, bErr = b.Write(bb)
		if bErr != nil {
			return bErr
		}
	}

	_ = zipper.Close()
	_ = file.Close()

	return nil
}

// Load will open a saved network
func Load(filename string) (n Network, err error) {
	zipFile, err := zip.OpenReader(filename)
	if err != nil {
		return Network{}, err
	}

	metaFile, err := zipFile.Open("meta.json")
	if err != nil {
		return Network{}, err
	}

	meta, err := ioutil.ReadAll(metaFile)
	if err != nil {
		return Network{}, err
	}

	var opts NetworkOptions

	err = json.Unmarshal(meta, &opts)
	if err != nil {
		return Network{}, err
	}

	n = NewNetwork(opts.I, opts.O, opts.H, opts.Learn, false)

	_ = metaFile.Close()

	for i := 0; i < n.h; i++ {
		w, wErr := zipFile.Open(fmt.Sprintf("%s", opts.WPaths[i]))
		if wErr != nil {
			return Network{}, wErr
		}

		n.layers[i].weights.(*mat.Dense).Reset()
		_, wErr = n.layers[i].weights.(*mat.Dense).UnmarshalBinaryFrom(w)
		if wErr != nil {
			return Network{}, wErr
		}

		_ = w.Close()

		b, bErr := zipFile.Open(fmt.Sprintf("%s", opts.BPaths[i]))
		if bErr != nil {
			return Network{}, bErr
		}

		n.layers[i].biases.(*mat.Dense).Reset()
		_, bErr = n.layers[i].biases.(*mat.Dense).UnmarshalBinaryFrom(b)
		if bErr != nil {
			return Network{}, bErr
		}

		_ = b.Close()
	}

	_ = zipFile.Close()

	return n, nil
}
