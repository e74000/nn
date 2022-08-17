package nn

import (
	"errors"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
)

var (
	errInvalidDataSize = errors.New("invalid data size")
)

// lerp is used to map random numbers across a range
func lerp(x, li, ui, lo, uo float64) float64 {
	return ((x-li)/(ui-li))*(uo-lo) + lo
}

// sigmoid is the network's activation function
func sigmoid(_, _ int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

// dSigmoid is the derivative of the network's activation function
func dSigmoid(_, _ int, v float64) float64 {
	return sigmoid(0, 0, v) * (1 - sigmoid(0, 0, v))
}

// Produces a random array for initialising the weights and biases
func randomArray(size int, u, l float64) []float64 {
	rand.Seed(time.Now().UnixNano())

	res := make([]float64, size)

	for i := 0; i < size; i++ {
		res[i] = lerp(rand.Float64(), 0, 1, u, l)
	}

	return res
}

// dot is a wrapper for Matrix.Dot()
func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	res := mat.NewDense(r, c, nil)
	res.Product(m, n)
	return res
}

// mul is a wrapper for Matrix.MulElem()
func mul(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	res := mat.NewDense(r, c, nil)
	res.MulElem(m, n)
	return res
}

// fun is a wrapper for Matrix.Apply()
func fun(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Apply(fn, m)
	return res
}

// scl is a wrapper for Matrix.Scale()
func scl(f float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Scale(f, m)
	return res
}

// add is a wrapper for Matrix.add()
func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Add(m, n)
	return res
}

// sub is a wrapper for Matrix.sub()
func sub(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Sub(m, n)
	return res
}

// totalCost calculates the sum of all the costs
func totalCost(got, expected []float64) float64 {
	if len(got) != len(expected) {
		panic(errInvalidDataSize)
	}

	total := 0.0

	for i := 0; i < len(got); i++ {
		total += math.Pow(got[i]-expected[i], 2)
	}

	return total
}
