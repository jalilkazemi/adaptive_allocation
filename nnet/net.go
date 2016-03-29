package nnet

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

func NewNNet(inputSize, layerSize int) *NNet {
	rnd := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

	coef0 := make([][]float64, layerSize)
	for i := 0; i < layerSize; i++ {
		coef0[i] = make([]float64, inputSize+1)
		for j := 0; j < inputSize+1; j++ {
			coef0[i][j] = rndNum(rnd) //todo
		}
	}
	coef1 := make([]float64, layerSize+1)
	for i := 0; i < layerSize+1; i++ {
		coef1[i] = rndNum(rnd) //todo
	}

	return &NNet{
		InputSize: inputSize,
		LayerSize: layerSize,
		Coef0:     coef0,
		Coef1:     coef1,
	}
}
func rndNum(rnd *rand.Rand) float64 {
	return 2*rnd.Float64() - 1.0
}
func LoadNNet(filename string) *NNet {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal(err.Error())
	}
	n := new(NNet)
	err = json.Unmarshal(b, n)
	if err != nil {
		log.Fatal(err.Error())
	}
	return n
}

type NNet struct {
	InputSize, LayerSize int
	Coef0                [][]float64
	Coef1                []float64
}

func (n *NNet) Backpropagate(x []float64, y float64, isDirection bool, learningRate float64) (yerr float64) {
	yhat, hiddenLayer := n.frontpropagate(x)
	if isDirection {
		yerr = y
	} else {
		yerr = y - yhat
	}

	outputDeltaW := make([]float64, n.LayerSize)
	for i := 0; i < n.LayerSize; i++ {
		outputDeltaW[i] = yerr * n.Coef1[i+1] * hiddenLayer[i] * (1 - hiddenLayer[i])
	}

	length := 0.0
	for i := 0; i < n.LayerSize; i++ {
		length += hiddenLayer[i] * hiddenLayer[i]
	}
	n.Coef1[0] += learningRate * yerr / length
	for i := 0; i < n.LayerSize; i++ {
		n.Coef1[i+1] += learningRate * yerr * hiddenLayer[i] / length
	}

	length = 0.0
	for i := 0; i < n.InputSize; i++ {
		length += x[i] * x[i]
	}
	for i := 0; i < n.LayerSize; i++ {
		n.Coef0[i][0] += learningRate * outputDeltaW[i] / length
		for j := 0; j < n.InputSize; j++ {
			n.Coef0[i][j+1] += learningRate * outputDeltaW[i] * x[j] / length
		}
	}
	return
}

func (n *NNet) Frontpropagate(x []float64) (y float64) {
	y, _ = n.frontpropagate(x)
	return
}

func (n *NNet) frontpropagate(x []float64) (y float64, hiddenLayer []float64) {
	hiddenLayer = make([]float64, n.LayerSize)
	for i := 0; i < n.LayerSize; i++ {
		hiddenLayer[i] = n.Coef0[i][0]
		for j := 0; j < n.InputSize; j++ {
			hiddenLayer[i] += n.Coef0[i][j+1] * x[j]
		}
		hiddenLayer[i] = sigmoid(hiddenLayer[i])
	}
	y = n.Coef1[0]
	for i := 0; i < n.LayerSize; i++ {
		y += n.Coef1[i+1] * hiddenLayer[i]
	}
	return
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (n *NNet) Save(filename string) {
	b, err := json.Marshal(n)
	if err != nil {
		log.Fatal(err.Error())
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal(err.Error())
	}
}
