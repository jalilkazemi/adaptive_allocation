package abbandit

import (
	"math"
	"math/rand"
	"nnet"
)

const (
	bootSize = 10
)

func NewActionValue(layerSize int) ActionValue {
	numFeature := len(Featurize(Action{State: State{N: [2]float64{1, 1}}, Move: 0.0}))
	return ActionValue{nnet.NewNNet(numFeature, layerSize)}
}
func LoadActionValue(filename string) ActionValue {
	return ActionValue{nnet.LoadNNet(filename)}
}

type ActionValue struct {
	n *nnet.NNet
}

func (v *ActionValue) ValueOf(q Action) float64 {
	x := Featurize(q)
	return v.n.Frontpropagate(x)
}
func (v *ActionValue) Error(rnd *rand.Rand, q Action) float64 {
	value := v.ValueOf(q)
	var expectedValue float64
	for i := 0; i < bootSize; i++ {
		instanceState := q.Travel(rnd)
		_, instanceValue := instanceState.BestMove(v.ValueOf)
		expectedValue += (instanceValue - expectedValue) / float64(i+1)
	}
	return expectedValue - value
}
func (v *ActionValue) Update(q Action, y float64, isDirection bool, learningRate float64) (yerr float64) {
	x := Featurize(q)
	return v.n.Backpropagate(x, y, isDirection, learningRate)
}

func Featurize(q Action) []float64 {
	if q.State.N[0] == 0 || q.State.N[1] == 0 {
		panic("N is zero")
	}
	p := []float64{
		q.State.K[0] / q.State.N[0],
		q.State.K[1] / q.State.N[1],
	}
	var aWins float64
	if p[0] > p[1] {
		aWins = 1.0
	}
	return []float64{
		q.Move,
		q.State.N[0] / float64(maxStep),
		q.State.N[1] / float64(maxStep),
		p[0],
		p[1],
		(q.State.K[0] + q.State.K[1]) / (q.State.N[0] + q.State.N[1]),
		max(p[0], p[1]),
		math.Pow(p[0], 2),
		math.Pow(p[1], 2),
		math.Pow(q.Move, 2),
		q.Move * aWins,
		q.Move * p[0],
		q.Move * p[1],
	}
}
func (v *ActionValue) Save(filename string) {
	v.n.Save(filename)
}
