package bandit

import (
	"math"
	"math/rand"
	"nnet"
	"time"
)

const (
	delta   = 0.9
	epsilon = 0.05
)

/*
	Discretely identified value function
*/
func GreedyValueOf(t ABState) float64 {
	return max(t.P[0], t.P[1]) / (1.0 - delta)
}

func LastStepValueOf(t ABState) float64 {
	return max(t.P[0], t.P[1])
}
func IteratedValueOf(v func(ABState) float64) func(ABState) float64 {
	return func(t ABState) float64 {
		gA, rA, pA := t.ActA()
		gB, rB, pB := t.ActB()
		iterativeA := gA + delta*(pA[0]*v(rA[0])+pA[1]*v(rA[1]))
		iterativeB := gB + delta*(pB[0]*v(rB[0])+pB[1]*v(rB[1]))
		return max(iterativeA, iterativeB)
	}
}

/*
	Generalization of the value function
*/
func NewNNetValue(layerSize int) NNetValue {
	return NNetValue{nnet.NewNNet(4, layerSize)}
}
func LoadNNetValue(filename string) NNetValue {
	return NNetValue{nnet.LoadNNet(filename)}
}

type NNetValue struct {
	n *nnet.NNet
}

func (v *NNetValue) ValueOf(t ABState) float64 {
	x := []float64{
		1.0 / t.N[0],
		1.0 / t.N[1],
		t.P[0],
		t.P[1],
	}
	return v.n.Frontpropagate(x)
}
func (v *NNetValue) BackpropagateDeeply(t ABState, learningRate float64) (yerr float64) {
	rnd := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

	var steps int
	if t.N[0] < t.N[1] {
		steps = int(math.Floor(t.N[0]))
	} else {
		steps = int(math.Floor(t.N[1]))
	}
	if steps > 100 {
		steps = 100
	}

	tt := t
	y := 0.0
	for i := 0; i < steps; i++ {
		gA, rA, pA := tt.ActA()
		gB, rB, pB := tt.ActB()
		iterativeA := gA + delta*(pA[0]*v.ValueOf(rA[0])+pA[1]*v.ValueOf(rA[1]))
		iterativeB := gB + delta*(pB[0]*v.ValueOf(rB[0])+pB[1]*v.ValueOf(rB[1]))

		var chooseA bool
		random := rnd.Float64()
		if random < epsilon {
			chooseA = true
		} else if random < 2*epsilon {
			chooseA = false
		} else if iterativeA > iterativeB {
			chooseA = true
		} else {
			chooseA = false
		}
		if chooseA {
			y += math.Pow(delta, float64(i)) * gA
			if rnd.Float64() < pA[0] {
				tt = rA[0]
			} else {
				tt = rA[1]
			}
		} else {
			y += math.Pow(delta, float64(i)) * gB
			if rnd.Float64() < pB[0] {
				tt = rB[0]
			} else {
				tt = rB[1]
			}
		}
	}
	y += math.Pow(delta, float64(steps)) * v.ValueOf(tt)
	return v.BackpropagateUsing(t, y, learningRate)
}
func (v *NNetValue) Backpropagate(t ABState, learningRate float64) (yerr float64) {
	gA, rA, pA := t.ActA()
	gB, rB, pB := t.ActB()
	iterativeA := gA + delta*(pA[0]*v.ValueOf(rA[0])+pA[1]*v.ValueOf(rA[1]))
	iterativeB := gB + delta*(pB[0]*v.ValueOf(rB[0])+pB[1]*v.ValueOf(rB[1]))
	y := max(iterativeA, iterativeB)

	return v.BackpropagateUsing(t, y, learningRate)
}
func (v *NNetValue) BackpropagateUsing(t ABState, y float64, learningRate float64) (yerr float64) {
	x := []float64{
		1.0 / t.N[0],
		1.0 / t.N[1],
		t.P[0],
		t.P[1],
	}

	return v.n.Backpropagate(x, y, learningRate)
}

func (v *NNetValue) Save(filename string) {
	v.n.Save(filename)
}
