package abbandit

import (
	"math/rand"
)

const (
	minStep = 100
	maxStep = 10000
)

type ActionFactory struct {
	rnd *rand.Rand
}

func NewActionFactory(rnd *rand.Rand) ActionFactory {
	return ActionFactory{rnd}
}
func (f *ActionFactory) Next() Action {
	q := Action{}
	q.Move = f.rnd.Float64()
	n := (1 + f.rnd.Intn(maxStep/minStep)) * minStep
	n0 := 1 + f.rnd.Intn(n-1)
	n1 := n - n0
	q.State.N[0] = float64(n0)
	q.State.N[1] = float64(n1)
	q.State.K[0] = float64(f.rnd.Intn(n0))
	q.State.K[1] = float64(f.rnd.Intn(n1))
	return q
}

type State struct {
	N [2]float64
	K [2]float64
}

func (s *State) IsTerminalState() bool {
	return s.N[0]+s.N[1] >= maxStep
}
func (s *State) FaceValue() float64 {
	return (s.K[0] + s.K[1]) / (s.N[0] + s.N[1])
}
func (s *State) Prob(segment int) float64 {
	return s.K[segment] / s.N[segment]
}
func (s *State) AddPositive(segment int) {
	s.N[segment]++
	s.K[segment]++
}
func (s *State) AddNegative(segment int) {
	s.N[segment]++
}
func (s *State) BestMove(valueOf func(Action) float64) (bestMove, value float64) {
	values := make([]float64, 101)
	for i := 0; i <= 100; i++ {
		move := float64(i) / 100.0
		values[i] = valueOf(Action{Move: move, State: *s})
	}
	var bestMoveInd int
	bestMoveInd, value = maxOf(values...)
	bestMove = float64(bestMoveInd) / 100.0
	return
}

type Action struct {
	State
	Move float64
}

func (q *Action) Travel(rnd *rand.Rand) (destination State) {
	destination = q.State
	for i := 0; i < minStep && q.N[0]+q.N[1]+float64(i) < float64(maxStep); i++ {
		random := rnd.Float64()
		if random < q.Move {
			if random < q.Move*q.Prob(0) {
				destination.AddPositive(0)
			} else {
				destination.AddNegative(0)
			}
		} else {
			if random-q.Move < (1-q.Move)*q.Prob(1) {
				destination.AddPositive(1)
			} else {
				destination.AddNegative(1)
			}
		}
	}
	return
}
