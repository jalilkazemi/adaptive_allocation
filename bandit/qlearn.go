package bandit

import (
	"math"
	"math/rand"
	"nnet"
)

const (
	eps      = 0.00
	bootSize = 20
)

func NewNNetScore(layerSize int) NNetScore {
	numFeature := len(Featurize(ABTestQ{State: ABTestS{N: [2]float64{1, 1}}}))
	return NNetScore{nnet.NewNNet(numFeature, layerSize)}
}
func LoadNNetScore(filename string) NNetScore {
	return NNetScore{nnet.LoadNNet(filename)}
}

type NNetScore struct {
	n *nnet.NNet
}

func (v *NNetScore) ScoreOf(q ABTestQ) float64 {
	x := Featurize(q)
	return v.n.Frontpropagate(x)
}
func (v *NNetScore) Error(rnd *rand.Rand, q ABTestQ) float64 {
	score := v.ScoreOf(q)
	var expectedScore float64
	for i := 0; i < bootSize; i++ {
		s := q.Travel(rnd)
		_, instanceScore := s.BestAction(v.ScoreOf)
		expectedScore += (instanceScore - expectedScore) / float64(i+1)
	}
	return expectedScore - score
}
func (v *NNetScore) Backpropagate(
	rnd *rand.Rand,
	steps int,
	q *ABTestQ,
	update bool,
	learningRate float64) (yerr float64, terminal bool) {

	if steps < 1 {
		panic("steps must be a positive integer")
	}
	qTrace := make([]ABTestQ, 0, steps)
	var y float64
	for i := 0; i < steps; i++ {
		qTrace = append(qTrace, *q)
		qOld := *q
		s := qOld.Travel(rnd)
		if s.IsTerminalState() {
			terminal = update
			y = s.FaceScore()
			break
		} else {
			random := rnd.Float64()
			if random < eps {
				*q = ABTestQ{Action: random / eps, State: s}
			} else {
				bestAction, _ := s.BestAction(v.ScoreOf)
				*q = ABTestQ{Action: bestAction, State: s}
			}
			y = v.ScoreOf(*q)
		}
	}
	for i, qOld := range qTrace {
		yerr += (v.BackpropagateUsing(qOld, y, learningRate) - yerr) / float64(i+1)
	}
	if update {
		// if len(qTrace) > 1 {
		// 	*q = qTrace[1]
		// 	terminal = false
		// }
	} else {
		*q = qTrace[0]
	}
	return
}

func (v *NNetScore) BackpropagateDeeply(rnd *rand.Rand, q ABTestQ, learningRate float64) (yerr float64, steps int) {
	qTrace := []ABTestQ{q}
	s := q.Travel(rnd)
	steps = 1
	for !s.IsTerminalState() {
		steps++
		random := rnd.Float64()
		if random < eps {
			q = ABTestQ{Action: random / eps, State: s}
		} else {
			bestAction, _ := s.BestAction(v.ScoreOf)
			q = ABTestQ{Action: bestAction, State: s}
		}
		s = q.Travel(rnd)
	}
	y := s.FaceScore()
	for i, qOld := range qTrace {
		yerr += (v.BackpropagateUsing(qOld, y, learningRate) - yerr) / float64(i+1)
	}
	return
}

func (v *NNetScore) BackpropagateUsing(q ABTestQ, y float64, learningRate float64) (yerr float64) {
	x := Featurize(q)

	return v.n.Backpropagate(x, y, learningRate)
}

func Featurize(q ABTestQ) []float64 {
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
		q.Action,
		q.State.N[0] / float64(maxStep),
		q.State.N[1] / float64(maxStep),
		p[0],
		p[1],
		(q.State.K[0] + q.State.K[1]) / (q.State.N[0] + q.State.N[1]),
		max(p[0], p[1]),
		math.Pow(p[0], 2),
		math.Pow(p[1], 2),
		math.Pow(q.Action, 2),
		q.Action * aWins,
		q.Action * p[0],
		q.Action * p[1],
	}
}
func (v *NNetScore) Save(filename string) {
	v.n.Save(filename)
}
