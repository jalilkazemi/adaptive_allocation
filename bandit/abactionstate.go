package bandit

import (
	rng "github.com/leesper/go_rng"
	// "math"
	"math/rand"
)

const (
	minStep = 100
	maxStep = 10000
)

type ABTestQFactory struct {
	rnd     *rand.Rand
	rndBeta *rng.BetaGenerator
}

func NewABTestQFactory(rnd *rand.Rand, rng *rng.BetaGenerator) ABTestQFactory {
	return ABTestQFactory{rnd, rng}
}
func (f *ABTestQFactory) Next() ABTestQ { // separate training sampling (biased to right) and testing sampleing (uniform)
	q := ABTestQ{}
	q.Action = f.rnd.Float64()
	n := f.rnd.Intn(maxStep)
	// n := 2 * (1 + f.rnd.Intn(maxStep-1))
	// n := int(math.Floor(float64(maxStep) * f.rndBeta.Beta(9.0, 1.0)))
	if n == 0 {
		n = 1
	}
	n0 := f.rnd.Intn(n)
	if n0 == 0 {
		n0 = 1
	}
	n1 := n - n0
	if n1 == 0 {
		n1 = 1
		n++
	}
	q.State.N[0] = float64(n0)
	q.State.N[1] = float64(n1)
	q.State.K[0] = float64(f.rnd.Intn(n0))
	q.State.K[1] = float64(f.rnd.Intn(n1))
	return q
}

type ABTestS struct {
	N [2]float64
	K [2]float64
}

func (s *ABTestS) IsTerminalState() bool {
	return s.N[0]+s.N[1] >= maxStep
}
func (s *ABTestS) FaceScore() float64 {
	return (s.K[0] + s.K[1]) / (s.N[0] + s.N[1])
}
func (s *ABTestS) Prob(segment int) float64 {
	return s.K[segment] / s.N[segment]
}
func (s *ABTestS) AddPositive(segment int) {
	s.N[segment]++
	s.K[segment]++
}
func (s *ABTestS) AddNegative(segment int) {
	s.N[segment]++
}
func (s *ABTestS) BestAction(scoreOf func(ABTestQ) float64) (bestAction, value float64) {
	scores := make([]float64, 101)
	for i := 0; i <= 100; i++ {
		action := float64(i) / 100.0
		scores[i] = scoreOf(ABTestQ{Action: action, State: *s})
	}
	var bestActionInd int
	bestActionInd, value = maxOf(scores...)
	bestAction = float64(bestActionInd) / 100.0
	return
}

type ABTestQ struct {
	Action float64
	State  ABTestS
}

func (q *ABTestQ) Travel(rnd *rand.Rand) (destination ABTestS) {
	destination = q.State
	for i := 0; i < minStep && q.State.N[0]+q.State.N[1]+float64(i) < float64(maxStep); i++ {
		random := rnd.Float64()
		if random < q.Action {
			if random < q.Action*q.State.Prob(0) {
				destination.AddPositive(0)
			} else {
				destination.AddNegative(0)
			}
		} else {
			if random-q.Action < (1-q.Action)*q.State.Prob(1) {
				destination.AddPositive(1)
			} else {
				destination.AddNegative(1)
			}
		}
	}
	return
}
