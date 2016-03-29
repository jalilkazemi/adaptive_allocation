package bandit

import (
"math/rand"
"math"
"time"
)
type ABStater struct {
r *rand.Rand
}
func NewABStater(seed int64) ABStater {
	return ABStater{rand.New(rand.NewSource(seed))}
}
func NewRandomSeedABStater() ABStater {
	return NewABStater(time.Now().UTC().UnixNano())
}
func (stater *ABStater) Next()ABState { // TODO use dirichlet dist. to enforce close p0 and p1 values
	t :=ABState{}
	t.N[0] = 1.0/math.Pow(stater.r.Float64()/2, 3)
	t.N[1] = 1.0/math.Pow(stater.r.Float64()/2, 3)
	t.P[0] = stater.r.Float64()
	t.P[1] = stater.r.Float64()
	return t
}
type ABState struct {
	N [2]float64
	P [2]float64
}

func (t *ABState) ActA() (gain float64, responses []ABState, probs []float64) {
return t.act(0)
}
func (t *ABState) ActB() (gain float64, responses []ABState, probs []float64) {
return t.act(1)
}

func (t *ABState) act(a int) (gain float64, responses []ABState, probs []float64) {
	gain = t.P[a]
	responses=make([]ABState, 2)
	probs = make([]float64, 2)

	probs[0]=t.P[a]	
	probs[1]=1-t.P[a]

	responses[0].N[a]=t.N[a]+1
	responses[0].N[1-a]=t.N[1-a]
	responses[1].N[a]=t.N[a]+1
	responses[1].N[1-a]=t.N[1-a]

	responses[0].P[a]=(t.P[a]*t.N[a]+1)/(t.N[a]+1)
	responses[0].P[1-a]=t.P[1-a]
	responses[1].P[a]=(t.P[a]*t.N[a])/(t.N[a]+1)
	responses[1].P[1-a]=t.P[1-a]

	return
}