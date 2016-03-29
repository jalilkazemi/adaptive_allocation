package main

import (
	b "bandit"
	"encoding/csv"
	rng "github.com/leesper/go_rng"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

const (
	trainSize      = 1000
	testSize       = 500
	trajectorySize = 5
	// fit_filename   = "D:/Documents and Settings/Nazi/Desktop/score_nnet.js"
)

var (
	predict_filenames = []string{
		"D:/Documents and Settings/Nazi/Desktop/score_approx_1.csv",
		"D:/Documents and Settings/Nazi/Desktop/score_approx_2.csv",
	}
)

// TODO
// bias the starting points toward the close-to-terminal states
func main() {
	a := new(Args)
	a.Parse()

	if a.Learn {
		learn(a.Resume, a.LayerSize, a.LearningRate, a.FitFilename)
	} else if a.Predict {
		predict(a.FitFilename)
	}
}

func learn(resume bool, layerSize int, alpha float64, fitFilename string) {
	seed := time.Now().UTC().UnixNano()
	rnd := rand.New(rand.NewSource(seed))
	rndBeta := rng.NewBetaGenerator(seed)
	f := b.NewABTestQFactory(rnd, rndBeta)
	// qTrain := make([]b.ABTestQ, trainSize)
	// for i := 0; i < trainSize; i++ {
	// 	qTrain[i] = f.Next()
	// }
	qTrain := f.Next()
	qTest := make([]b.ABTestQ, testSize)
	for i := 0; i < testSize; i++ {
		qTest[i] = f.Next()
	}

	var v b.NNetScore
	if resume {
		v = b.LoadNNetScore(fitFilename)
	} else {
		v = b.NewNNetScore(layerSize)
	}
	for epoch := 0; epoch < 100; epoch++ {
		// for i, _ := range qTrain {
		// 	_, terminal := v.Backpropagate(rnd, trajectorySize, &qTrain[i], true, alpha)
		// 	if terminal {
		// 		// println(i)
		// 		qTrain[i] = f.Next()
		// 	}
		// }
		_, steps := v.BackpropagateDeeply(rnd, qTrain, alpha)
		println(steps)
		qTrain = f.Next()

		rmse := new(RMSE)
		for _, q := range qTest {
			yerr := v.Error(rnd, q)
			rmse.Add(yerr)
		}
		log.Printf("Epoch %d: RMSE=%f", epoch, rmse.Eval())
	}
	v.Save(fitFilename)
	log.Printf("Stored the neural net fit in %s\n", fitFilename)
}

type RMSE struct {
	count int
	mse   float64
}

func (r *RMSE) Add(err float64) {
	r.count++
	r.mse += (math.Pow(err, 2) - r.mse) / float64(r.count)
}
func (r *RMSE) Eval() float64 {
	return math.Sqrt(r.mse)
}
func predict(fitFilename string) {
	v := b.LoadNNetScore(fitFilename)
	ComputeVector(v, 50, 70, 0.2, 0.25, predict_filenames[0])
	ComputeVector(v, 1000, 1000, 0.002, 0.003, predict_filenames[1])
}

func ComputeVector(v b.NNetScore, n0, n1 int, p0, p1 float64, filename string) {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	w.Comma = ','

	n := [2]float64{float64(n0), float64(n1)}
	k := [2]float64{p0 * float64(n0), p1 * float64(n1)}
	for action := 0.0; action < 1.0; action += 0.01 {
		row := []string{strconv.FormatFloat(
			v.ScoreOf(b.ABTestQ{Action: action, State: b.ABTestS{N: n, K: k}}),
			'f', 6, 64)}
		w.Write(row)
	}
	log.Printf("Stored the prediction {n0=%d,n1=%d,p0=%.4f,p1=%.4f} in %s\n", n0, n1, p0, p1, filename)
}
