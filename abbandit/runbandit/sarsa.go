package main

import (
	b "abbandit"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

const (
	testSize   = 500
	maxEpisode = 100
)

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

func sarsa(resume bool, layerSize int, lambda, epsilon, alpha float64, fitFilename string, logFilename string) {
	seed := time.Now().UTC().UnixNano()
	rnd := rand.New(rand.NewSource(seed))
	f := b.NewActionFactory(rnd)

	trainerAction := f.Next()
	testActions := make([]b.Action, testSize)
	for i := 0; i < testSize; i++ {
		testActions[i] = f.Next()
	}

	var v b.ActionValue
	if resume {
		v = b.LoadActionValue(fitFilename)
	} else {
		v = b.NewActionValue(layerSize)
	}
	rmseSeries := make([]float64, maxEpisode)
	for epoch := 0; epoch < maxEpisode; epoch++ {
		fullEpisode(rnd, v, trainerAction, lambda, epsilon, alpha)
		trainerAction = f.Next()

		rmse := new(RMSE)
		for _, q := range testActions {
			yerr := v.Error(rnd, q)
			rmse.Add(yerr)
		}
		log.Printf("Epoch %d: RMSE=%f", epoch, rmse.Eval())
		rmseSeries[epoch] = rmse.Eval()
	}
	v.Save(fitFilename)
	log.Printf("Stored the neural net fit in %s\n", fitFilename)
	logFloat64(rmseSeries, resume, logFilename)
	log.Printf("Stored the RMSE series in %s\n", logFilename)
}

func fullEpisode(rnd *rand.Rand, v b.ActionValue, q b.Action, lambda, epsilon, learningRate float64) (steps int) {
	actionTrace := []b.Action{q}
	greedyMoves := []bool{false}
	y := v.ValueOf(q)
	for !q.IsTerminalState() {
		steps++
		s := q.Travel(rnd)
		var isGreedy bool
		random := rnd.Float64()
		var yPrime float64
		if random < epsilon {
			q = b.Action{State: s, Move: random / epsilon}
			yPrime = v.ValueOf(q)
		} else {
			isGreedy = true
			var bestMove float64
			bestMove, yPrime = s.BestMove(v.ValueOf)
			q = b.Action{State: s, Move: bestMove}
		}

		decay := 1.0
		for i := len(actionTrace) - 1; i >= 0; i-- {
			v.Update(actionTrace[i], yPrime-y, true, decay*learningRate)
			decay *= lambda
		}
		y = yPrime
		actionTrace = append(actionTrace, q)
		greedyMoves = append(greedyMoves, isGreedy)
	}
	yPrime := q.FaceValue()
	decay := 1.0
	for i := len(actionTrace) - 1; i >= 0; i-- {
		v.Update(actionTrace[i], yPrime-y, true, decay*learningRate)
		decay *= lambda
	}

	logTrace(actionTrace, greedyMoves)
	return
}

func logTrace(actionTrace []b.Action, greedyMoves []bool) {
	fmt.Printf("\nSteps: %d\n", len(actionTrace)-1)
	fmt.Printf("*** Start: (%.2f in %.0f, %.2f in %.0f) ***\n",
		actionTrace[0].Prob(0),
		actionTrace[0].N[0],
		actionTrace[0].Prob(1),
		actionTrace[0].N[1],
	)
	last := len(actionTrace) - 1
	fmt.Printf("### End: (%.2f in %.0f, %.2f in %.0f) ###\n",
		actionTrace[last].Prob(0),
		actionTrace[last].N[0],
		actionTrace[last].Prob(1),
		actionTrace[last].N[1],
	)
	println("Trace:")
	for i, action := range actionTrace {
		if greedyMoves[i] {
			fmt.Printf(" =>%.2f", action.Move)
		} else {
			fmt.Printf(" ->%.2f", action.Move)
		}
	}
	println("\n")
}
