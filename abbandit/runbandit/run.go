package main

import (
	b "abbandit"
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

var (
	predict_filenames = []string{
		"D:/Documents and Settings/Nazi/Desktop/score_approx_1.csv",
		"D:/Documents and Settings/Nazi/Desktop/score_approx_2.csv",
	}
)

func main() {
	a := new(Args)
	a.Parse()

	if a.Learn {
		sarsa(a.Resume, a.LayerSize, a.Lambda, a.Epsilon, a.LearningRate, a.FitFilename, a.LogFilename)
	} else if a.Predict {
		predict(a.FitFilename)
	}
}

func predict(fitFilename string) {
	v := b.LoadActionValue(fitFilename)
	ComputeVector(v, 50, 70, 0.2, 0.25, predict_filenames[0])
	ComputeVector(v, 1000, 1000, 0.002, 0.003, predict_filenames[1])
}

func ComputeVector(v b.ActionValue, n0, n1 int, p0, p1 float64, filename string) {
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
	for move := 0.0; move < 1.0; move += 0.01 {
		row := []string{strconv.FormatFloat(
			v.ValueOf(b.Action{State: b.State{N: n, K: k}, Move: move}),
			'f', 6, 64)}
		w.Write(row)
	}
	log.Printf("Stored the prediction {n0=%d,n1=%d,p0=%.4f,p1=%.4f} in %s\n", n0, n1, p0, p1, filename)
}
