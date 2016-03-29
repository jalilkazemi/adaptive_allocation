package main

import (
	"flag"
	"fmt"
	"os"
)

type Args struct {
	Resume       bool
	Learn        bool
	Predict      bool
	LayerSize    int
	LearningRate float64
	FitFilename  string
}

func (a *Args) Parse() {
	flag.BoolVar(&a.Resume, "resume", false, "Resume from previous trained model")
	flag.BoolVar(&a.Learn, "learn", false, "Train the neural net approximation")
	flag.BoolVar(&a.Predict, "pred", false, "Evaluate the value function approximation")
	flag.IntVar(&a.LayerSize, "layer", 30, "The size neural net's hidden layer")
	flag.Float64Var(&a.LearningRate, "alpha", 0.1, "The learning rate")
	flag.StringVar(&a.FitFilename, "file", "D:/Documents and Settings/Nazi/Desktop/score_nnet.js", "The file that contains the neural net fit")
	flag.Parse()

	numFlagsOn := 0
	if a.Learn {
		numFlagsOn++
	}
	if a.Predict {
		numFlagsOn++
	}

	if numFlagsOn == 0 {
		a.Usage("One of the learn or pred flags must be turned on.")
	}
	if numFlagsOn > 1 {
		a.Usage("Only one of the learn or pred flags can be turned on.")
	}

	if a.Predict {
		fmt.Println("resume, layer and alpha flags will not be applied.")
	}
	if a.Resume {
		fmt.Println("layer flag will not be applied")
	}
}

func (a *Args) Usage(message string) {
	fmt.Println(message)
	flag.Usage()
	os.Exit(1)
}
