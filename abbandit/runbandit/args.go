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
	Lambda       float64
	Epsilon      float64
	LearningRate float64
	FitFilename  string
	LogFilename  string
}

func (a *Args) Parse() {
	flag.BoolVar(&a.Resume, "resume", false, "Resume from previous trained model")
	flag.BoolVar(&a.Learn, "learn", false, "Train the neural net approximation")
	flag.BoolVar(&a.Predict, "pred", false, "Evaluate the value function approximation")
	flag.IntVar(&a.LayerSize, "layer", 30, "The size neural net's hidden layer")
	flag.Float64Var(&a.Lambda, "lambda", 1.0, "The lambda in Sarsa(lambda)")
	flag.Float64Var(&a.Epsilon, "epsilon", 0.01, "The epsilon in eps-greedy policy selection")
	flag.Float64Var(&a.LearningRate, "alpha", 0.1, "The learning rate")
	flag.StringVar(&a.FitFilename, "file", "D:/Documents and Settings/Nazi/Desktop/value_nnet.js", "The file that contains the neural net fit")
	flag.StringVar(&a.LogFilename, "log", "D:/Documents and Settings/Nazi/Desktop/value_nnet_rmse.log", "The file that contains the series of RMSE")
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

	if a.Learn {
		if a.Lambda < 0.0 || a.Lambda > 1.0 {
			a.Usage("Lambda must fall between zero and one.")
		}
		if a.Epsilon < 0.0 || a.Epsilon > 1.0 {
			a.Usage("Epsilon must fall between zero and one.")
		}
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
