package main

import (
"fmt"
"os"
"flag"
)

type Args struct {
	Resume bool
	Initiate bool
	Learn bool
	Predict bool
	LayerSize int
	LearningRate float64
}

func (a *Args) Parse() {
	flag.BoolVar(&a.Resume, "resume", false, "Resume from previous trained model")
	flag.BoolVar(&a.Initiate, "init", false, "Initiate the neural net approximation")
	flag.BoolVar(&a.Learn, "learn", false, "Train the neural net approximation")
	flag.BoolVar(&a.Predict, "pred", false, "Evaluate the value function approximation")
	flag.IntVar(&a.LayerSize, "layer", 30, "The size neural net's hidden layer")
	flag.Float64Var(&a.LearningRate, "alpha", 0.1, "The learning rate")
	flag.Parse()

	numFlagsOn := 0
	if a.Initiate {
		numFlagsOn++
	}
	if a.Learn {
		numFlagsOn++
	}
	if a.Predict {
		numFlagsOn++
	}

	if numFlagsOn == 0 {
		a.Usage("One of flages init, lean, or pred must be turned on.")
	}
	if numFlagsOn > 1 {
		a.Usage("Only one flag can be turned on.")
	}

	if a.Learn {
		fmt.Println("layer flag will not be applied.")
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