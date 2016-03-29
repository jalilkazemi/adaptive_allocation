package main

import (
"nnet"
"log"
"strconv"
"os"
)


func main() {
	if len(os.Args) ==1 {
		log.Fatal("Enter real numbers")
	}
	var x []float64
	for i, arg := range os.Args {
		if i ==0 {
			continue
		}
		val, err :=strconv.ParseFloat(arg, 64)
		if err != nil {
			log.Fatal(err.Error())
		}
		x = append(x, val)
	}
	y := 10.3
	n := nnet.NewNNet(len(x), 30)
	println(n.Frontpropagate(x))
	for i := 0; i < 20; i++ {
		n.Backpropagate(x, y, 0.01)
		println(n.Frontpropagate(x))
	}
	n.Save("D:/Documents and Settings/Nazi/Desktop/nnet_temp.js")
println()

	nn := nnet.LoadNNet("D:/Documents and Settings/Nazi/Desktop/nnet_temp.js")
	println(nn.Frontpropagate(x))
}