package main

import (
b "bandit"
"math"
"flag"
"strconv"
"log"
"os"
"encoding/csv"
)
const (
sampleSize = 10000
)

// TODO
// Avoid overfit: change the random sample every other epoch
//                use trajectory to generate sample points

func main() {
	a:=new(Args)
	a.Parse()

	if a.Initiate {
		initiate(a.Resume, a.LayerSize, a.LearningRate)
	} else if a.Learn {
		learn(a.Resume, a.LayerSize, a.LearningRate)
	} else if a.Predict {
		predict()
	}
}

func initiate(resume bool, layerSize int, alpha float64) {
	stater := b.NewRandomSeedABStater()	
	tArr := make([]b.ABState, sampleSize)
	for i := 0; i < sampleSize; i++ {
		tArr[i]=stater.Next()
	}

	var v b.NNetValue
	if resume {
		v = b.LoadNNetValue("D:/Documents and Settings/Nazi/Desktop/value_nnet_init.js")	
	} else {
		v = b.NewNNetValue(layerSize)		
	}
	for epoch := 0; epoch < 100; epoch++ {
		dynalpha := alpha 
		rmse := new(RMSE)
		for _, t :=range tArr {
			yerr := v.BackpropagateUsing(t, b.GreedyValueOf(t), dynalpha)
			rmse.Add(yerr) 
		}
		log.Printf("Epoch %d: RMSE=%f", epoch, rmse.Eval())
	}
	v.Save("D:/Documents and Settings/Nazi/Desktop/value_nnet_init.js")
}

func learn(resume bool, layerSize int, alpha float64) {
	stater := b.NewRandomSeedABStater()	
	tArr := make([]b.ABState, sampleSize)
	for i := 0; i < sampleSize; i++ {
		tArr[i]=stater.Next()
	}

	var v b.NNetValue
	if resume {
		v = b.LoadNNetValue("D:/Documents and Settings/Nazi/Desktop/value_nnet.js")	
	} else {
		v = b.LoadNNetValue("D:/Documents and Settings/Nazi/Desktop/value_nnet_init.js")				
	}
	for epoch := 0; epoch < 100; epoch++ {
		rmse :=new(RMSE)
		for _,t :=range tArr {
			yerr := v.BackpropagateDeeply(t, alpha)
			rmse.Add(yerr)
		}
		log.Printf("Epoch %d: RMSE=%f", epoch, rmse.Eval())
	}
	v.Save("D:/Documents and Settings/Nazi/Desktop/value_nnet.js")
}

type RMSE struct {
	count int
	mse float64
}
func (r *RMSE) Add(err float64) {
	r.count++
	r.mse += (math.Pow(err, 2) - r.mse) / float64(r.count)
}
func (r *RMSE) Eval() float64 {
	return math.Sqrt(r.mse)
}
func predict() {
	v := b.LoadNNetValue("D:/Documents and Settings/Nazi/Desktop/value_nnet.js")	
	ComputeMatrix(v, 2, 2, "D:/Documents and Settings/Nazi/Desktop/value_approx_2-2.csv")
	ComputeMatrix(v, 10, 10, "D:/Documents and Settings/Nazi/Desktop/value_approx_10-10.csv")
	ComputeMatrix(v, 100, 100, "D:/Documents and Settings/Nazi/Desktop/value_approx_100-100.csv")
}

func ComputeMatrix(v b.NNetValue, n0, n1 int, filename string) {
	f,err:=os.Create(filename)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()
	w:=csv.NewWriter(f)
	defer w.Flush()
	w.Comma = ','

	n := [2]float64{float64(n0), float64(n1)}
	for p0 := 0.0; p0 < 1.0; p0+=0.01 {
		row :=make([]string, 0, 101)
		for p1 := 0.0; p1 < 1.0; p1+=0.01 {
			row = append(row, strconv.FormatFloat(
				v.ValueOf(b.ABState{n, [2]float64{p0, p1}}), 'f', 6, 64,
				))
		}
		w.Write(row)
	}

} 

func depreciated_main() {
	var iter int
	flag.IntVar(&iter, "iter", 0, "The number of value iterations")
	flag.Parse()

valueOf:=b.LastStepValueOf
for i := 0; i < iter; i++ {
	valueOf=b.IteratedValueOf(valueOf)
}

stater := b.NewRandomSeedABStater()	
tArr := make([]b.ABState, sampleSize)
for i := 0; i < sampleSize; i++ {
	tArr[i]=stater.Next()
}
values :=make([]float64, sampleSize)

for i, t :=range tArr {
	values[i] = valueOf(t)
}
storeValues(tArr, values, "D:/Documents and Settings/Nazi/Desktop/value_nnet_temp.csv")
}

func storeValues(tArr []b.ABState, values []float64, filename string) {
	f,err:=os.Create(filename)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()
	w:=csv.NewWriter(f)
	defer w.Flush()
	w.Comma = ','
	for i, t:=range tArr {
		w.Write([]string {
			strconv.FormatFloat(t.N[0], 'f', 6, 64),
			strconv.FormatFloat(t.N[1], 'f', 6, 64),
			strconv.FormatFloat(t.P[0], 'f', 6, 64),
			strconv.FormatFloat(t.P[1], 'f', 6, 64),
			strconv.FormatFloat(values[i], 'f', 6, 64),
			})
	}
}
