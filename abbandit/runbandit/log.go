package main

import (
	"fmt"
	"log"
	"os"
)

func logFloat64(array []float64, exist bool, filename string) {
	var f *os.File
	var err error
	if exist {
		f, err = os.OpenFile(filename, os.O_APPEND, os.ModePerm)
	} else {
		f, err = os.Create(filename)
	}
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()
	for _, val := range array {
		_, err = fmt.Fprintf(f, "%f\n", val)
		if err != nil {
			log.Fatal(err.Error())
		}
	}
}
