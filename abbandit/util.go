package abbandit

func max(a, b float64) float64 {
	if a > b {
		return a
	} else {
		return b
	}
}

func maxOf(values ...float64) (ind int, val float64) {
	for i, v := range values {
		if i == 0 || v > val {
			ind = i
			val = v
		}
	}
	return
}
