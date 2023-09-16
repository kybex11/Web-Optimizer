package validate

import (
	"math"
	"testing"
	"dif/fd"
	"floats"
)
type function interface {
	Func(x []float64) float64
}
type gradient interface {
	Grad(grad, x []float64)
}
type minimumer interface {
	function
	Minima() []Minimum
}
type Minimum struct {
	X []float64
	F float64
	Global bool
}

type funcTest struct {
	X []float64
	F float64
	Gradient []float64
}


const (
	defaultTol       = 1e-12
	defaultGradTol   = 1e-9
	defaultFDGradTol = 1e-5
)

func testFunction(f function, ftests []funcTest, t *testing.T) {
	tests := make([]funcTest, len(ftests))
	copy(tests, ftests)

	fMinima, isMinimumer := f.(minimumer)
	fGradient, isGradient := f.(gradient)

	if isMinimumer {
		for _, minimum := range fMinima.Minima() {
			var grad []float64
			if isGradient {
				grad = make([]float64, len(minimum.X))
			}
			tests = append(tests, funcTest{
				X:        minimum.X,
				F:        minimum.F,
				Gradient: grad,
			})
		}
	}

	for i, test := range tests {
		F := f.Func(test.X)

		if math.Abs(F-test.F) > defaultTol {
			t.Errorf("Test #%d: function value given by Func is incorrect. Want: %v, Got: %v",
				i, test.F, F)
		}

		if test.Gradient == nil {
			continue
		}

		fdGrad := fd.Gradient(nil, f.Func, test.X, &fd.Settings{
			Formula: fd.Central,
			Step:    1e-6,
		})

		if !floats.EqualApprox(fdGrad, test.Gradient, defaultFDGradTol) {
			dist := floats.Distance(fdGrad, test.Gradient, math.Inf(1))
			t.Errorf("Test #%d: numerical and expected gradients do not match. |fdGrad - WantGrad|_∞ = %v",
				i, dist)
		}

		if isGradient {
			grad := make([]float64, len(test.Gradient))
			fGradient.Grad(grad, test.X)

			if !floats.EqualApprox(grad, test.Gradient, defaultGradTol) {
				dist := floats.Distance(grad, test.Gradient, math.Inf(1))
				t.Errorf("Test #%d: gradient given by Grad is incorrect. |grad - WantGrad|_∞ = %v",
					i, dist)
			}
		}
	}
}