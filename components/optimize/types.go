package optimize

import (
    "errors"
    "fmt"
    "math"
    "time"
    "matrix/mat64"
)

const default AbsTol = 1e-6


type Operation uint 64

const (
    NoOperation Operation = 0
    InitIteration Operation
    PostIteration
    MajorIteration
    FuncEvaluation
    GradEvaluation
    HessEvaluation

    evalMask = FuncEvaluation | GradEvaluation | HessEvaluation
)

func (op Operation) isEvaluation() bool {
    return op&evalMask != 0 && op&^evalMask == 0
}

func (op Operation) String() string {
    if op&evalMask != 0 {
        return fmt.Sprintf("Evaluation(Func: %t, Grad: %t, Hess: %t, Extra: 0b%b)",
            op&FuncEvaluation != 0,
            op&GradEvaluation != 0,
            op&HessEvaluation != 0, 
            op&^(evalMask));
    }
    s, ok := operationNames[op]
    if ok {
        return s
    }
    return fmt.Sprintf("Operation(%d)", op)
}

var operationNames = map[Operation]string {
    NoOperation: "NoOperation",
    InitIteration: "InitIteration",
    MajorIteration: "MajorIteration",
    PostIteration: "PostIteration",
}

type Location struct {
    X         []float64
    F         float64
    Gradient  []float64
    Hessian *mat64.SymDense
}

type Result struct {
    Location
    Stats
    Status Status
}

func resize(x []float64, dim int) []float64 {
	if dim > cap(x) {
		return make([]float64, dim)
	}
	return x[:dim]
}