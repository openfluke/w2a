package polyops

import (
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

// MakeBackend builds a Kind grid then stamps Exec.Backend onto the Op tree
// (so Forward/Backward use DotTile / Saxpy when BackendSIMD).
func MakeBackend(k Kind, dt core.DType, format quant.Format, be core.Backend) (*architecture.Grid, error) {
	g, err := k.Make(dt, format)
	if err != nil {
		return nil, err
	}
	StampExec(g, be)
	return g, nil
}

// StampExec sets g.Exec.Backend and mirrors ExecConfig onto every cell Op
// (including Dense children inside MHA / SwiGLU / CNN / RNN / …).
func StampExec(g *architecture.Grid, be core.Backend) {
	if g == nil {
		return
	}
	g.Exec.Backend = be
	for i := range g.Cells {
		stampOpExec(g.Cells[i].Op, g.Exec)
	}
}

func stampOpExec(op any, exec core.ExecConfig) {
	switch v := op.(type) {
	case *dense.Layer:
		if v != nil {
			v.Exec = exec
		}
	case *rmsnorm.Layer:
		if v != nil {
			v.Exec = exec
		}
	case *layernorm.Layer:
		if v != nil {
			v.Exec = exec
		}
	case *swiglu.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		stampOpExec(v.Gate, exec)
		stampOpExec(v.Up, exec)
		stampOpExec(v.Down, exec)
	case *mha.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		stampOpExec(v.Q, exec)
		stampOpExec(v.K, exec)
		stampOpExec(v.V, exec)
		stampOpExec(v.O, exec)
	case *cnn1.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		stampOpExec(v.Proj, exec)
	case *cnn2.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		stampOpExec(v.Proj, exec)
	case *cnn3.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		stampOpExec(v.Proj, exec)
	case *rnn.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		stampOpExec(v.IH, exec)
		stampOpExec(v.HH, exec)
	case *lstm.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		for _, g := range []*lstm.Gate{v.I, v.F, v.G, v.O} {
			if g == nil {
				continue
			}
			stampOpExec(g.IH, exec)
			stampOpExec(g.HH, exec)
		}
	case *embedding.Layer:
		if v != nil {
			v.Exec = exec
		}
	case *sequential.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		for _, ch := range v.Children {
			stampOpExec(ch, exec)
		}
	case *residual.Layer:
		if v == nil {
			return
		}
		v.Exec = exec
		for _, ch := range v.Children {
			stampOpExec(ch, exec)
		}
	case *softmax.Layer:
		if v != nil {
			v.Exec = exec
		}
	}
}
