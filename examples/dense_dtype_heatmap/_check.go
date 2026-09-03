package main
import (
  "fmt"
  "github.com/openfluke/welvet/architecture"
  "github.com/openfluke/welvet/core"
  "github.com/openfluke/welvet/layers/dense"
  "github.com/openfluke/welvet/quant"
  "github.com/openfluke/welvet/runtime/forward"
  "github.com/openfluke/welvet/stub/serialization"
  "github.com/openfluke/welvet/weights"
)
func main() {
  g := architecture.NewGrid(1,1,1,1)
  init := make([]float32, 16)
  for i:=0;i<4;i++ { init[i*4+i]=1 }
  l,_ := dense.NewConfigured(4,4,core.ActivationLinear,core.DTypeFloat32,quant.FormatNone,init)
  _ = dense.Place(g,0,0,0,0,l)
  x := core.NewTensor[float32](1,4)
  for i:=0;i<4;i++ { x.Data[i]=float32(i+1) }
  a,_ := forward.Forward(g,x)
  ent,_ := serialization.SerializeEntity(g)
  g2,_ := serialization.DeserializeEntity(ent)
  b,_ := forward.Forward(g2,x)
  fmt.Println("before", a.Output.Data)
  fmt.Println("after ", b.Output.Data)
  dl := g2.At(0,0,0,0).Op.(*dense.Layer)
  _ = weights.Convert(dl.Weights, weights.ConvertOpts{DType: core.DTypeFloat32, Format: quant.FormatNone})
  c,_ := forward.Forward(g2,x)
  fmt.Println("conv32", c.Output.Data)
}
