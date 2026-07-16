package sequential

import "github.com/openfluke/w2a/suites"

func rec(op, dt, format, backend, grid, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer:   "sequential",
		Op:      op,
		DType:   dt,
		Format:  format,
		Backend: backend,
		Grid:    grid,
		Status:  status,
		Note:    note,
	})
}
