module github.com/openfluke/w2a

go 1.22

require github.com/openfluke/welvet v0.0.0

require github.com/openfluke/webgpu v1.0.4 // indirect

replace (
	github.com/openfluke/webgpu => ../../webgpu
	github.com/openfluke/welvet => ../
)
