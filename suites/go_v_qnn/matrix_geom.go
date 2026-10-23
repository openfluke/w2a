package go_v_qnn

// mx = matrix geometry. Sized so HTP has real work (not 128-dim toys).
// Same numbers for Go / native / NPU. Online-compose layers still pay graph
// build in Go→NPU call; DLL exec is graphExecute only.
var mx = struct {
	DenseB, DenseIn, DenseOut                                           int
	SoftRows, SoftCols                                                  int
	NormB, NormSeq, NormDim                                             int
	Cnn2N, Cnn2H, Cnn2W, Cnn2Cin, Cnn2Cout                              int
	Cnn3N, Cnn3D, Cnn3H, Cnn3W, Cnn3Cin, Cnn3Cout                       int
	ConvT2N, ConvT2H, ConvT2W, ConvT2Cin, ConvT2Cout                    int
	ConvT3N, ConvT3D, ConvT3H, ConvT3W, ConvT3Cin, ConvT3Cout           int
	LstmT, LstmB, LstmIn, LstmHid                                       int
	EmbVocab, EmbDim, EmbN                                              int
	ResN                                                                int
	SwiB, SwiIn, SwiHid                                                 int
	MhaB, MhaSeq, MhaDim, MhaHeads                                      int
}{
	DenseB: 32, DenseIn: 512, DenseOut: 512,
	SoftRows: 64, SoftCols: 4096,
	NormB: 8, NormSeq: 128, NormDim: 512,
	Cnn2N: 4, Cnn2H: 64, Cnn2W: 64, Cnn2Cin: 32, Cnn2Cout: 32,
	Cnn3N: 2, Cnn3D: 16, Cnn3H: 16, Cnn3W: 16, Cnn3Cin: 16, Cnn3Cout: 16,
	ConvT2N: 4, ConvT2H: 32, ConvT2W: 32, ConvT2Cin: 32, ConvT2Cout: 32,
	ConvT3N: 1, ConvT3D: 12, ConvT3H: 12, ConvT3W: 12, ConvT3Cin: 16, ConvT3Cout: 16,
	LstmT: 8, LstmB: 4, LstmIn: 64, LstmHid: 64,
	EmbVocab: 4096, EmbDim: 256, EmbN: 512,
	ResN: 128 * 1024,
	SwiB: 8, SwiIn: 256, SwiHid: 768,
	MhaB: 2, MhaSeq: 32, MhaDim: 128, MhaHeads: 4,
}

func mxGeomNote() string {
	return "dense 32×512×512 · softmax 64×4096 · norms [8,128,512] · " +
		"cnn2 4×64×64×32 · cnn3 2×16³×16 · convt2 4×32×32×32 · convt3 1×12³×16 · " +
		"lstm T8 B4 64/64 · emb 4k×256 n=512 · residual 128k · swiglu 8×256×768 · mha 2×32×128×4"
}
