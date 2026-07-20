package entity_test

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/model/entity"
)

func TestWriteInspectLoadBlobRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tiny.entity")

	payload := make([]byte, 0, 64)
	f32 := []float32{1.5, -2.25, 0.125}
	for _, v := range f32 {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		payload = append(payload, b[:]...)
	}
	f16off := uint64(len(payload))
	for _, v := range []float32{3.0, 4.0} {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], core.Float32ToFloat16(v))
		payload = append(payload, b[:]...)
	}
	tok := []byte(`{"version":"1.0"}`)
	tokOff := uint64(len(payload))
	payload = append(payload, tok...)

	spec := &entity.TransformerSpec{
		Architecture: "llama_style_decoder",
		HiddenSize:   8,
		VocabSize:    16,
		HasFinalNorm: true,
		Engine:       "welvet",
		Dims: &entity.TransformerDims{
			NumLayers:        1,
			NumHeads:         2,
			NumKVHeads:       2,
			HeadDim:          4,
			IntermediateSize: 16,
		},
	}
	blobs := []entity.WeightBlob{
		{Path: "transformer.embeddings", Offset: 0, Length: 12, DType: "FLOAT32", Format: "none", Native: true},
		{Path: "transformer.norm", Offset: f16off, Length: 4, DType: "FLOAT16", Format: "none", Native: true},
		{Path: entity.TokenizerBlobPath, Offset: tokOff, Length: uint64(len(tok)), DType: "JSON", Format: "none", Native: true},
	}
	if err := entity.WriteTransformerFile(path, spec, blobs, payload); err != nil {
		t.Fatal(err)
	}
	if !entity.IsEntity(path) {
		t.Fatal("IsEntity false")
	}
	info, err := entity.Inspect(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Status != "packed" || info.BlobCount != 3 || !info.HasTokenizer {
		t.Fatalf("info=%+v", info)
	}
	if info.Transformer == nil || info.Transformer.HiddenSize != 8 {
		t.Fatalf("transformer=%+v", info.Transformer)
	}

	ef, err := entity.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ef.Close()

	emb, err := ef.LoadBlob("transformer.embeddings")
	if err != nil {
		t.Fatal(err)
	}
	if len(emb) != 3 || emb[0] != 1.5 || emb[1] != -2.25 {
		t.Fatalf("emb=%v", emb)
	}
	norm, err := ef.LoadBlob("transformer.norm")
	if err != nil {
		t.Fatal(err)
	}
	if len(norm) != 2 {
		t.Fatalf("norm=%v", norm)
	}
	tokBytes, err := ef.LoadTokenizerJSON()
	if err != nil {
		t.Fatal(err)
	}
	if string(tokBytes) != string(tok) {
		t.Fatalf("tok=%q", tokBytes)
	}
	if _, err := ef.LoadBlob(entity.TokenizerBlobPath); err == nil {
		t.Fatal("expected JSON blob LoadBlob error")
	}
}

func TestImportFromHFRejectsMissing(t *testing.T) {
	err := entity.ImportFromHF("", "out.entity", entity.PackOptions{})
	if err == nil {
		t.Fatal("expected error")
	}
}
