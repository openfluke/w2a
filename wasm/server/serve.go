// Local static server for Welvet WASM HTML benches (MIME application/wasm).
package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	root := "../typescript/dist"
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	abs, _ := filepath.Abs(root)
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		rel := strings.TrimPrefix(filepath.Clean(r.URL.Path), "/")
		if rel == "" || rel == "." {
			rel = "matrix.html"
		}
		path := filepath.Join(abs, rel)
		if filepath.Ext(path) == ".wasm" {
			w.Header().Set("Content-Type", "application/wasm")
		}
		http.ServeFile(w, r, path)
	})
	addr := ":3000"
	fmt.Printf("Serving %s at http://localhost%s\n", abs, addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
