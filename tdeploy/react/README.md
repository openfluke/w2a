# React (Vite) frontend — `@openfluke/welvet` from npmjs

```bash
bash run.sh                 # install + build + headless Chrome smoke
npm run dev                 # http://localhost:5177/
```

`postinstall` / `sync-wasm` copies `main.wasm` + `wasm_exec.js` into `public/`.
`index.html` loads `/wasm_exec.js` before the app; `initBrowser("/main.wasm")` boots Welvet.
