# tdeploy — prove published `@openfluke/welvet` works on Node, Bun, and React

Each subfolder installs **from npmjs** (`1.1.1`), not `file:../typescript`.

| Folder | Runtime | What |
|--------|---------|------|
| [`node/`](node/) | Node.js ≥18 | dense + train + entity + cameral |
| [`bun/`](bun/) | Bun | same suite via `bun run` |
| [`react/`](react/) | Vite + React + Chrome | browser WASM via `initBrowser` + headless smoke |

```bash
cd apps/w2a/tdeploy
bash run.sh                 # all three

bash node/run.sh
bash bun/run.sh             # installs Bun to ~/.bun if needed
bash react/run.sh           # build + puppeteer-core + system Chrome

cd react && npm run dev     # http://localhost:5177/
```

Requires network for `npm`/`bun` install. React smoke needs Google Chrome (`CHROME_PATH` override optional).
