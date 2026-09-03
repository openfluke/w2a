import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// No StrictMode — Go WASM init is not safe to double-mount.
createRoot(document.getElementById("root")!).render(<App />);
