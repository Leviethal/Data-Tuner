import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Thumbnail } from "./screens/Thumbnail";

createRoot(document.getElementById("app") as HTMLElement).render(
  <StrictMode>
    <Thumbnail />
  </StrictMode>,
);
