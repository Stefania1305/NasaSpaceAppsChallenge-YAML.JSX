# Fire Pattern Predictor — frontend

React + TypeScript + Vite. The Python backend (`../server.py`) serves the built app at
**http://localhost:8000/app**. The original single-file version is still available at `/app-old`.

## Everyday use

The built app is already in `dist/`, so just run the backend:

```
cd ..
python server.py
```

## Developing

```
npm install        # once
npm run dev        # http://localhost:5173/app/ with instant reload; API calls go to the backend on :8000
npm run build      # type-check and rebuild dist/ (then reload http://localhost:8000/app)
```

Keep `python server.py` running while you use `npm run dev`.

## Where things are

| File | What it does |
|---|---|
| `src/engine.ts` | Everything on the map: the globe, space backdrop (stars, embers, halo), spin, the glide between layouts, globe sizing, drawing, the pin, map layers and popups. |
| `src/store.ts` | App state and actions (Zustand): zones, the Analyse window, loading, predicting, the forecast, the panel and the opening screen. |
| `src/components/` | The UI: `Splash` (opening screen), `Panel` (pill + side panel), `ZoneSection`, `Forecast`, `AnalyseModal`, `MapView`. |
| `src/styles.css` | All styles (carried over unchanged from the original app). |
| `src/geo.ts`, `src/format.ts`, `src/api.ts` | Geometry helpers, number/date formatting, backend calls. |

The page's top-level state also lives in classes on `<body>` (`splash`, `collapsed`, `intro`), which the CSS uses
for the opening-screen transition and the panel slide.
