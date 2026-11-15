import { createRoot } from 'react-dom/client';
import 'maplibre-gl/dist/maplibre-gl.css';
// Type: IBM Plex Sans (variable, every UI and display role) and IBM Plex Mono (data and measurements only),
// bundled with the app so they also work offline.
import '@fontsource-variable/ibm-plex-sans';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import '@fontsource/ibm-plex-mono/latin-600.css';
import './styles.css';
import { App } from './App';
import { Boundary } from './components/Boundary';

const crashed = (
  <div role="alert" style={{ position: 'fixed', inset: 0, zIndex: 3000, display: 'grid', placeContent: 'center', gap: 14,
                             textAlign: 'center', background: '#07090c', color: '#f5f5f4' }}>
    <b>Something went wrong.</b>
    <button type="button" onClick={() => location.reload()}>Reload the page</button>
  </div>
);
createRoot(document.getElementById('root')!).render(<Boundary fallback={crashed}><App /></Boundary>);
