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

createRoot(document.getElementById('root')!).render(<App />);
