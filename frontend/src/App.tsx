import { useEffect } from 'react';
import * as engine from './engine';
import { useApp } from './store';
import { Splash } from './components/Splash';
import { Panel } from './components/Panel';
import { NavBar } from './components/NavBar';
import { AnalyseModal } from './components/AnalyseModal';
import { MapView } from './components/MapView';

export function App() {
  useEffect(() => {
    engine.init();                                  // creates the map once the map elements exist
    useApp.getState().loadSensors();
    // Esc: close the Analyse window, or cancel a half-started drawing / pin placement.
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      const s = useApp.getState();
      if (s.modalOpen && !s.pipelineRunning) s.closeModal();       // not while the pipeline runs (as for the backdrop click)
      if (engine.isDrawing()) engine.setDrawing(false);
      if (engine.isPinMode()) engine.setPinMode(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  return (
    <>
      <Splash />
      <NavBar />
      <Panel />
      <AnalyseModal />
      <MapView />
    </>
  );
}
