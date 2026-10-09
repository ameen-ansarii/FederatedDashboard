import React, { useEffect, useState, useRef } from 'react';
import Lenis from 'lenis';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { FederatedNodes } from './components/FederatedNodes';
import { TrainingEngine } from './components/TrainingEngine';
import { StudentPredictor } from './components/StudentPredictor';
import { BenchmarkTable } from './components/BenchmarkTable';
import { BatchUploader } from './components/BatchUploader';
import { Footer } from './components/Footer';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('topology');

  // References for scrolling
  const topologyRef = useRef<HTMLDivElement>(null);
  const trainingRef = useRef<HTMLDivElement>(null);
  const batchRef = useRef<HTMLDivElement>(null);
  const predictorRef = useRef<HTMLDivElement>(null);
  const benchmarksRef = useRef<HTMLDivElement>(null);

  // Initialize Lenis Smooth Scroll
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    });

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    const rafId = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(rafId);
      lenis.destroy();
    };
  }, []);

  // Scroll to section when tab changes
  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    let targetRef: React.RefObject<HTMLDivElement | null> | null = null;

    if (tabId === 'topology') targetRef = topologyRef;
    if (tabId === 'training') targetRef = trainingRef;
    if (tabId === 'batch') targetRef = batchRef;
    if (tabId === 'predictor') targetRef = predictorRef;
    if (tabId === 'benchmarks') targetRef = benchmarksRef;

    if (targetRef?.current) {
      targetRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans selection:bg-zinc-100 selection:text-zinc-950">
      <Navbar
        activeTab={activeTab}
        setActiveTab={handleTabChange}
      />

      <main className="flex-1">
        <Hero onExplore={handleTabChange} />

        <div ref={topologyRef}>
          <FederatedNodes />
        </div>

        <div ref={trainingRef}>
          <TrainingEngine />
        </div>

        <div ref={batchRef}>
          <BatchUploader />
        </div>

        <div ref={predictorRef}>
          <StudentPredictor />
        </div>

        <div ref={benchmarksRef}>
          <BenchmarkTable />
        </div>
      </main>

      <Footer />
    </div>
  );
}
