import React from 'react';
import { FaShieldHalved, FaLock } from 'react-icons/fa6';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/5 bg-zinc-950/80 backdrop-blur-xl font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-white text-zinc-950 flex items-center justify-center font-bold text-xs shadow-md">
            <FaShieldHalved className="w-4 h-4 text-zinc-950" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-sm tracking-tight text-white uppercase">
                Student Retention
              </span>
              <span className="pill-badge pill-purple text-[9px] py-0 px-2">
                FEDERATED ML
              </span>
            </div>
            <p className="text-[11px] text-zinc-500 font-medium hidden sm:block">
              Privacy-preserving learning analytics platform
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center space-x-1.5 border border-white/5 bg-zinc-900/90 p-1.5 rounded-full shadow-inner">
          {[
            { id: 'batch', label: 'Batch CSV Upload' },
            { id: 'topology', label: 'Campuses' },
            { id: 'training', label: 'Federated Training' },
            { id: 'predictor', label: 'Risk Predictor' },
            { id: 'benchmarks', label: 'Test Benchmarks' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-1.5 text-xs font-semibold transition-all rounded-full ${
                activeTab === tab.id
                  ? 'bg-white text-zinc-950 shadow-sm'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Security / Isolation Status */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-2 px-3.5 py-1.5 border border-white/5 bg-zinc-900/80 text-xs font-medium text-zinc-300 rounded-full shadow-sm">
            <FaLock className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-zinc-200">Zero Data Leakage</span>
          </div>
        </div>
      </div>
    </header>
  );
};
