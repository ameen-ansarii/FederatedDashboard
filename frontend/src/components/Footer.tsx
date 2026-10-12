import React from 'react';
import { FaShieldHalved, FaGraduationCap, FaBuildingColumns } from 'react-icons/fa6';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-zinc-800 bg-zinc-950 py-12 text-zinc-500 text-xs font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 pb-8 border-b border-zinc-900">
          
          <div className="flex items-center space-x-3">
            <div className="w-7 h-7 rounded bg-zinc-100 text-zinc-950 flex items-center justify-center font-bold">
              <FaShieldHalved className="w-3.5 h-3.5 text-zinc-950" />
            </div>
            <div>
              <div className="text-zinc-200 font-semibold tracking-tight uppercase">
                Federated Learning-Based Education System
              </div>
              <div className="text-[11px] text-zinc-400 font-medium">
                Privacy-Preserving Student Performance Analytics &amp; Personalized Learning
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-6 text-[11px] font-medium">
            <span className="flex items-center space-x-1.5">
              <FaGraduationCap className="w-3.5 h-3.5 text-zinc-400" />
              <span>Early student support</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <FaBuildingColumns className="w-3.5 h-3.5 text-zinc-400" />
              <span>Multi campus privacy</span>
            </span>
          </div>

        </div>

        <div className="pt-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-[11px] text-zinc-600 font-medium">
          <div>
            Built with PyTorch, FastAPI, React, and Tailwind CSS.
          </div>
          <div className="text-zinc-500">
            Runs locally on your computer with zero external data sharing
          </div>
        </div>
      </div>
    </footer>
  );
};
