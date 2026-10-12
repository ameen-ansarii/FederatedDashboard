import React from 'react';
import { motion } from 'framer-motion';
import { FaBolt, FaBuildingColumns, FaArrowRight, FaLock } from 'react-icons/fa6';

interface HeroProps {
  onExplore: (tab: string) => void;
}

export const Hero: React.FC<HeroProps> = ({ onExplore }) => {
  return (
    <section className="relative border-b border-zinc-800 py-16 sm:py-20 bg-zinc-950 font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Thesis Tag */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="inline-flex items-center space-x-2 px-3 py-1 border border-zinc-800 bg-zinc-900 text-xs font-medium text-zinc-300 mb-6 rounded"
        >
          <FaLock className="w-3 h-3 text-zinc-400" />
          <span>Federated Learning Research</span>
          <span className="text-zinc-600">/</span>
          <span className="text-zinc-400">Distributed Education &amp; Personalized Learning</span>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-zinc-100 max-w-4xl leading-[1.1]"
        >
          Federated Learning-Based Education System
        </motion.h1>

        {/* Description */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-6 text-base sm:text-lg text-zinc-400 max-w-3xl leading-relaxed"
        >
          Collaborative machine learning across distributed educational institutions without sharing raw student data. The system identifies students academically at risk, preserves privacy, provides Explainable AI (XAI) insights, and generates personalized learning recommendations.
        </motion.p>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-8 flex flex-wrap gap-3"
        >
          <button
            onClick={() => onExplore('predictor')}
            className="inline-flex items-center space-x-2 px-5 py-2.5 bg-zinc-100 text-zinc-950 font-semibold text-xs tracking-wider uppercase rounded hover:bg-white transition-colors"
          >
            <span>Personalized Learning &amp; Risk Predictor</span>
            <FaArrowRight className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => onExplore('batch')}
            className="inline-flex items-center space-x-2 px-5 py-2.5 border border-zinc-700 bg-zinc-900 text-zinc-200 font-semibold text-xs tracking-wider uppercase rounded hover:border-zinc-400 hover:text-white transition-colors"
          >
            <span>Cohort Batch Upload (CSV)</span>
          </button>

          <button
            onClick={() => onExplore('training')}
            className="inline-flex items-center space-x-2 px-5 py-2.5 border border-zinc-800 bg-zinc-900/50 text-zinc-400 font-semibold text-xs tracking-wider uppercase rounded hover:border-zinc-700 hover:text-zinc-200 transition-colors"
          >
            <FaBolt className="w-3.5 h-3.5 text-zinc-500" />
            <span>Federated Training (FedProx / FedAvg)</span>
          </button>

          <button
            onClick={() => onExplore('benchmarks')}
            className="inline-flex items-center space-x-2 px-5 py-2.5 border border-zinc-800 bg-zinc-900/50 text-zinc-400 font-semibold text-xs tracking-wider uppercase rounded hover:border-zinc-700 hover:text-zinc-200 transition-colors"
          >
            <FaBuildingColumns className="w-3.5 h-3.5 text-zinc-500" />
            <span>System Evaluation</span>
          </button>
        </motion.div>

        {/* Key Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-14 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 pt-4"
        >
          <div className="card-squircle p-6 flex flex-col justify-between group">
            <div className="flex items-center justify-between">
              <span className="pill-badge pill-purple">Architecture</span>
              <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors text-xs">●●●</span>
            </div>
            <div className="mt-6">
              <h3 className="text-xl font-bold tracking-tight text-white">4.69 KB</h3>
              <p className="text-xs text-zinc-400 mt-2 leading-relaxed">Compact PyTorch neural architecture enabling zero-leakage local gradient training.</p>
            </div>
            <div className="mt-4 pt-3 border-t border-white/5 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
              MODEL WEIGHTS
            </div>
          </div>

          <div className="card-squircle p-6 flex flex-col justify-between group">
            <div className="flex items-center justify-between">
              <span className="pill-badge pill-emerald">Privacy</span>
              <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors text-xs">●●●</span>
            </div>
            <div className="mt-6">
              <h3 className="text-xl font-bold tracking-tight text-white">0 records</h3>
              <p className="text-xs text-zinc-400 mt-2 leading-relaxed">Zero raw student logs transmitted outside local campus node boundaries.</p>
            </div>
            <div className="mt-4 pt-3 border-t border-white/5 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
              PRIVACY PRESERVATION
            </div>
          </div>

          <div className="card-squircle p-6 flex flex-col justify-between group">
            <div className="flex items-center justify-between">
              <span className="pill-badge pill-amber">Comm. Efficiency</span>
              <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors text-xs">●●●</span>
            </div>
            <div className="mt-6">
              <h3 className="text-xl font-bold tracking-tight text-white">99.85%</h3>
              <p className="text-xs text-zinc-400 mt-2 leading-relaxed">Bandwidth reduction compared to centralized pooling of student records.</p>
            </div>
            <div className="mt-4 pt-3 border-t border-white/5 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
              COMMUNICATION EFFICIENCY
            </div>
          </div>

          <div className="card-squircle p-6 flex flex-col justify-between group">
            <div className="flex items-center justify-between">
              <span className="pill-badge pill-rose">Personalization</span>
              <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors text-xs">●●●</span>
            </div>
            <div className="mt-6">
              <h3 className="text-xl font-bold tracking-tight text-white">Adaptive Plans</h3>
              <p className="text-xs text-zinc-400 mt-2 leading-relaxed">Tailored study pacing, curated resources, and milestone guidance.</p>
            </div>
            <div className="mt-4 pt-3 border-t border-white/5 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
              PERSONALIZED LEARNING
            </div>
          </div>
        </motion.div>

      </div>
    </section>
  );
};
