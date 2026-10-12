import React from 'react';
import { motion } from 'framer-motion';
import { FaBuildingColumns, FaDatabase, FaLock, FaShieldHalved, FaCheck } from 'react-icons/fa6';

export interface CampusNode {
  id: string;
  title: string;
  role: string;
  students: number;
  profile: string;
  localData: string;
}

const DEFAULT_NODES: CampusNode[] = [
  {
    id: 'node_1',
    title: 'Faculty of Engineering and Technology',
    role: 'Campus 1 (STEM Node)',
    students: 450,
    profile: 'Frequent portal usage and heavy weekly assessments',
    localData: 'Saved on campus server. No outside read access.',
  },
  {
    id: 'node_2',
    title: 'Faculty of Social Sciences and Humanities',
    role: 'Campus 2 (Humanities Node)',
    students: 380,
    profile: 'Continuous essays and varied student age groups',
    localData: 'Saved on campus server. No outside read access.',
  },
  {
    id: 'node_3',
    title: 'School of Business and Management',
    role: 'Campus 3 (Management Node)',
    students: 520,
    profile: 'Flexible credits and varied prior schooling',
    localData: 'Saved on campus server. No outside read access.',
  }
];

export const FederatedNodes: React.FC = () => {
  const [nodes, setNodes] = React.useState<CampusNode[]>(DEFAULT_NODES);

  React.useEffect(() => {
    const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
    fetch(`${apiBase}/api/clients`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && Array.isArray(data.clients)) {
          setNodes(
            data.clients.map((c: any, idx: number) => ({
              id: c.id || `node_${idx + 1}`,
              title: c.name,
              role: `Campus ${idx + 1} (${c.type || 'Decentralized Node'})`,
              students: c.students || 400,
              profile: c.profile || 'Distributed campus partition',
              localData: c.status || 'Saved on campus server. No outside read access.',
            }))
          );
        }
      })
      .catch(() => {});
  }, []);

  return (
    <section className="py-16 border-b border-zinc-800 bg-zinc-950 font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10">
          <div>
            <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center space-x-2">
              <span className="pill-badge pill-purple">OBJECTIVES 1 &amp; 2</span>
              <span>Collaborative Distributed ML</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-100 mt-2">
              Distributed Educational Institutions &amp; Privacy Topology
            </h2>
            <p className="text-sm text-zinc-400 mt-2 max-w-2xl leading-relaxed">
              Enables collaborative machine learning across distributed educational faculties and student devices without sharing raw data. Model weights train locally on student cohorts while preserving absolute privacy and confidentiality.
            </p>
          </div>

          <div className="mt-4 md:mt-0 flex items-center space-x-2 border border-zinc-800 bg-zinc-900 px-3.5 py-1.5 text-xs font-medium text-zinc-300 rounded-full shadow-sm">
            <FaShieldHalved className="w-3.5 h-3.5 text-emerald-400" />
            <span>Zero Raw Record Transfer • Edge Isolation</span>
          </div>
        </div>

        {/* Client Nodes Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {nodes.map((node, idx) => {
            const pillColors = ['pill-purple', 'pill-blue', 'pill-amber'];
            const pillColor = pillColors[idx % pillColors.length];
            return (
              <motion.div
                key={node.id}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: idx * 0.1 }}
                className="card-squircle p-7 flex flex-col justify-between group"
              >
                <div>
                  <div className="flex items-center justify-between pb-4 border-b border-white/5 text-xs">
                    <span className={`pill-badge ${pillColor}`}>CAMPUS {idx + 1}</span>
                    <div className="flex items-center space-x-2">
                      <span className="pill-badge pill-dark text-[10px]">
                        ISOLATED
                      </span>
                      <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors">●●●</span>
                    </div>
                  </div>

                  <div className="mt-5 flex items-center space-x-3">
                    <div className="w-11 h-11 rounded-2xl bg-zinc-800/80 border border-white/5 flex items-center justify-center text-zinc-100 shadow-inner">
                      <FaBuildingColumns className="w-5 h-5 text-zinc-300" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-white leading-snug">
                        {node.title}
                      </h3>
                      <div className="text-xs text-zinc-400 mt-0.5 font-medium">
                        {node.students} students enrolled
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 space-y-4 text-xs">
                    <div>
                      <span className="text-zinc-500 block text-[10px] uppercase font-bold tracking-wider">
                        CAMPUS STUDENT PROFILE
                      </span>
                      <p className="text-zinc-300 mt-1.5 leading-relaxed">
                        {node.profile}
                      </p>
                    </div>

                    <div className="flex items-center space-x-2 text-zinc-400 pt-1">
                      <FaLock className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
                      <span className="text-xs">{node.localData}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-7 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-zinc-500 font-medium">
                  <span className="flex items-center space-x-1.5 text-zinc-400">
                    <FaDatabase className="w-3.5 h-3.5 text-zinc-400" />
                    <span>Local database</span>
                  </span>
                  <span className="text-zinc-300 font-semibold">Zero raw exports</span>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Comparison Cards (Clean Squircle Grid) */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          
          <div className="card-squircle p-7 flex flex-col justify-between group">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="pill-badge pill-rose">CENTRALIZED</span>
                <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors">●●●</span>
              </div>
              <div className="text-base font-bold text-white mt-1">
                Collects all student records into one server
              </div>
              <p className="text-xs text-zinc-400 mt-2 leading-relaxed">
                All campuses must upload student attendance, personal details, and assessment marks to a central cloud database. If that central repository is breached, all student privacy is compromised.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-white/5 flex items-center space-x-2 text-xs text-zinc-400">
              <span className="font-semibold text-rose-400">Privacy check:</span>
              <span>Fails cross-campus data protection regulations</span>
            </div>
          </div>

          <div className="card-squircle p-7 flex flex-col justify-between group bg-zinc-900/90 border-zinc-700/60 shadow-xl">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="pill-badge pill-emerald">FEDERATED LEARNING</span>
                <span className="text-zinc-600 group-hover:text-zinc-400 transition-colors">●●●</span>
              </div>
              <div className="text-base font-bold text-white mt-1">
                Trains locally and shares only model weight updates
              </div>
              <p className="text-xs text-zinc-300 mt-2 leading-relaxed">
                Student records never leave the local campus machine. Each department computes model updates independently, extracting numerical weights to be securely aggregated without viewing raw student logs.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-white/5 flex items-center space-x-2 text-xs text-emerald-300">
              <FaCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-semibold text-white">Privacy check:</span>
              <span>Student identities and files remain 100% on campus</span>
            </div>
          </div>

        </div>

      </div>
    </section>
  );
};
