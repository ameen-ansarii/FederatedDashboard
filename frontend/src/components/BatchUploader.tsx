import React, { useState, useRef } from 'react';
import { 
  FaCloudArrowUp, 
  FaDownload, 
  FaShieldHalved, 
  FaLock
} from 'react-icons/fa6';

export interface BatchStudentResult {
  student_id: string;
  vle_clicks: number;
  assessment_score: number;
  studied_credits: number;
  num_attempts: number;
  probability: number;
  percentage: string;
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW';
  primary_driver: string;
  recommended_action: string;
}

export interface BatchResponse {
  status: string;
  filename: string;
  total_students: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  average_cohort_risk: string;
  privacy_log: {
    mode: string;
    bytes_sent_external: number;
    data_residency: string;
    weights_used: string;
  };
  cohort: BatchStudentResult[];
}

export const BatchUploader: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [batchData, setBatchData] = useState<BatchResponse | null>(null);
  const [filterRisk, setFilterRisk] = useState<'ALL' | 'HIGH' | 'MEDIUM' | 'LOW'>('ALL');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

  // Download real sample OULAD CSV
  const handleDownloadSample = async () => {
    try {
      const res = await fetch(`${apiBase}/api/sample-csv`);
      if (res.ok) {
        const data = await res.json();
        const blob = new Blob([data.content], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename || 'sample_oulad_cohort.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch {
      setErrorMsg('Could not fetch sample CSV file from backend.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setErrorMsg(null);
    }
  };

  const handleUploadAndRun = async () => {
    if (!file) {
      setErrorMsg('Please select a student CSV file first.');
      return;
    }

    setIsProcessing(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${apiBase}/api/upload-batch`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data: BatchResponse = await res.json();
        setBatchData(data);
      } else {
        const err = await res.json().catch(() => ({ detail: 'Upload error' }));
        setErrorMsg(err.detail || 'Failed to process student CSV.');
      }
    } catch {
      setErrorMsg('Network error: Unable to connect to PyTorch backend on port 8000.');
    } finally {
      setIsProcessing(false);
    }
  };

  const filteredCohort = batchData?.cohort.filter((stu) => {
    if (filterRisk === 'ALL') return true;
    return stu.risk_level === filterRisk;
  }) || [];

  return (
    <section id="batch-upload-section" className="py-16 border-b border-zinc-800 bg-zinc-950 font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10">
          <div>
            <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center space-x-2">
              <span className="pill-badge pill-purple">DATASET EVALUATION (BATCH)</span>
              <span>Upload CSV Cohort</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-100 mt-2">
              Batch Student Risk Analysis
            </h2>
            <p className="text-sm text-zinc-400 mt-2 max-w-2xl leading-relaxed">
              Upload an actual campus spreadsheet (`.csv`). The on-device PyTorch model evaluates all students in your file at once and displays the distribution metrics and triage list directly below.
            </p>
          </div>

          <div className="mt-4 md:mt-0 flex items-center space-x-2">
            <button
              onClick={handleDownloadSample}
              className="inline-flex items-center space-x-2 px-3 py-1.5 border border-zinc-700 bg-zinc-900 text-xs font-medium text-zinc-200 rounded hover:bg-zinc-800 transition-colors"
            >
              <FaDownload className="w-3 h-3 text-zinc-400" />
              <span>Download 50-Student OULAD Sample</span>
            </button>
          </div>
        </div>

        {/* Upload Zone */}
        <div className="card-squircle p-8">
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-white/10 hover:border-white/20 rounded-3xl p-10 transition-colors text-center bg-zinc-950/40">
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept=".csv" 
              className="hidden" 
            />
            
            <div className="w-14 h-14 rounded-2xl bg-zinc-800/80 border border-white/5 flex items-center justify-center text-white mb-4 shadow-inner">
              <FaCloudArrowUp className="w-6 h-6" />
            </div>

            <h3 className="text-base font-bold text-white">
              {file ? file.name : 'Select or drop campus student CSV'}
            </h3>
            <p className="text-xs text-zinc-400 mt-1.5 max-w-md leading-relaxed">
              Supports standard OULAD headers (`total_vle_clicks`, `avg_assessment_score`, `studied_credits`, `num_of_prev_attempts`) or custom university student tables.
            </p>

            <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2.5 rounded-full border border-white/10 bg-zinc-800/80 hover:bg-zinc-700 text-xs font-semibold text-white transition-colors"
              >
                Choose CSV File
              </button>

              <button
                type="button"
                onClick={handleUploadAndRun}
                disabled={!file || isProcessing}
                className={`px-6 py-2.5 text-xs font-bold rounded-full uppercase tracking-wider transition-colors ${
                  !file || isProcessing
                    ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                    : 'bg-white text-zinc-950 hover:bg-zinc-100 shadow-md'
                }`}
              >
                {isProcessing ? 'Evaluating Neural Network...' : 'Analyze Cohort Risk'}
              </button>
            </div>

            {errorMsg && (
              <div className="mt-4 text-xs text-rose-400 font-semibold">
                {errorMsg}
              </div>
            )}
          </div>
        </div>

        {/* Results Area */}
        {batchData && (
          <div className="mt-8 space-y-6">
            {/* Reference 2 & Reference 1 Cohort Synthesis Widgets */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              
              {/* Left: Concentric Arc Rings Widget (Reference 2) */}
              <div className="lg:col-span-6 card-squircle p-7 relative overflow-hidden bg-zinc-900/90 shadow-2xl flex flex-col justify-between">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="flex-1 space-y-4">
                    <div className="flex items-center space-x-2 text-zinc-400 text-sm font-semibold">
                      <span>Cohort Distribution</span>
                      <span className="text-zinc-600">/</span>
                      <span className="text-zinc-300">Live Breakdown</span>
                    </div>

                    <div className="space-y-3 pt-1">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold text-xs">
                          ♥
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-base font-bold text-white">{batchData.high_risk_count} critical</span>
                            <span className="text-xs font-bold text-rose-400">
                              {((batchData.high_risk_count / batchData.total_students) * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-[11px] text-zinc-400 font-medium">Immediate counseling needed</div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-xs">
                          ◐
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-base font-bold text-white">{batchData.medium_risk_count} warning</span>
                            <span className="text-xs font-bold text-amber-400">
                              {((batchData.medium_risk_count / batchData.total_students) * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-[11px] text-zinc-400 font-medium">Advisor check-in recommended</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Concentric Arc Ring Animation */}
                  <div className="relative w-44 h-44 flex items-center justify-center shrink-0 self-center sm:self-auto">
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 160 160">
                      <circle cx="80" cy="80" r="66" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />
                      <circle cx="80" cy="80" r="50" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />
                      <circle cx="80" cy="80" r="34" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />

                      {/* Ring 1: Low Risk (Green) */}
                      <circle
                        cx="80"
                        cy="80"
                        r="66"
                        stroke="#4ade80"
                        strokeWidth="10"
                        fill="transparent"
                        strokeDasharray="414"
                        strokeDashoffset={414 - (batchData.low_risk_count / batchData.total_students) * 310}
                        className="animate-ring-arc"
                      />

                      {/* Ring 2: Medium Risk (Amber) */}
                      <circle
                        cx="80"
                        cy="80"
                        r="50"
                        stroke="#fbbf24"
                        strokeWidth="10"
                        fill="transparent"
                        strokeDasharray="314"
                        strokeDashoffset={314 - (batchData.medium_risk_count / batchData.total_students) * 235}
                        className="animate-ring-arc"
                      />

                      {/* Ring 3: High Risk (Rose) */}
                      <circle
                        cx="80"
                        cy="80"
                        r="34"
                        stroke="#f43f5e"
                        strokeWidth="10"
                        fill="transparent"
                        strokeDasharray="213"
                        strokeDashoffset={213 - (batchData.high_risk_count / batchData.total_students) * 160}
                        className="animate-ring-arc"
                      />
                    </svg>

                    <div className="absolute w-20 h-20 rounded-full bg-white text-zinc-950 flex flex-col items-center justify-center shadow-2xl border-4 border-zinc-900">
                      <span className="text-xl font-extrabold tracking-tight leading-none">
                        {batchData.average_cohort_risk}
                      </span>
                      <span className="text-[9px] font-extrabold text-zinc-600 tracking-wider mt-0.5">
                        AVG RISK
                      </span>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-zinc-400">
                  <span className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    <span>{batchData.low_risk_count} Safe</span>
                  </span>
                  <span className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-400" />
                    <span>{batchData.medium_risk_count} Moderate</span>
                  </span>
                  <span className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-400" />
                    <span>{batchData.high_risk_count} Critical</span>
                  </span>
                </div>
              </div>

              {/* Right: Athletic Score Style Cohort Metric Matrix (Reference 1) */}
              <div className="lg:col-span-6 card-squircle p-7 bg-zinc-900/90 shadow-2xl flex flex-col justify-between">
                <div className="flex flex-col sm:flex-row gap-6">
                  
                  {/* Big Score + Dot Matrix */}
                  <div className="sm:w-5/12 flex flex-col justify-between border-b sm:border-b-0 sm:border-r border-white/5 pb-4 sm:pb-0 sm:pr-6">
                    <div>
                      <div className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider">
                        COHORT INTEGRITY
                      </div>
                      <div className="text-5xl font-extrabold text-white mt-2 tracking-tight">
                        {Math.round((batchData.low_risk_count / batchData.total_students) * 100)}
                      </div>
                      <div className="text-[11px] text-zinc-400 mt-1 font-medium">
                        Safety index (/100)
                      </div>
                    </div>

                    <div className="mt-5">
                      <div className="grid grid-cols-7 gap-1.5 w-fit p-2.5 rounded-2xl bg-zinc-950/80 border border-white/5 shadow-inner">
                        {Array.from({ length: 28 }).map((_, i) => {
                          const activeCount = Math.round((batchData.low_risk_count / batchData.total_students) * 28);
                          const isActive = i < activeCount;
                          return (
                            <div
                              key={i}
                              className={`matrix-dot ${isActive ? 'matrix-dot-emerald' : 'matrix-dot-rose'}`}
                            />
                          );
                        })}
                      </div>
                      <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider mt-2">
                        Dataset Partition Integrity
                      </div>
                    </div>
                  </div>

                  {/* Right: Rows with Icons (Reference 1) */}
                  <div className="sm:w-7/12 flex flex-col justify-between divide-y divide-white/5 space-y-2.5">
                    
                    <div className="flex items-center justify-between pb-2.5">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold text-xs">
                          ◈
                        </div>
                        <div>
                          <span className="text-xs font-bold text-white uppercase tracking-wider block">STUDENTS</span>
                          <span className="text-[10px] text-zinc-400">Total Evaluated</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-sm font-bold text-white">{batchData.total_students}</span>
                        <span className="text-[10px] text-zinc-400 block">records</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between py-2.5">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold text-xs">
                          ♥
                        </div>
                        <div>
                          <span className="text-xs font-bold text-white uppercase tracking-wider block">HIGH RISK</span>
                          <span className="text-[10px] text-zinc-400">&ge; 70% Probability</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-sm font-bold text-rose-400">{batchData.high_risk_count}</span>
                        <span className="text-[10px] text-zinc-400 block">students</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between py-2.5">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-xs">
                          ◐
                        </div>
                        <div>
                          <span className="text-xs font-bold text-white uppercase tracking-wider block">MODERATE</span>
                          <span className="text-[10px] text-zinc-400">40% &ndash; 70% Range</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-sm font-bold text-amber-400">{batchData.medium_risk_count}</span>
                        <span className="text-[10px] text-zinc-400 block">students</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-2.5">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
                          ◎
                        </div>
                        <div>
                          <span className="text-xs font-bold text-white uppercase tracking-wider block">SAFE</span>
                          <span className="text-[10px] text-zinc-400">&lt; 40% On Track</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-sm font-bold text-emerald-400">{batchData.low_risk_count}</span>
                        <span className="text-[10px] text-zinc-400 block">students</span>
                      </div>
                    </div>

                  </div>

                </div>

                <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-zinc-400">
                  <span>Privacy mode: {batchData.privacy_log.mode}</span>
                  <span className="text-emerald-400 font-semibold">{batchData.privacy_log.data_residency}</span>
                </div>
              </div>

            </div>

            {/* Privacy Verification Banner */}
            <div className="p-4 border border-zinc-800 bg-zinc-900/80 rounded flex flex-col sm:flex-row sm:items-center justify-between text-xs text-zinc-400 gap-2">
              <div className="flex items-center space-x-2">
                <FaShieldHalved className="w-4 h-4 text-emerald-400 shrink-0" />
                <span className="text-zinc-200 font-semibold">{batchData.privacy_log.mode}:</span>
                <span>{batchData.privacy_log.data_residency}</span>
              </div>
              <div className="flex items-center space-x-2 font-mono text-[11px] text-zinc-300">
                <FaLock className="w-3 h-3 text-zinc-400" />
                <span>External egress: {batchData.privacy_log.bytes_sent_external} bytes</span>
              </div>
            </div>

            {/* Triage Table */}
            <div className="border border-zinc-800 bg-zinc-900/60 rounded overflow-hidden">
              <div className="p-4 border-b border-zinc-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <div className="font-semibold text-zinc-200">
                  Student Risk Triage ({filteredCohort.length} students displayed)
                </div>
                
                <div className="flex items-center space-x-1.5">
                  <span className="text-zinc-500 text-[11px] mr-1">Filter Risk:</span>
                  {(['ALL', 'HIGH', 'MEDIUM', 'LOW'] as const).map((r) => (
                    <button
                      key={r}
                      onClick={() => setFilterRisk(r)}
                      className={`px-2.5 py-1 text-[11px] font-medium rounded transition-colors ${
                        filterRisk === r
                          ? 'bg-zinc-100 text-zinc-950 font-bold'
                          : 'border border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-zinc-800 bg-zinc-950 text-zinc-400 text-[11px] font-semibold">
                      <th className="py-3 px-4">STUDENT ID</th>
                      <th className="py-3 px-4">MARKS</th>
                      <th className="py-3 px-4">PORTAL CLICKS</th>
                      <th className="py-3 px-4">CREDITS</th>
                      <th className="py-3 px-4">DROPOUT PROBABILITY</th>
                      <th className="py-3 px-4">RISK STATUS</th>
                      <th className="py-3 px-4">PRIMARY FACTOR</th>
                      <th className="py-3 px-4">ACTION NEEDED</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
                    {filteredCohort.map((stu, i) => (
                      <tr key={i} className="hover:bg-zinc-800/30 transition-colors">
                        <td className="py-3 px-4 font-mono font-medium text-zinc-100">
                          {stu.student_id}
                        </td>
                        <td className="py-3 px-4 font-medium text-zinc-200">
                          {stu.assessment_score}%
                        </td>
                        <td className="py-3 px-4 text-zinc-300">
                          {stu.vle_clicks}
                        </td>
                        <td className="py-3 px-4 text-zinc-400">
                          {stu.studied_credits}
                        </td>
                        <td className="py-3 px-4 font-bold text-zinc-100">
                          {stu.percentage}
                        </td>
                        <td className="py-3 px-4">
                          <span className={`inline-flex px-2 py-0.5 text-[10px] font-bold rounded uppercase tracking-wider ${
                            stu.risk_level === 'HIGH'
                              ? 'bg-red-500/10 text-red-400 border border-red-500/30'
                              : stu.risk_level === 'MEDIUM'
                              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                          }`}>
                            {stu.risk_level}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-zinc-300 text-[11px]">
                          {stu.primary_driver}
                        </td>
                        <td className="py-3 px-4 text-zinc-400 text-[11px]">
                          {stu.recommended_action}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

      </div>
    </section>
  );
};
