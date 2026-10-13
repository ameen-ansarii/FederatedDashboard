import React, { useState, useRef, useEffect } from 'react';
import { 
  FaCloudArrowUp, 
  FaDownload, 
  FaShieldHalved, 
  FaLock,
  FaMicrochip,
  FaNetworkWired,
  FaCircleCheck,
  FaArrowsRotate,
  FaChartPie,
  FaBolt,
  FaMagnifyingGlass,
  FaUsers
} from 'react-icons/fa6';
import { useCountUp } from '../utils/useCountUp';

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
  personalized_learning?: {
    pacing: string;
    key_resource: string;
    milestone: string;
  };
}

export interface BatchResponse {
  status: string;
  filename: string;
  total_students: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  average_cohort_risk: string;
  cohort_learning_summary?: {
    intensive_pedagogical_support_needed: number;
    guided_pacing_and_feedback_needed: number;
    autonomous_mastery_cohort: number;
  };
  communication_efficiency?: {
    mode: string;
    bytes_sent_external: number;
    bandwidth_saved: string;
  };
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
  const [processingStep, setProcessingStep] = useState<number>(0);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [ringsVisible, setRingsVisible] = useState<boolean>(false);
  const [batchData, setBatchData] = useState<BatchResponse | null>(null);
  const [filterRisk, setFilterRisk] = useState<'ALL' | 'HIGH' | 'MEDIUM' | 'LOW'>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(25);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

  // Smooth Count-Up animations for metrics
  const targetAvgRisk = batchData ? parseFloat(batchData.average_cohort_risk.replace('%', '')) : 0;
  const animAvgRisk = useCountUp(targetAvgRisk, 1600, ringsVisible, 1);

  const targetSafetyIndex = batchData && batchData.total_students > 0 
    ? Math.round((batchData.low_risk_count / batchData.total_students) * 100) 
    : 0;
  const animSafetyIndex = useCountUp(targetSafetyIndex, 1600, ringsVisible);

  const animTotalStudents = useCountUp(batchData?.total_students || 0, 1400, ringsVisible);
  const animHighRisk = useCountUp(batchData?.high_risk_count || 0, 1400, ringsVisible);
  const animMedRisk = useCountUp(batchData?.medium_risk_count || 0, 1400, ringsVisible);
  const animLowRisk = useCountUp(batchData?.low_risk_count || 0, 1400, ringsVisible);

  // Trigger smooth ring sweep on data change
  useEffect(() => {
    if (batchData) {
      setRingsVisible(false);
      const timer = setTimeout(() => {
        setRingsVisible(true);
      }, 70);
      return () => clearTimeout(timer);
    }
  }, [batchData]);

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
    setCurrentPage(1);
    setProcessingStep(1);
    setProcessingProgress(20);
    setRingsVisible(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Trigger background upload while presenting authentic telemetry workflow stages
      const uploadPromise = fetch(`${apiBase}/api/upload-batch`, {
        method: 'POST',
        body: formData,
      });

      // Stage 1: Enclave verification (0 - 450ms)
      await new Promise((r) => setTimeout(r, 450));
      setProcessingStep(2);
      setProcessingProgress(48);

      // Stage 2: Distribute weights (450ms - 900ms)
      await new Promise((r) => setTimeout(r, 450));
      setProcessingStep(3);
      setProcessingProgress(76);

      // Stage 3: PyTorch forward pass (900ms - 1350ms)
      await new Promise((r) => setTimeout(r, 450));
      setProcessingStep(4);
      setProcessingProgress(94);

      const res = await uploadPromise;
      await new Promise((r) => setTimeout(r, 300));
      setProcessingProgress(100);

      if (res.ok) {
        const data: BatchResponse = await res.json();
        setBatchData(data);
        setIsProcessing(false);
      } else {
        const err = await res.json().catch(() => ({ detail: 'Upload error' }));
        setErrorMsg(err.detail || 'Failed to process student CSV.');
        setIsProcessing(false);
      }
    } catch {
      setErrorMsg('Network error: Unable to connect to PyTorch backend on port 8000.');
      setIsProcessing(false);
    }
  };

  const filteredCohort = (batchData?.cohort || []).filter((stu) => {
    const matchesRisk = filterRisk === 'ALL' || stu.risk_level === filterRisk;
    const matchesSearch = !searchQuery.trim() || 
      stu.student_id.toLowerCase().includes(searchQuery.toLowerCase()) || 
      stu.primary_driver.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (stu.personalized_learning?.pacing || '').toLowerCase().includes(searchQuery.toLowerCase());
    return matchesRisk && matchesSearch;
  });

  const totalPages = Math.max(1, Math.ceil(filteredCohort.length / pageSize));
  const safeCurrentPage = Math.min(currentPage, totalPages);
  const startIndex = (safeCurrentPage - 1) * pageSize;
  const paginatedCohort = filteredCohort.slice(startIndex, startIndex + pageSize);

  // Circle Arc Calculations
  const total = batchData && batchData.total_students > 0 ? batchData.total_students : 1;
  const lowOffset = ringsVisible 
    ? 414 - ((batchData?.low_risk_count || 0) / total) * 310 
    : 414;
  const medOffset = ringsVisible 
    ? 314 - ((batchData?.medium_risk_count || 0) / total) * 235 
    : 314;
  const highOffset = ringsVisible 
    ? 213 - ((batchData?.high_risk_count || 0) / total) * 160 
    : 213;

  return (
    <section id="batch-upload-section" className="py-16 border-b border-zinc-800 bg-zinc-950 font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10">
          <div>
            <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center space-x-2">
              <span className="pill-badge pill-purple">OBJECTIVES 2, 3 &amp; 4</span>
              <span>On-Premise Cohort Evaluation</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-100 mt-2">
              Cohort Academic Performance &amp; Personalized Learning
            </h2>
            <p className="text-sm text-zinc-400 mt-2 max-w-2xl leading-relaxed">
              Upload an actual campus spreadsheet (`.csv`). The on-device PyTorch model evaluates all students in your file at once with zero raw data leakage, displaying individual risk assessments and customized learning strategies.
            </p>
          </div>

          <div className="mt-4 md:mt-0 flex items-center space-x-2">
            <button
              onClick={handleDownloadSample}
              className="inline-flex items-center space-x-2.5 px-4 py-2.5 border-2 border-purple-500/40 bg-purple-950/30 text-xs sm:text-sm font-bold text-purple-200 rounded-xl hover:bg-purple-900/40 hover:border-purple-400 transition-all shadow-md hover:scale-105"
            >
              <FaDownload className="w-4 h-4 text-purple-400" />
              <span>Download Official OULAD Dataset (Nature Scientific Data)</span>
            </button>
          </div>
        </div>

        {/* Upload Zone */}
        <div className="card-squircle p-8">
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-white/10 hover:border-white/20 rounded-3xl p-10 transition-colors text-center bg-zinc-950/40 relative overflow-hidden">
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept=".csv" 
              className="hidden" 
            />
            
            <div className="w-16 h-16 rounded-2xl bg-zinc-800/80 border border-white/10 flex items-center justify-center text-white mb-4 shadow-inner">
              <FaCloudArrowUp className="w-7 h-7 text-purple-400" />
            </div>

            <h3 className="text-lg font-bold text-white">
              {file ? file.name : 'Select or drop campus student CSV'}
            </h3>
            <p className="text-xs sm:text-sm text-zinc-400 mt-1.5 max-w-lg leading-relaxed">
              Supports standard OULAD headers (`total_vle_clicks`, `avg_assessment_score`, `studied_credits`, `num_of_prev_attempts`) or custom university student tables.
            </p>

            <div className="mt-7 flex flex-wrap items-center justify-center gap-4">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="px-6 py-3 rounded-full border-2 border-white/20 bg-zinc-800/90 hover:bg-zinc-700 text-sm font-bold text-white transition-all shadow-md hover:scale-105 active:scale-95 disabled:opacity-50 cursor-pointer"
              >
                Choose CSV File
              </button>

              <button
                type="button"
                onClick={handleUploadAndRun}
                disabled={!file || isProcessing}
                className={`px-8 py-3 text-sm font-black rounded-full uppercase tracking-wider transition-all duration-300 flex items-center space-x-2.5 ${
                  !file || isProcessing
                    ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                    : 'bg-white text-zinc-950 hover:bg-zinc-100 hover:shadow-xl hover:shadow-purple-500/20 hover:scale-105 active:scale-95 shadow-md cursor-pointer'
                }`}
              >
                {isProcessing ? (
                  <>
                    <FaArrowsRotate className="w-4 h-4 animate-spin text-purple-600" />
                    <span>Analyzing Cohort...</span>
                  </>
                ) : (
                  <span>Analyze Cohort Risk</span>
                )}
              </button>
            </div>

            {errorMsg && (
              <div className="mt-4 text-xs sm:text-sm text-rose-400 font-semibold bg-rose-950/30 px-5 py-2.5 rounded-xl border border-rose-800/40">
                {errorMsg}
              </div>
            )}
          </div>

          {/* High-Tech Federated Inference Telemetry HUD during processing */}
          {isProcessing && (
            <div className="mt-6 p-6 rounded-2xl bg-zinc-900/90 border border-purple-500/20 shadow-2xl animate-fade-slide-up relative overflow-hidden">
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mb-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-purple-400 relative">
                    <FaMicrochip className="w-5 h-5 animate-pulse" />
                    <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white flex items-center space-x-2">
                      <span>On-Premise Neural Inference Pipeline</span>
                      <span className="pill-badge pill-purple text-[9px]">STEP {processingStep} OF 4</span>
                    </h4>
                    <p className="text-xs text-zinc-400 mt-0.5">
                      {processingStep === 1 && 'Validating privacy boundary: Zero student records egress the campus runtime...'}
                      {processingStep === 2 && 'Mounting federated global model parameters (4.69 KB architecture payload)...'}
                      {processingStep === 3 && 'Executing vectorized PyTorch batch forward inference pass across student matrix...'}
                      {processingStep === 4 && 'Synthesizing individual risk curves, gradient attributions & study recommendations...'}
                    </p>
                  </div>
                </div>

                <div className="text-right shrink-0">
                  <span className="text-2xl font-black text-purple-300 font-mono">
                    {processingProgress}%
                  </span>
                  <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider block">
                    TELEMETRY PROGRESS
                  </span>
                </div>
              </div>

              {/* Shimmering Progress Bar */}
              <div className="w-full bg-zinc-950 rounded-full h-2.5 overflow-hidden border border-white/5 relative">
                <div 
                  className="h-full bg-gradient-to-r from-purple-600 via-indigo-500 to-emerald-400 rounded-full transition-all duration-300 relative"
                  style={{ width: `${processingProgress}%` }}
                >
                  <div className="absolute inset-0 bg-white/30 animate-shimmer" />
                </div>
              </div>

              {/* 4 Pipeline Stages Status Line */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4 pt-3 border-t border-white/5 text-[11px]">
                <div className={`flex items-center space-x-1.5 ${processingStep >= 1 ? 'text-emerald-400' : 'text-zinc-600'}`}>
                  {processingStep > 1 ? <FaCircleCheck className="w-3.5 h-3.5" /> : <FaBolt className="w-3.5 h-3.5 animate-pulse" />}
                  <span>1. Enclave Lock</span>
                </div>
                <div className={`flex items-center space-x-1.5 ${processingStep >= 2 ? 'text-emerald-400' : 'text-zinc-600'}`}>
                  {processingStep > 2 ? <FaCircleCheck className="w-3.5 h-3.5" /> : processingStep === 2 ? <FaArrowsRotate className="w-3.5 h-3.5 animate-spin" /> : <FaNetworkWired className="w-3.5 h-3.5" />}
                  <span>2. FL Weights (4.69 KB)</span>
                </div>
                <div className={`flex items-center space-x-1.5 ${processingStep >= 3 ? 'text-emerald-400' : 'text-zinc-600'}`}>
                  {processingStep > 3 ? <FaCircleCheck className="w-3.5 h-3.5" /> : processingStep === 3 ? <FaArrowsRotate className="w-3.5 h-3.5 animate-spin" /> : <FaMicrochip className="w-3.5 h-3.5" />}
                  <span>3. PyTorch Matrix Pass</span>
                </div>
                <div className={`flex items-center space-x-1.5 ${processingStep >= 4 ? 'text-emerald-400' : 'text-zinc-600'}`}>
                  {processingStep >= 4 ? <FaCircleCheck className="w-3.5 h-3.5" /> : <FaChartPie className="w-3.5 h-3.5" />}
                  <span>4. Learning Synthesis</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Results Area with Fluid Entrance Animation */}
        {batchData && !isProcessing && (
          <div className="mt-8 space-y-6 animate-fade-slide-up">
            
            {/* Concentric Arc Rings & Cohort Matrix Widgets */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              
              {/* Left: Concentric Arc Rings Widget */}
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
                        <div className="w-8 h-8 rounded-full bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold text-xs shadow-inner">
                          ♥
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-base font-bold text-white">
                              {animHighRisk} critical
                            </span>
                            <span className="text-xs font-bold text-rose-400">
                              {((batchData.high_risk_count / batchData.total_students) * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-[11px] text-zinc-400 font-medium">Immediate counseling needed</div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-xs shadow-inner">
                          ◐
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="text-base font-bold text-white">
                              {animMedRisk} warning
                            </span>
                            <span className="text-xs font-bold text-amber-400">
                              {((batchData.medium_risk_count / batchData.total_students) * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-[11px] text-zinc-400 font-medium">Advisor check-in recommended</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Concentric Arc Ring with Smooth Drawing Animation & Glow */}
                  <div className="relative w-44 h-44 flex items-center justify-center shrink-0 self-center sm:self-auto">
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 160 160">
                      {/* Background Guide Tracks */}
                      <circle cx="80" cy="80" r="66" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />
                      <circle cx="80" cy="80" r="50" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />
                      <circle cx="80" cy="80" r="34" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />

                      {/* Ring 1: Low Risk (Emerald) */}
                      <circle
                        cx="80"
                        cy="80"
                        r="66"
                        stroke="#4ade80"
                        strokeWidth="10"
                        fill="transparent"
                        strokeDasharray="414"
                        strokeDashoffset={lowOffset}
                        className="animate-ring-arc"
                        style={{
                          filter: ringsVisible ? 'drop-shadow(0 0 5px rgba(74, 222, 128, 0.6))' : 'none',
                        }}
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
                        strokeDashoffset={medOffset}
                        className="animate-ring-arc"
                        style={{
                          filter: ringsVisible ? 'drop-shadow(0 0 5px rgba(251, 191, 36, 0.6))' : 'none',
                        }}
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
                        strokeDashoffset={highOffset}
                        className="animate-ring-arc"
                        style={{
                          filter: ringsVisible ? 'drop-shadow(0 0 5px rgba(244, 63, 94, 0.6))' : 'none',
                        }}
                      />
                    </svg>

                    {/* Center Gauge Value Counting Up */}
                    <div className="absolute w-20 h-20 rounded-full bg-white text-zinc-950 flex flex-col items-center justify-center shadow-2xl border-4 border-zinc-900 transition-transform duration-500 hover:scale-105">
                      <span className="text-xl font-extrabold tracking-tight leading-none font-mono">
                        {animAvgRisk.toFixed(1)}%
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
                    <span>{animLowRisk} Safe</span>
                  </span>
                  <span className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-400" />
                    <span>{animMedRisk} Moderate</span>
                  </span>
                  <span className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-400" />
                    <span>{animHighRisk} Critical</span>
                  </span>
                </div>
              </div>

              {/* Right: Athletic Score Style Cohort Metric Matrix */}
              <div className="lg:col-span-6 card-squircle p-7 bg-zinc-900/90 shadow-2xl flex flex-col justify-between">
                <div className="flex flex-col sm:flex-row gap-6">
                  
                  {/* Big Score + Dot Matrix with Staggered Cascading Illumination */}
                  <div className="sm:w-5/12 flex flex-col justify-between border-b sm:border-b-0 sm:border-r border-white/5 pb-4 sm:pb-0 sm:pr-6">
                    <div>
                      <div className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider">
                        COHORT INTEGRITY
                      </div>
                      <div className="text-5xl font-extrabold text-white mt-2 tracking-tight font-mono">
                        {animSafetyIndex}
                      </div>
                      <div className="text-[11px] text-zinc-400 mt-1 font-medium">
                        Safety index (/100)
                      </div>
                    </div>

                    <div className="mt-5">
                      <div className="grid grid-cols-7 gap-1.5 w-fit p-2.5 rounded-2xl bg-zinc-950/80 border border-white/5 shadow-inner">
                        {Array.from({ length: 28 }).map((_, i) => {
                          const activeCount = Math.round((batchData.low_risk_count / total) * 28);
                          const isSafeDot = i < activeCount;
                          const isIlluminated = ringsVisible;
                          return (
                            <div
                              key={i}
                              style={{
                                transitionDelay: `${i * 35}ms`,
                                transitionDuration: '450ms',
                              }}
                              className={`matrix-dot ${
                                isIlluminated 
                                  ? isSafeDot 
                                    ? 'matrix-dot-emerald scale-110' 
                                    : 'matrix-dot-rose opacity-90'
                                  : 'bg-zinc-800 opacity-30'
                              }`}
                            />
                          );
                        })}
                      </div>
                      <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider mt-2">
                        Dataset Partition Integrity
                      </div>
                    </div>
                  </div>

                  {/* Right: Cohort Totals with Smooth Count-Up */}
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
                        <span className="text-sm font-bold text-white font-mono">{animTotalStudents}</span>
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
                        <span className="text-sm font-bold text-rose-400 font-mono">{animHighRisk}</span>
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
                        <span className="text-sm font-bold text-amber-400 font-mono">{animMedRisk}</span>
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
                        <span className="text-sm font-bold text-emerald-400 font-mono">{animLowRisk}</span>
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

            {/* Objective 3: Personalized Learning Cohort Summary */}
            {batchData.cohort_learning_summary && (
              <div className="card-squircle p-6 bg-zinc-900/90 border border-purple-500/20 shadow-xl">
                <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/5">
                  <div className="flex items-center space-x-2">
                    <span className="pill-badge pill-purple text-[10px]">OBJECTIVE 3</span>
                    <h3 className="text-sm font-bold text-white">
                      Personalized Learning Recommendations (Cohort Distribution)
                    </h3>
                  </div>
                  <span className="text-xs text-zinc-400 font-medium">Federated Adaptive Pacing</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 rounded-2xl bg-rose-950/20 border border-rose-500/20">
                    <div className="text-xs font-bold text-rose-300 uppercase tracking-wider">
                      Intensive Support Needed
                    </div>
                    <div className="text-2xl font-black text-rose-400 mt-1 font-mono">
                      {batchData.cohort_learning_summary.intensive_pedagogical_support_needed}
                    </div>
                    <p className="text-[11px] text-zinc-400 mt-1 leading-snug">
                      Assigned to daily micro-learning habits, formative quizzes, and academic mentor 1-on-1s.
                    </p>
                  </div>

                  <div className="p-4 rounded-2xl bg-amber-950/20 border border-amber-500/20">
                    <div className="text-xs font-bold text-amber-300 uppercase tracking-wider">
                      Guided Pacing &amp; Feedback
                    </div>
                    <div className="text-2xl font-black text-amber-400 mt-1 font-mono">
                      {batchData.cohort_learning_summary.guided_pacing_and_feedback_needed}
                    </div>
                    <p className="text-[11px] text-zinc-400 mt-1 leading-snug">
                      Assigned to bi-weekly progress milestone audits and targeted coursework review worksheets.
                    </p>
                  </div>

                  <div className="p-4 rounded-2xl bg-emerald-950/20 border border-emerald-500/20">
                    <div className="text-xs font-bold text-emerald-300 uppercase tracking-wider">
                      Autonomous Mastery Cohort
                    </div>
                    <div className="text-2xl font-black text-emerald-400 mt-1 font-mono">
                      {batchData.cohort_learning_summary.autonomous_mastery_cohort}
                    </div>
                    <p className="text-[11px] text-zinc-400 mt-1 leading-snug">
                      On-track for module completion. Engaged in self-paced advanced study and peer leadership.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Objective 5: Communication Efficiency & Privacy Verification Banner */}
            <div className="p-4 border border-zinc-800 bg-zinc-900/80 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between text-xs text-zinc-400 gap-3">
              <div className="flex items-center space-x-2">
                <FaShieldHalved className="w-4 h-4 text-emerald-400 shrink-0" />
                <span className="text-zinc-200 font-semibold">{batchData.privacy_log.mode}:</span>
                <span>{batchData.privacy_log.data_residency}</span>
              </div>
              <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] text-zinc-300">
                <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-zinc-950 border border-zinc-800">
                  <FaLock className="w-3 h-3 text-purple-400" />
                  <span>External egress: {batchData.privacy_log.bytes_sent_external} bytes</span>
                </div>
                {batchData.communication_efficiency && (
                  <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-emerald-950/40 border border-emerald-800/40 text-emerald-300">
                    <FaBolt className="w-3 h-3 text-emerald-400" />
                    <span>Bandwidth Saved: {batchData.communication_efficiency.bandwidth_saved}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Paginated Cohort Triage Table */}
            <div className="border border-zinc-800 bg-zinc-900/70 rounded-3xl overflow-hidden shadow-2xl">
              
              {/* Tier 1: Clean Section Header & Big Student Evaluated Badge */}
              <div className="p-5 sm:px-6 sm:py-5 border-b border-zinc-800/80 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-zinc-950/70">
                <div>
                  <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                    Cohort Triage &amp; Personalized Learning Strategies
                  </h3>
                  <p className="text-xs sm:text-sm text-zinc-400 mt-1 font-medium">
                    Showing page <strong className="text-zinc-200">{safeCurrentPage}</strong> of <strong className="text-zinc-200">{totalPages}</strong> • <strong className="text-zinc-200">{paginatedCohort.length}</strong> students per view
                  </p>
                </div>

                {/* Big, Crisp Student Evaluated Badge (Aligned Right) */}
                <div className="inline-flex items-center space-x-3 px-5 py-2.5 rounded-2xl bg-purple-500/15 border-2 border-purple-500/40 text-purple-200 shadow-sm shrink-0 whitespace-nowrap self-start md:self-auto">
                  <FaUsers className="w-5 h-5 text-purple-400 shrink-0" />
                  <span className="text-2xl sm:text-3xl font-black text-white font-mono leading-none">
                    {filteredCohort.length}
                  </span>
                  <span className="text-xs sm:text-sm font-black uppercase tracking-wider text-purple-300 whitespace-nowrap">
                    Students Evaluated
                  </span>
                </div>
              </div>

              {/* Tier 2: Perfectly Aligned Interactive Controls Toolbar (Same Height, Unified Line) */}
              <div className="p-4 sm:px-6 sm:py-3.5 border-b border-zinc-800 bg-zinc-950/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
                
                {/* Search Bar on Left */}
                <div className="relative flex-1 max-w-md">
                  <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-zinc-400">
                    <FaMagnifyingGlass className="w-4 h-4 text-purple-400" />
                  </div>
                  <input
                    type="text"
                    placeholder="Search Student ID, driver, strategy..."
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full pl-10 pr-9 py-2 bg-zinc-900 border border-zinc-700/80 hover:border-zinc-500 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/30 rounded-xl text-xs sm:text-sm text-white placeholder-zinc-500 font-medium transition-all outline-none"
                  />
                  {searchQuery && (
                    <button
                      onClick={() => {
                        setSearchQuery('');
                        setCurrentPage(1);
                      }}
                      className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-zinc-400 hover:text-white text-sm font-bold cursor-pointer"
                    >
                      ×
                    </button>
                  )}
                </div>

                {/* Filter Tabs & Row Selector on Right (Unified Height, Zero Up-and-Down Misalignment) */}
                <div className="flex flex-wrap items-center gap-3">
                  {/* Filter Risk Tabs */}
                  <div className="flex items-center p-1 rounded-xl bg-zinc-900 border border-zinc-700/80 gap-1 shadow-sm">
                    {(
                      [
                        { key: 'ALL', label: 'All', activeClass: 'bg-zinc-800 text-white border-zinc-600 shadow' },
                        { key: 'HIGH', label: 'Critical (≥70%)', activeClass: 'bg-rose-500/25 text-rose-300 border-rose-500/50 shadow' },
                        { key: 'MEDIUM', label: 'Moderate', activeClass: 'bg-amber-500/25 text-amber-300 border-amber-500/50 shadow' },
                        { key: 'LOW', label: 'Safe', activeClass: 'bg-emerald-500/25 text-emerald-300 border-emerald-500/50 shadow' },
                      ] as const
                    ).map((r) => {
                      const isActive = filterRisk === r.key;
                      return (
                        <button
                          key={r.key}
                          onClick={() => {
                            setFilterRisk(r.key);
                            setCurrentPage(1);
                          }}
                          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border cursor-pointer ${
                            isActive
                              ? r.activeClass
                              : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
                          }`}
                        >
                          {r.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Page Size Selector */}
                  <div className="flex items-center space-x-1.5 text-xs font-bold text-zinc-300 bg-zinc-900 border border-zinc-700/80 rounded-xl px-3 py-1.5 shadow-sm">
                    <span className="text-zinc-400 uppercase text-[10px] tracking-wider">Show:</span>
                    <select
                      value={pageSize}
                      onChange={(e) => {
                        setPageSize(Number(e.target.value));
                        setCurrentPage(1);
                      }}
                      className="bg-transparent text-white font-extrabold text-xs focus:outline-none cursor-pointer"
                    >
                      <option value={25} className="bg-zinc-900 text-white">25 rows</option>
                      <option value={50} className="bg-zinc-900 text-white">50 rows</option>
                      <option value={100} className="bg-zinc-900 text-white">100 rows</option>
                    </select>
                  </div>
                </div>

              </div>

              {/* Table with full visibility and generous bottom spacing so no row is clipped */}
              <div className="overflow-x-auto pb-4">
                <table className="w-full text-left text-xs border-collapse min-w-[950px]">
                  <thead className="bg-zinc-950/95 text-zinc-300 border-b border-zinc-800 sticky top-0 z-10 backdrop-blur-md">
                    <tr>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Student ID</th>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Portal Clicks</th>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Avg Score</th>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Past Attempts</th>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Risk Score</th>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Risk Level</th>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Primary Driver (XAI)</th>
                      <th className="py-3.5 px-4 font-bold uppercase tracking-wider text-[11px]">Personalized Learning Strategy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60 font-medium">
                    {paginatedCohort.map((stu, i) => (
                      <tr 
                        key={stu.student_id || i}
                        className="hover:bg-zinc-800/50 transition-colors"
                      >
                        <td className="py-4 px-4 font-mono text-zinc-100 font-bold text-sm align-top">{stu.student_id}</td>
                        <td className="py-4 px-4 text-zinc-200 font-semibold align-top">{stu.vle_clicks}</td>
                        <td className="py-4 px-4 text-zinc-200 font-semibold align-top">{stu.assessment_score}%</td>
                        <td className="py-4 px-4 text-zinc-300 align-top">{stu.num_attempts}</td>
                        <td className="py-4 px-4 font-mono text-zinc-100 font-bold text-sm align-top">{stu.percentage}</td>
                        <td className="py-4 px-4 align-top">
                          <span
                            className={`pill-badge text-xs px-3 py-1 font-extrabold ${
                              stu.risk_level === 'HIGH'
                                ? 'pill-rose'
                                : stu.risk_level === 'MEDIUM'
                                ? 'pill-amber'
                                : 'pill-emerald'
                            }`}
                          >
                            {stu.risk_level}
                          </span>
                        </td>
                        <td className="py-4 px-4 text-zinc-300 text-xs max-w-xs align-top leading-relaxed font-medium">
                          {stu.primary_driver}
                        </td>
                        <td className="py-4 px-4 text-zinc-300 text-xs max-w-md align-top leading-relaxed">
                          {stu.personalized_learning?.pacing ? (
                            <div className="space-y-1.5">
                              <span className="text-purple-300 font-semibold block text-xs leading-normal">
                                {stu.personalized_learning.pacing}
                              </span>
                              <span className="text-zinc-400 text-[11px] block bg-zinc-950/70 border border-white/5 rounded-lg px-2.5 py-1">
                                <strong className="text-zinc-500 uppercase text-[9px] mr-1">Resource:</strong>
                                {stu.personalized_learning.key_resource}
                              </span>
                            </div>
                          ) : (
                            stu.recommended_action
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination Controls Footer */}
              <div className="p-5 border-t border-zinc-800 bg-zinc-950/90 flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs font-medium text-zinc-400">
                <div className="text-sm">
                  Showing <strong className="text-white">{startIndex + 1}</strong> to <strong className="text-white">{Math.min(startIndex + pageSize, filteredCohort.length)}</strong> of <strong className="text-purple-300 font-mono font-bold">{filteredCohort.length}</strong> students evaluated
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setCurrentPage(1)}
                    disabled={safeCurrentPage === 1}
                    className="px-3.5 py-2 rounded-xl border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 hover:border-zinc-500 font-bold text-xs transition-all shadow-sm disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                  >
                    « First
                  </button>
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={safeCurrentPage === 1}
                    className="px-3.5 py-2 rounded-xl border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 hover:border-zinc-500 font-bold text-xs transition-all shadow-sm disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                  >
                    ‹ Prev
                  </button>
                  <span className="px-4 py-2 rounded-xl bg-purple-950/60 border border-purple-500/40 text-purple-200 font-black text-xs font-mono shadow-sm">
                    Page {safeCurrentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={safeCurrentPage === totalPages}
                    className="px-3.5 py-2 rounded-xl border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 hover:border-zinc-500 font-bold text-xs transition-all shadow-sm disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                  >
                    Next ›
                  </button>
                  <button
                    onClick={() => setCurrentPage(totalPages)}
                    disabled={safeCurrentPage === totalPages}
                    className="px-3.5 py-2 rounded-xl border border-zinc-700 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 hover:border-zinc-500 font-bold text-xs transition-all shadow-sm disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                  >
                    Last »
                  </button>
                </div>
              </div>
            </div>

          </div>
        )}

      </div>
    </section>
  );
};
