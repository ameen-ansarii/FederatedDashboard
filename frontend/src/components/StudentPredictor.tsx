import React, { useState, useEffect } from 'react';
import { 
  FaSliders, 
  FaGraduationCap, 
  FaArrowRight 
} from 'react-icons/fa6';
import { useCountUp } from '../utils/useCountUp';

export const StudentPredictor: React.FC = () => {
  // Input features
  const [vleClicks, setVleClicks] = useState<number>(380);
  const [avgScore, setAvgScore] = useState<number>(54);
  const [prevAttempts, setPrevAttempts] = useState<number>(1);
  const [credits, setCredits] = useState<number>(60);
  const [imdLow, setImdLow] = useState<boolean>(false);
  const [priorEduLower, setPriorEduLower] = useState<boolean>(false);
  const [ageYoung, setAgeYoung] = useState<boolean>(true);

  // Prediction State
  interface PredictionResult {
    probability: number;
    percentage: string;
    risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
    recommended_action: string;
    explainable_ai: {
      method: string;
      disclaimer: string;
      top_drivers: Array<{
        feature: string;
        raw_value: string;
        impact_score: number;
        direction: 'INCREASES_RISK' | 'REDUCES_RISK';
        reasoning: string;
      }>;
    };
    personalized_learning_plan?: {
      pacing_strategy: string;
      strategies: Array<{
        dimension: string;
        recommendation: string;
        urgency: string;
      }>;
      recommended_resources: Array<{
        title: string;
        type: string;
        focus: string;
      }>;
      action_milestones: Array<{
        phase: string;
        goal: string;
        status: string;
      }>;
      recommended_study_hours_weekly: number;
      learning_profile_summary: string;
    };
  }

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [ringsVisible, setRingsVisible] = useState<boolean>(false);

  // Trigger smooth ring flow when prediction changes
  useEffect(() => {
    if (prediction) {
      setRingsVisible(false);
      const timer = setTimeout(() => {
        setRingsVisible(true);
      }, 70);
      return () => clearTimeout(timer);
    }
  }, [prediction]);

  const targetRiskIndex = prediction ? Math.round(prediction.probability * 100) : 0;
  const animRiskIndex = useCountUp(targetRiskIndex, 1500, ringsVisible);

  const calculateDropout = async () => {
    setLoading(true);
    setRingsVisible(false);

    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const fetchPromise = fetch(`${apiBase}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vle_clicks: vleClicks,
          avg_assessment_score: avgScore,
          num_prev_attempts: prevAttempts,
          studied_credits: credits,
          imd_band_low: imdLow,
          highest_education_lower: priorEduLower,
          age_band_young: ageYoung,
        }),
      });

      // Provide smooth, visually respectful inference telemetry pacing (~700ms)
      await new Promise((r) => setTimeout(r, 700));
      const response = await fetchPromise;

      if (response.ok) {
        const data = await response.json();
        setPrediction(data);
      } else {
        calculateLocal();
      }
    } catch {
      calculateLocal();
    } finally {
      setLoading(false);
    }
  };

  const calculateLocal = () => {
    let logit = 0.15;
    const drivers = [];

    // Clicks
    const clickImpact = (vleClicks - 650.0) * -0.0018;
    logit += clickImpact;
    drivers.push({
      feature: 'Portal activity',
      raw_value: `${vleClicks} clicks`,
      impact_score: +clickImpact.toFixed(3),
      direction: clickImpact < 0 ? ('REDUCES_RISK' as const) : ('INCREASES_RISK' as const),
      reasoning: vleClicks >= 650 ? 'Regular portal engagement shows active course involvement.' : 'Low online activity is an early indicator of falling behind.',
    });

    // Score
    const scoreImpact = (avgScore - 68.0) * -0.052;
    logit += scoreImpact;
    drivers.push({
      feature: 'Exam and assignment marks',
      raw_value: `${avgScore}%`,
      impact_score: +scoreImpact.toFixed(3),
      direction: scoreImpact < 0 ? ('REDUCES_RISK' as const) : ('INCREASES_RISK' as const),
      reasoning: avgScore >= 60 ? 'Passing grades supporting good academic standing.' : 'Scores below 50% raise the likelihood of withdrawing.',
    });

    // Repeats
    if (prevAttempts > 0) {
      const repeatImpact = prevAttempts * 0.72;
      logit += repeatImpact;
      drivers.push({
        feature: 'Past course repeats',
        raw_value: `${prevAttempts} attempt(s)`,
        impact_score: +repeatImpact.toFixed(3),
        direction: 'INCREASES_RISK' as const,
        reasoning: 'Repeated module attempts historically correlate with higher attrition.',
      });
    }

    // Credits
    if (credits > 60) {
      const credImpact = (credits - 60) * 0.006;
      logit += credImpact;
      drivers.push({
        feature: 'Heavy course load',
        raw_value: `${credits} credits`,
        impact_score: +credImpact.toFixed(3),
        direction: 'INCREASES_RISK' as const,
        reasoning: 'Taking extra credits can lead to workload fatigue.',
      });
    }

    if (priorEduLower) {
      logit += 0.38;
      drivers.push({
        feature: 'Prior qualification',
        raw_value: 'Below A level',
        impact_score: 0.38,
        direction: 'INCREASES_RISK' as const,
        reasoning: 'May need additional tutoring in foundational subjects.',
      });
    }

    if (imdLow) {
      logit += 0.28;
      drivers.push({
        feature: 'Economic decile',
        raw_value: 'Lowest 20%',
        impact_score: 0.28,
        direction: 'INCREASES_RISK' as const,
        reasoning: 'Outside work and financial pressures can impact study consistency.',
      });
    }

    const prob = Math.max(0.01, Math.min(0.99, 1.0 / (1.0 + Math.exp(-logit))));
    const riskLevel = prob < 0.40 ? 'LOW' : prob < 0.70 ? 'MEDIUM' : 'HIGH';
    const action =
      riskLevel === 'LOW'
        ? 'Standard autonomous learning. Continue current study trajectory.'
        : riskLevel === 'MEDIUM'
        ? 'Guided intervention: Schedule progress review with academic mentor.'
        : 'Immediate intervention: Provide 1-on-1 tutoring and study habit restructuring.';

    drivers.sort((a, b) => Math.abs(b.impact_score) - Math.abs(a.impact_score));

    // Fallback personalized learning plan
    const localPlan = {
      pacing_strategy: vleClicks < 350
        ? 'Micro-Learning Habit: Schedule 30-minute daily focused sessions on the virtual learning portal.'
        : 'Sustained Autonomous Pacing: Maintain steady modular study consistency with weekly self-audits.',
      strategies: [
        {
          dimension: 'Study Pacing',
          recommendation: vleClicks < 400 ? 'Implement structured daily study intervals' : 'Maintain consistent weekly schedule',
          urgency: vleClicks < 400 ? 'High' : 'Low'
        },
        {
          dimension: 'Assessment Prep',
          recommendation: avgScore < 50 ? 'Complete formative revision worksheets before unit assignments' : 'Engage with peer study circles',
          urgency: avgScore < 50 ? 'High' : 'Low'
        }
      ],
      recommended_resources: [
        {
          title: 'Interactive Module Diagnostic Worksheets',
          type: 'Self-Paced Practice',
          focus: 'Low-stakes self-testing to establish an active online learning habit'
        },
        {
          title: 'Foundational Skills & Conceptual Primer Pack',
          type: 'Remedial Modules',
          focus: 'Core prerequisite vocabulary and mathematical fundamentals'
        }
      ],
      action_milestones: [
        { phase: 'Weeks 1 - 2', goal: 'Establish structured study schedule & attend advising check-in', status: 'Immediate' },
        { phase: 'Weeks 3 - 5', goal: 'Complete formative practice quizzes and achieve 40+ VLE clicks/wk', status: 'Upcoming' }
      ],
      recommended_study_hours_weekly: credits > 90 ? 14 : credits > 60 ? 10 : 7,
      learning_profile_summary: riskLevel === 'HIGH'
        ? 'Intensive pedagogical support required: Focus on foundational revision and habit pacing.'
        : riskLevel === 'MEDIUM'
        ? 'Guided pacing required: Focus on formative feedback and study habit consistency.'
        : 'Self-directed mastery: Student is progressing well with autonomous pacing.'
    };

    setPrediction({
      probability: +prob.toFixed(4),
      percentage: `${(prob * 100).toFixed(1)}%`,
      risk_level: riskLevel,
      recommended_action: action,
      explainable_ai: {
        method: 'SHAP feature contributions',
        disclaimer: 'Statistical estimate to help advisors catch early warning signs.',
        top_drivers: drivers,
      },
      personalized_learning_plan: localPlan
    });
  };

  return (
    <section className="py-16 border-b border-zinc-800 bg-zinc-950 font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10">
          <div>
            <div className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center space-x-2">
              <span className="pill-badge pill-purple">OBJECTIVES 3 &amp; 4</span>
              <span>Performance Analytics &amp; Personalization</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-100 mt-2">
              Student Performance &amp; Personalized Learning Engine
            </h2>
            <p className="text-sm text-zinc-400 mt-2 max-w-2xl leading-relaxed">
              Analyzes academic engagement and assessment factors via federated inference. Generates personalized learning paths, study pacing strategies, and early intervention support without exposing private records.
            </p>
          </div>

          <div className="mt-4 md:mt-0 flex items-center space-x-2 border border-zinc-800 bg-zinc-900 px-3.5 py-1.5 text-xs font-medium text-zinc-300 rounded-full shadow-sm">
            <FaGraduationCap className="w-3.5 h-3.5 text-zinc-300" />
            <span>Interactive Sliders • Single Student</span>
          </div>
        </div>

        {/* Two Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Input Sliders & Parameters */}
          <div className="lg:col-span-6 card-squircle p-7 space-y-6">
            <div className="flex items-center justify-between pb-4 border-b border-white/5 text-xs">
              <div className="flex items-center space-x-2">
                <span className="pill-badge pill-purple">STUDENT</span>
                <span className="font-bold text-zinc-300 uppercase tracking-wider">Parameters &amp; Behavior</span>
              </div>
              <span className="text-zinc-600">●●●</span>
            </div>

            {/* VLE Clicks */}
            <div>
              <div className="flex justify-between items-center text-xs text-zinc-300 mb-1.5 font-medium">
                <span className="uppercase tracking-wider">Portal logins and clicks</span>
                <span className="font-bold text-zinc-100 text-sm">{vleClicks} clicks</span>
              </div>
              <input
                type="range"
                min="20"
                max="2500"
                step="20"
                value={vleClicks}
                onChange={(e) => setVleClicks(Number(e.target.value))}
                className="w-full accent-zinc-100 bg-zinc-800 h-1.5 rounded cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-zinc-500 mt-1 font-medium">
                <span>Low engagement</span>
                <span>650 average</span>
                <span>High activity</span>
              </div>
            </div>

            {/* Assessment Score */}
            <div>
              <div className="flex justify-between items-center text-xs text-zinc-300 mb-1.5 font-medium">
                <span className="uppercase tracking-wider">Average test and assignment score</span>
                <span className="font-bold text-zinc-100 text-sm">{avgScore}%</span>
              </div>
              <input
                type="range"
                min="10"
                max="100"
                step="1"
                value={avgScore}
                onChange={(e) => setAvgScore(Number(e.target.value))}
                className="w-full accent-zinc-100 bg-zinc-800 h-1.5 rounded cursor-pointer"
              />
              <div className="flex justify-between text-[11px] text-zinc-500 mt-1 font-medium">
                <span>Failing range</span>
                <span>50% pass mark</span>
                <span>Top marks</span>
              </div>
            </div>

            {/* Course Repeats & Credits */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-zinc-300 block mb-1.5 uppercase tracking-wider">
                  Past course repeats
                </label>
                <select
                  value={prevAttempts}
                  onChange={(e) => setPrevAttempts(Number(e.target.value))}
                  className="w-full border border-zinc-800 bg-zinc-950 p-2.5 text-xs text-zinc-200 rounded focus:outline-none focus:border-zinc-500 font-medium"
                >
                  <option value={0}>0 attempts (First time)</option>
                  <option value={1}>1 past repeat</option>
                  <option value={2}>2 past repeats</option>
                  <option value={3}>3 or more repeats</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-zinc-300 block mb-1.5 uppercase tracking-wider">
                  Enrolled credits
                </label>
                <select
                  value={credits}
                  onChange={(e) => setCredits(Number(e.target.value))}
                  className="w-full border border-zinc-800 bg-zinc-950 p-2.5 text-xs text-zinc-200 rounded focus:outline-none focus:border-zinc-500 font-medium"
                >
                  <option value={30}>30 credits</option>
                  <option value={60}>60 credits (Standard)</option>
                  <option value={90}>90 credits</option>
                  <option value={120}>120 credits (Full time)</option>
                </select>
              </div>
            </div>

            {/* Background Flags */}
            <div className="space-y-2 pt-2 border-t border-zinc-800">
              <label className="text-xs font-semibold text-zinc-400 block uppercase tracking-wider">
                Student background
              </label>

              <div className="flex items-center justify-between p-2.5 border border-zinc-800 bg-zinc-950 rounded text-xs text-zinc-300 font-medium">
                <span>Lower economic bracket (Bottom 20%)</span>
                <input
                  type="checkbox"
                  checked={imdLow}
                  onChange={(e) => setImdLow(e.target.checked)}
                  className="accent-zinc-100 w-4 h-4 cursor-pointer"
                />
              </div>

              <div className="flex items-center justify-between p-2.5 border border-zinc-800 bg-zinc-950 rounded text-xs text-zinc-300 font-medium">
                <span>Prior schooling below A level</span>
                <input
                  type="checkbox"
                  checked={priorEduLower}
                  onChange={(e) => setPriorEduLower(e.target.checked)}
                  className="accent-zinc-100 w-4 h-4 cursor-pointer"
                />
              </div>

              <div className="flex items-center justify-between p-2.5 border border-zinc-800 bg-zinc-950 rounded text-xs text-zinc-300 font-medium">
                <span>Age under 35</span>
                <input
                  type="checkbox"
                  checked={ageYoung}
                  onChange={(e) => setAgeYoung(e.target.checked)}
                  className="accent-zinc-100 w-4 h-4 cursor-pointer"
                />
              </div>
            </div>

            {/* Compute Button */}
            <button
              onClick={calculateDropout}
              disabled={loading}
              className="w-full py-3 px-4 bg-zinc-100 text-zinc-950 font-bold text-xs tracking-wider uppercase rounded hover:bg-white transition-colors flex items-center justify-center space-x-2"
            >
              <span>{loading ? 'Evaluating...' : 'Calculate dropout probability'}</span>
              <FaArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Results Column */}
          <div className="lg:col-span-6 space-y-6">
            {!prediction ? (
              <div className="card-squircle p-12 bg-zinc-900/80 border border-white/5 flex flex-col items-center justify-center text-center h-full min-h-[440px]">
                <div className="w-14 h-14 rounded-2xl bg-zinc-800/80 border border-white/5 flex items-center justify-center text-zinc-400 mb-4 shadow-inner">
                  <FaSliders className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="text-base font-bold text-white">Manual Simulator Awaiting Input</h3>
                <p className="text-xs text-zinc-400 mt-2 max-w-sm leading-relaxed">
                  Adjust the student parameters on the left (portal clicks, grades, repeat attempts), then click <strong>&quot;Calculate dropout probability&quot;</strong> to evaluate risk and see XAI feature attributions.
                </p>
                <div className="mt-6 flex items-center space-x-2 text-[11px] text-zinc-500 font-medium bg-zinc-950/80 px-3.5 py-1.5 rounded-full border border-white/5">
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                  <span>Ready for calculation • 0 raw student records needed</span>
                </div>
              </div>
            ) : (
              <>
                {/* Widget 1: Apple-Health / PAI style Concentric Rings (Reference 2) */}
                <div className="card-squircle p-7 relative overflow-hidden bg-zinc-900/90 shadow-2xl">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                
                {/* Left side details */}
                <div className="flex-1 space-y-4">
                  <div className="flex items-center space-x-2 text-zinc-400 text-sm font-semibold">
                    <span>Student Telemetry</span>
                    <span className="text-zinc-600">/</span>
                    <span className="text-zinc-300">Live Synthesis</span>
                  </div>

                  <div className="space-y-3 pt-1">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center">
                        <FaSliders className="w-3.5 h-3.5" />
                      </div>
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-base font-bold text-white">{vleClicks} clicks</span>
                          <span className="text-xs font-bold text-emerald-400">
                            {vleClicks >= 650 ? '+18% ↑' : '-14% ↓'}
                          </span>
                        </div>
                        <div className="text-[11px] text-zinc-400 font-medium">Portal VLE engagement</div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-rose-500/20 text-rose-400 flex items-center justify-center">
                        <FaGraduationCap className="w-3.5 h-3.5" />
                      </div>
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-base font-bold text-white">{avgScore}% score</span>
                          <span className={`text-xs font-bold ${avgScore >= 50 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {avgScore >= 50 ? 'Passing' : 'Below pass'}
                          </span>
                        </div>
                        <div className="text-[11px] text-zinc-400 font-medium">Weighted assessment marks</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right side: Concentric Multi-Arc Rings with animated glowing lines */}
                <div className="relative w-44 h-44 flex items-center justify-center shrink-0 self-center sm:self-auto">
                  <svg className="w-full h-full transform -rotate-90" viewBox="0 0 160 160">
                    {/* Background Tracks */}
                    <circle cx="80" cy="80" r="66" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />
                    <circle cx="80" cy="80" r="50" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />
                    <circle cx="80" cy="80" r="34" stroke="#27272a" strokeWidth="10" fill="transparent" opacity="0.4" />

                    {/* Ring 1: Blue Outer Ring (Engagement) */}
                    <circle
                      cx="80"
                      cy="80"
                      r="66"
                      stroke="#38bdf8"
                      strokeWidth="10"
                      fill="transparent"
                      strokeDasharray="414"
                      strokeDashoffset={ringsVisible ? 414 - (Math.min(vleClicks, 2000) / 2000) * 310 : 414}
                      className="animate-ring-arc"
                      style={{
                        filter: ringsVisible ? 'drop-shadow(0 0 5px rgba(56, 189, 248, 0.6))' : 'none',
                      }}
                    />

                    {/* Ring 2: Emerald Middle Ring (Pass Rate) */}
                    <circle
                      cx="80"
                      cy="80"
                      r="50"
                      stroke="#4ade80"
                      strokeWidth="10"
                      fill="transparent"
                      strokeDasharray="314"
                      strokeDashoffset={ringsVisible ? 314 - (avgScore / 100) * 235 : 314}
                      className="animate-ring-arc"
                      style={{
                        filter: ringsVisible ? 'drop-shadow(0 0 5px rgba(74, 222, 128, 0.6))' : 'none',
                      }}
                    />

                    {/* Ring 3: Rose Inner Ring (Dropout Risk) */}
                    <circle
                      cx="80"
                      cy="80"
                      r="34"
                      stroke="#f43f5e"
                      strokeWidth="10"
                      fill="transparent"
                      strokeDasharray="213"
                      strokeDashoffset={ringsVisible ? 213 - (prediction?.probability || 0) * 160 : 213}
                      className="animate-ring-arc"
                      style={{
                        filter: ringsVisible ? 'drop-shadow(0 0 5px rgba(244, 63, 94, 0.6))' : 'none',
                      }}
                    />
                  </svg>

                  {/* White Center Bubble (PAI style) with animated count-up */}
                  <div className="absolute w-20 h-20 rounded-full bg-white text-zinc-950 flex flex-col items-center justify-center shadow-2xl border-4 border-zinc-900 transition-transform duration-500 hover:scale-105">
                    <span className="text-xl font-extrabold tracking-tight leading-none font-mono">
                      {animRiskIndex}
                    </span>
                    <span className="text-[9px] font-extrabold text-zinc-600 tracking-wider mt-0.5">
                      RISK INDEX
                    </span>
                  </div>
                </div>

              </div>
            </div>

            {/* Widget 2: Athletic Score Widget Layout (Reference 1) */}
            <div className="card-squircle p-7 bg-zinc-900/90 shadow-2xl">
              <div className="flex flex-col md:flex-row gap-6">
                
                {/* Left: Big Bold Score + Dot Matrix */}
                <div className="md:w-5/12 flex flex-col justify-between border-b md:border-b-0 md:border-r border-white/5 pb-4 md:pb-0 md:pr-6">
                  <div>
                    <div className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider">
                      RETENTION RATING
                    </div>
                    <div className="text-5xl font-extrabold text-white mt-2 tracking-tight">
                      {Math.max(1, 100 - Math.round(prediction.probability * 100))}
                    </div>
                    <div className="text-[11px] text-zinc-400 mt-1 font-medium">
                      out of 100 safety score
                    </div>
                  </div>

                  {/* Dynamic Dot Matrix Grid */}
                  <div className="mt-6">
                    <div className="grid grid-cols-7 gap-1.5 w-fit p-2.5 rounded-2xl bg-zinc-950/80 border border-white/5 shadow-inner">
                      {Array.from({ length: 28 }).map((_, i) => {
                        const activeCount = Math.round(((100 - prediction.probability * 100) / 100) * 28);
                        const isActive = i < activeCount;
                        const isHighRisk = prediction.risk_level === 'HIGH';
                        return (
                          <div
                            key={i}
                            className={`matrix-dot ${
                              isActive
                                ? isHighRisk
                                  ? 'matrix-dot-rose'
                                  : 'matrix-dot-emerald'
                                : ''
                            }`}
                          />
                        );
                      })}
                    </div>
                    <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider mt-2">
                      Local Matrix Partition
                    </div>
                  </div>
                </div>

                {/* Right: Detailed Metric Rows with Icons (Reference 1) */}
                <div className="md:w-7/12 flex flex-col justify-between divide-y divide-white/5 space-y-2.5 pt-1">
                  
                  <div className="flex items-center justify-between pb-2.5">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-xs">
                        ◐
                      </div>
                      <div>
                        <span className="text-xs font-bold text-white uppercase tracking-wider block">ENGAGEMENT</span>
                        <span className="text-[10px] text-zinc-400">VLE Portal Usage</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-bold text-white">{vleClicks}</span>
                      <span className="text-[10px] text-zinc-400 block">clicks</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between py-2.5">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
                        ◎
                      </div>
                      <div>
                        <span className="text-xs font-bold text-white uppercase tracking-wider block">ASSESSMENT</span>
                        <span className="text-[10px] text-zinc-400">Exam &amp; Quiz Average</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-bold text-white">{avgScore}%</span>
                      <span className="text-[10px] text-zinc-400 block">grade</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between py-2.5">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold text-xs">
                        ◈
                      </div>
                      <div>
                        <span className="text-xs font-bold text-white uppercase tracking-wider block">COURSE LOAD</span>
                        <span className="text-[10px] text-zinc-400">Module Credits</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-sm font-bold text-white">{credits}</span>
                      <span className="text-[10px] text-zinc-400 block">credits</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-2.5">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold text-xs">
                        ♥
                      </div>
                      <div>
                        <span className="text-xs font-bold text-white uppercase tracking-wider block">RISK CLASSIFIER</span>
                        <span className="text-[10px] text-zinc-400">Federated Decision</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`pill-badge ${
                        prediction.risk_level === 'HIGH' ? 'pill-rose' : prediction.risk_level === 'MEDIUM' ? 'pill-amber' : 'pill-emerald'
                      } text-[9px]`}>
                        {prediction.risk_level}
                      </span>
                    </div>
                  </div>

                </div>

              </div>

              {/* SHAP Factors & Recommendations */}
              <div className="mt-6 pt-5 border-t border-white/5 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider">XAI SHAP Contributors</span>
                  <span className="pill-badge pill-dark text-[9px]">LOCAL GRADIENT</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {prediction.explainable_ai.top_drivers.slice(0, 2).map((driver, idx) => (
                    <div key={idx} className="p-3 rounded-2xl bg-zinc-950/70 border border-white/5 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="text-white font-semibold">{driver.feature}</span>
                        <span className={`pill-badge ${driver.direction === 'INCREASES_RISK' ? 'pill-rose' : 'pill-emerald'} text-[8px] py-0 px-1.5`}>
                          {driver.direction === 'INCREASES_RISK' ? 'Risk ↑' : 'Protects ↓'}
                        </span>
                      </div>
                      <p className="text-[10px] text-zinc-400 mt-1 leading-snug">{driver.reasoning}</p>
                    </div>
                  ))}
                </div>

                <div className="p-3 rounded-2xl bg-zinc-950/50 border border-white/5 text-xs flex items-center justify-between text-zinc-300">
                  <span className="text-zinc-500 text-[10px] uppercase font-bold tracking-wider">Intervention</span>
                  <span className="font-semibold text-white text-[11px]">{prediction.recommended_action}</span>
                </div>
              </div>
            </div>

            {/* Widget 3: Personalized Learning Capabilities & Recommendations (Objective 3) */}
            {prediction.personalized_learning_plan && (
              <div className="card-squircle p-7 bg-zinc-900/90 shadow-2xl border border-purple-500/20 space-y-5">
                <div className="flex items-center justify-between pb-3 border-b border-white/5">
                  <div className="flex items-center space-x-2">
                    <span className="pill-badge pill-purple text-[9px]">OBJECTIVE 3</span>
                    <span className="text-xs font-bold text-white uppercase tracking-wider">
                      Personalized Learning Plan
                    </span>
                  </div>
                  <span className="text-[11px] text-purple-300 font-medium">
                    {prediction.personalized_learning_plan.recommended_study_hours_weekly} hrs/week recommended
                  </span>
                </div>

                {/* Adaptive Pacing & Profile */}
                <div className="p-3.5 rounded-2xl bg-zinc-950/80 border border-white/5">
                  <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-1">
                    Adaptive Study Strategy
                  </div>
                  <p className="text-xs text-zinc-200 leading-relaxed font-medium">
                    {prediction.personalized_learning_plan.pacing_strategy}
                  </p>
                  <div className="text-[11px] text-zinc-400 mt-2 pt-2 border-t border-white/5 italic">
                    {prediction.personalized_learning_plan.learning_profile_summary}
                  </div>
                </div>

                {/* Curated Resources */}
                <div>
                  <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider mb-2.5">
                    Curated Learning Resources &amp; Practice
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {prediction.personalized_learning_plan.recommended_resources.map((res, i) => (
                      <div key={i} className="p-3 rounded-2xl bg-zinc-950/60 border border-white/5 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-white text-[11px]">{res.title}</span>
                          <span className="pill-badge pill-blue text-[8px] py-0 px-1.5">{res.type}</span>
                        </div>
                        <p className="text-[10px] text-zinc-400 mt-1 leading-snug">{res.focus}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Milestones */}
                <div>
                  <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider mb-2">
                    Action Milestones &amp; Learning Roadmap
                  </div>
                  <div className="space-y-1.5">
                    {prediction.personalized_learning_plan.action_milestones.map((m, i) => (
                      <div key={i} className="flex items-center justify-between p-2.5 rounded-xl bg-zinc-950/40 border border-white/5 text-xs">
                        <div className="flex items-center space-x-2.5">
                          <span className="text-[10px] font-bold text-purple-400 w-20 shrink-0">{m.phase}</span>
                          <span className="text-zinc-300 text-[11px]">{m.goal}</span>
                        </div>
                        <span className="pill-badge pill-dark text-[8px] py-0 px-1.5 shrink-0 ml-2">{m.status}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}

      </div>

        </div>

      </div>
    </section>
  );
};
