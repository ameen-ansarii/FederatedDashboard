import { useState, useEffect } from 'react';

export function useCountUp(target: number, duration: number = 1400, active: boolean = true, decimals: number = 0): number {
  const [val, setVal] = useState(0);

  useEffect(() => {
    if (!active || target === undefined || isNaN(target)) {
      setVal(0);
      return;
    }

    let startTimestamp: number | null = null;
    let animationFrameId: number;

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      // easeOutCubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = ease * target;
      
      setVal(decimals === 0 ? Math.round(current) : parseFloat(current.toFixed(decimals)));

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(step);
      } else {
        setVal(target);
      }
    };

    animationFrameId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animationFrameId);
  }, [target, duration, active, decimals]);

  return val;
}
