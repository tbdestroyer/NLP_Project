import { Line, LineChart, ResponsiveContainer } from 'recharts';
import { STATUS_COLOR } from './ui';

export default function Sparkline({ values, width, height, className = '' }) {
  const data = values.map((value, turn) => ({ turn, score: value }));
  const color = values.at(-1) - values[0] >= 0 ? STATUS_COLOR.Rising : STATUS_COLOR.Cooling;
  return (
    <div className={`inline-block shrink-0 ${className}`} style={{ width, height }} aria-hidden="true">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 3, right: 1, bottom: 3, left: 1 }}>
          <Line type="monotone" dataKey="score" stroke={color} strokeWidth={1.8} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
