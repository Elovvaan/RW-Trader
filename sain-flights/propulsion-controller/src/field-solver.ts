/**
 * field-solver.ts
 *
 * Pure function converting a FieldState into 4 per-motor output values
 * for the Version-1 lynchpin drone using tangential-field (4-node) geometry.
 *
 * Motor layout — 90° node spacing:
 *   A = front-left  (0°   / 0 rad)
 *   B = front-right (90°  / π/2 rad)
 *   C = rear-right  (180° / π rad)
 *   D = rear-left   (270° / 3π/2 rad)
 *
 * Base model (per motor i at angular offset θᵢ):
 *   outputᵢ = clamp( base + biasOffset + modAmp · sin(phase + θᵢ),  0, 1 )
 *
 * where:
 *   base       = intensity / 100          (normalise 0..100 → 0..1)
 *   modAmp     = base × MODULATION_DEPTH  (scales with intensity so idle is quiet)
 *   biasOffset = bias × modAmp            (−1..1 contraction/expansion)
 *   phase      advances each tick via advancePhase()
 *
 * The modulation terms sum to zero across all 4 motors (sin property),
 * so total thrust is preserved under pure phase rotation.
 */

/** Master lift/energy and rotation parameters passed to the solver. */
export type FieldState = {
  /** Normalised master lift/energy: 0 (off) to 100 (full). */
  intensity: number;
  /** Current phase angle in radians. */
  phase: number;
  /** Phase advance per second in radians/s. */
  phaseVelocity: number;
  /** Rotation direction: 1 = CCW (positive advance), −1 = CW (negative advance). */
  spin: 1 | -1;
  /**
   * Contraction/expansion bias: −1 = full contraction (reduce all outputs),
   * +1 = full expansion (increase all outputs).
   */
  bias: number;
  /** When false the field is inactive and all motors receive equal base output. */
  enabled: boolean;
};

/** Normalised [0..1] per-motor outputs for motors A, B, C, D. */
export type MotorOutputs = {
  /** Front-left motor output, normalised 0..1. */
  a: number;
  /** Front-right motor output, normalised 0..1. */
  b: number;
  /** Rear-right motor output, normalised 0..1. */
  c: number;
  /** Rear-left motor output, normalised 0..1. */
  d: number;
};

// Angular offsets for each motor (radians), ordered A → D.
const MOTOR_OFFSETS_RAD: [number, number, number, number] = [
  0,               // A: 0°
  Math.PI / 2,     // B: 90°
  Math.PI,         // C: 180°
  (3 * Math.PI) / 2, // D: 270°
];

/**
 * Maximum phase-modulation depth expressed as a fraction of base output.
 * At full intensity the per-motor oscillation amplitude is ±30 % of base.
 */
const MODULATION_DEPTH = 0.3;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Clamp `value` to the closed interval [min, max]. */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Advance `phase` by one time step.
 *
 * @param phase    Current phase in radians.
 * @param velocity Phase advance per second in radians/s.
 * @param spin     1 = CCW (positive advance), −1 = CW (negative advance).
 * @param dt       Elapsed time in seconds.
 * @returns        New phase in radians, wrapped to [0, 2π).
 */
export function advancePhase(
  phase: number,
  velocity: number,
  spin: 1 | -1,
  dt: number,
): number {
  const twoPi = 2 * Math.PI;
  const delta = velocity * spin * dt;
  return ((phase + delta) % twoPi + twoPi) % twoPi;
}

/**
 * Solve per-motor outputs from a FieldState.
 *
 * This function is pure and deterministic — given the same FieldState it
 * always produces the same MotorOutputs.  It never reads from the
 * environment and has no side-effects.
 *
 * When `state.enabled` is false, all motors receive the same base output
 * (equal to `intensity / 100`), preserving a flat-throttle quad behaviour.
 *
 * @param state  The current field state.
 * @returns      Normalised motor outputs, each guaranteed in [0, 1].
 */
export function solveField(state: FieldState): MotorOutputs {
  const base = clamp(state.intensity / 100, 0, 1);

  if (!state.enabled) {
    return { a: base, b: base, c: base, d: base };
  }

  const modAmp = base * MODULATION_DEPTH;
  const biasOffset = clamp(state.bias, -1, 1) * modAmp;
  const phase = state.phase;

  const [θA, θB, θC, θD] = MOTOR_OFFSETS_RAD;

  return {
    a: clamp(base + biasOffset + modAmp * Math.sin(phase + θA), 0, 1),
    b: clamp(base + biasOffset + modAmp * Math.sin(phase + θB), 0, 1),
    c: clamp(base + biasOffset + modAmp * Math.sin(phase + θC), 0, 1),
    d: clamp(base + biasOffset + modAmp * Math.sin(phase + θD), 0, 1),
  };
}
