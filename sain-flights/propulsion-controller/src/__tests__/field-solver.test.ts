/**
 * __tests__/field-solver.test.ts
 *
 * Unit tests for the field-solver module.
 *
 * Covered:
 *   1. Output shape — solveField returns an object with a, b, c, d
 *   2. Normalisation — intensity values outside 0..100 are clamped to [0,1]
 *   3. Clamping — no output exceeds [0, 1] even with extreme bias
 *   4. CW vs CCW phase progression — advancePhase direction follows spin sign
 *   5. Equal output when field is disabled — a = b = c = d = base
 *   6. No unsafe output values at maximum energy + maximum positive bias
 *   7. Phase symmetry — at phase=0 and bias=0 the sin model is verifiable
 *   8. Total thrust preservation — sum A+B+C+D equals 4×base when bias=0
 *   9. Disabled-mode is independent of phase / phaseVelocity / spin
 *  10. Phase wraps within [0, 2π)
 */

import { advancePhase, solveField, FieldState, MotorOutputs } from '../field-solver';

// ─── Helpers ──────────────────────────────────────────────────────────────────

const TWO_PI = 2 * Math.PI;
const MODULATION_DEPTH = 0.3; // must match field-solver constant

function makeState(overrides: Partial<FieldState> = {}): FieldState {
  return {
    intensity: 50,
    phase: 0,
    phaseVelocity: TWO_PI,
    spin: 1,
    bias: 0,
    enabled: true,
    ...overrides,
  };
}

function allInRange(outputs: MotorOutputs): boolean {
  return (
    outputs.a >= 0 && outputs.a <= 1 &&
    outputs.b >= 0 && outputs.b <= 1 &&
    outputs.c >= 0 && outputs.c <= 1 &&
    outputs.d >= 0 && outputs.d <= 1
  );
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('solveField — output shape', () => {
  it('returns an object with exactly the properties a, b, c, d', () => {
    const result = solveField(makeState());
    expect(typeof result).toBe('object');
    expect(result).toHaveProperty('a');
    expect(result).toHaveProperty('b');
    expect(result).toHaveProperty('c');
    expect(result).toHaveProperty('d');
    expect(Object.keys(result).sort()).toEqual(['a', 'b', 'c', 'd']);
  });

  it('all outputs are finite numbers', () => {
    const result = solveField(makeState());
    for (const key of ['a', 'b', 'c', 'd'] as const) {
      expect(Number.isFinite(result[key])).toBe(true);
    }
  });
});

describe('solveField — normalisation and clamping', () => {
  it('intensity > 100 is clamped: outputs stay in [0, 1]', () => {
    const result = solveField(makeState({ intensity: 150 }));
    expect(allInRange(result)).toBe(true);
  });

  it('intensity < 0 is clamped: all outputs are 0', () => {
    const result = solveField(makeState({ intensity: -50 }));
    expect(result.a).toBe(0);
    expect(result.b).toBe(0);
    expect(result.c).toBe(0);
    expect(result.d).toBe(0);
  });

  it('intensity = 0 produces all-zero outputs regardless of phase', () => {
    const result = solveField(makeState({ intensity: 0, phase: 1.23 }));
    expect(result.a).toBe(0);
    expect(result.b).toBe(0);
    expect(result.c).toBe(0);
    expect(result.d).toBe(0);
  });

  it('max bias (+1) does not push any output above 1', () => {
    const result = solveField(makeState({ intensity: 100, bias: 1 }));
    expect(allInRange(result)).toBe(true);
  });

  it('min bias (−1) does not push any output below 0', () => {
    const result = solveField(makeState({ intensity: 100, bias: -1 }));
    expect(allInRange(result)).toBe(true);
  });

  it('bias > 1 is clamped to 1 before applying', () => {
    const clamped = solveField(makeState({ intensity: 50, bias: 1 }));
    const excess = solveField(makeState({ intensity: 50, bias: 999 }));
    expect(excess.a).toBeCloseTo(clamped.a);
    expect(excess.b).toBeCloseTo(clamped.b);
    expect(excess.c).toBeCloseTo(clamped.c);
    expect(excess.d).toBeCloseTo(clamped.d);
  });
});

describe('solveField — stable equal-output when field is disabled', () => {
  it('when enabled=false all four motors get base = intensity/100', () => {
    const state = makeState({ intensity: 60, enabled: false });
    const result = solveField(state);
    expect(result.a).toBeCloseTo(0.6);
    expect(result.b).toBeCloseTo(0.6);
    expect(result.c).toBeCloseTo(0.6);
    expect(result.d).toBeCloseTo(0.6);
  });

  it('disabled mode ignores phase, phaseVelocity, spin, and bias', () => {
    const base = solveField(makeState({ intensity: 40, enabled: false }));
    const varied = solveField(
      makeState({
        intensity: 40,
        enabled: false,
        phase: 2.5,
        phaseVelocity: 10,
        spin: -1,
        bias: 0.9,
      }),
    );
    expect(varied.a).toBeCloseTo(base.a);
    expect(varied.b).toBeCloseTo(base.b);
    expect(varied.c).toBeCloseTo(base.c);
    expect(varied.d).toBeCloseTo(base.d);
  });

  it('disabled outputs equal the same value computed by enabled at zero modulation', () => {
    // intensity = 0 ⇒ modAmp = 0, so enabled and disabled are the same
    const enabled = solveField(makeState({ intensity: 0, enabled: true }));
    const disabled = solveField(makeState({ intensity: 0, enabled: false }));
    expect(enabled.a).toBeCloseTo(disabled.a);
    expect(enabled.b).toBeCloseTo(disabled.b);
    expect(enabled.c).toBeCloseTo(disabled.c);
    expect(enabled.d).toBeCloseTo(disabled.d);
  });
});

describe('solveField — phase symmetry and base model', () => {
  it('at phase=0 and bias=0, motor outputs follow sin offsets (0°/90°/180°/270°)', () => {
    const state = makeState({ intensity: 50, phase: 0, bias: 0 });
    const base = 0.5;
    const modAmp = base * MODULATION_DEPTH;
    const result = solveField(state);

    // A: base + modAmp * sin(0) = 0.5 (sin 0 = 0)
    expect(result.a).toBeCloseTo(base + modAmp * Math.sin(0));
    // B: base + modAmp * sin(π/2) = base + modAmp (sin π/2 = 1)
    expect(result.b).toBeCloseTo(base + modAmp * Math.sin(Math.PI / 2));
    // C: base + modAmp * sin(π) ≈ base (sin π ≈ 0)
    expect(result.c).toBeCloseTo(base + modAmp * Math.sin(Math.PI));
    // D: base + modAmp * sin(3π/2) = base − modAmp (sin 3π/2 = −1)
    expect(result.d).toBeCloseTo(base + modAmp * Math.sin((3 * Math.PI) / 2));
  });

  it('total thrust is preserved (sum = 4×base) when bias=0', () => {
    // sin terms sum to zero around a full period
    const state = makeState({ intensity: 70, phase: 1.1, bias: 0 });
    const base = 0.7;
    const result = solveField(state);
    const sum = result.a + result.b + result.c + result.d;
    expect(sum).toBeCloseTo(4 * base, 4);
  });

  it('phase offset of π between motor A and motor C produces opposite modulation', () => {
    const state = makeState({ intensity: 60, phase: 0.8, bias: 0 });
    const result = solveField(state);
    // A and C are 180° apart, so their deviations from base should be equal & opposite
    const base = 0.6;
    const devA = result.a - base;
    const devC = result.c - base;
    expect(devA + devC).toBeCloseTo(0, 5);
  });
});

describe('solveField — no unsafe output values', () => {
  it('does not produce NaN for any standard input', () => {
    const variants: FieldState[] = [
      makeState({ intensity: 0 }),
      makeState({ intensity: 100 }),
      makeState({ intensity: 50, phase: Math.PI }),
      makeState({ intensity: 50, bias: -1 }),
      makeState({ intensity: 50, bias: 1 }),
    ];
    for (const s of variants) {
      const r = solveField(s);
      expect(Number.isNaN(r.a)).toBe(false);
      expect(Number.isNaN(r.b)).toBe(false);
      expect(Number.isNaN(r.c)).toBe(false);
      expect(Number.isNaN(r.d)).toBe(false);
    }
  });

  it('all outputs are in [0, 1] for a sweep of intensity and bias values', () => {
    for (let intensity = 0; intensity <= 100; intensity += 10) {
      for (let bias = -1; bias <= 1; bias += 0.5) {
        const result = solveField(makeState({ intensity, bias }));
        expect(allInRange(result)).toBe(true);
      }
    }
  });
});

// ─── advancePhase tests ───────────────────────────────────────────────────────

describe('advancePhase — CW vs CCW phase progression', () => {
  it('spin=+1 (CCW) advances phase forward', () => {
    const next = advancePhase(0, TWO_PI, 1, 0.25);
    // 0.25 s × 2π rad/s × (+1) = π/2
    expect(next).toBeCloseTo(Math.PI / 2, 5);
  });

  it('spin=−1 (CW) advances phase backward, wrapping to [0, 2π)', () => {
    const next = advancePhase(0, TWO_PI, -1, 0.25);
    // 0 − π/2 + 2π = 3π/2
    expect(next).toBeCloseTo((3 * Math.PI) / 2, 5);
  });

  it('CCW and CW produce different phase directions', () => {
    const ccw = advancePhase(1.0, 1.0, 1, 0.5);
    const cw = advancePhase(1.0, 1.0, -1, 0.5);
    expect(ccw).not.toBeCloseTo(cw, 5);
  });

  it('result is always in [0, 2π)', () => {
    const cases: Array<[number, number, 1 | -1, number]> = [
      [0, TWO_PI, 1, 10],         // many full rotations CCW
      [0, TWO_PI, -1, 10],        // many full rotations CW
      [6, 1, 1, 0.5],             // wraps just past 2π
      [0.1, 1, -1, 0.5],          // wraps below 0
    ];
    for (const [phase, vel, spin, dt] of cases) {
      const result = advancePhase(phase, vel, spin, dt);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(TWO_PI);
    }
  });

  it('zero phaseVelocity leaves phase unchanged', () => {
    const result = advancePhase(1.5, 0, 1, 1.0);
    expect(result).toBeCloseTo(1.5, 10);
  });

  it('zero dt leaves phase unchanged regardless of velocity', () => {
    const result = advancePhase(2.1, 5.0, 1, 0);
    expect(result).toBeCloseTo(2.1, 10);
  });
});
