/**
 * config.ts
 *
 * Environment-driven configuration for the propulsion controller.
 * All new field-mode settings follow the same parse-and-validate pattern
 * used throughout the rest of the system.
 *
 * Environment variables
 * ─────────────────────
 * FIELD_MODE_ENABLED      "true"|"1" to enable Version-1 field control.  Default: false.
 * FIELD_PHASE_VELOCITY    Phase advance in radians/s.  Default: 2π (one full rotation/s).
 * FIELD_SPIN              "1" = CCW, "-1" = CW.  Default: "1".
 * FIELD_OUTPUT_SCALE      Linear scale applied to [0..1] solver outputs before dispatch.
 *                         Default: 1.0 (identity).
 * FIELD_OUTPUT_MIN        Lower clamp after scaling.  Default: 0.0.
 * FIELD_OUTPUT_MAX        Upper clamp after scaling.  Default: 1.0.
 */

export type PropulsionConfig = {
  /** Enable geometry-driven field control mode (Version 1). */
  fieldModeEnabled: boolean;

  /**
   * Default phase advance in radians per second.
   * One full CCW rotation per second = 2π ≈ 6.283 rad/s.
   */
  defaultPhaseVelocity: number;

  /** Default spin direction: 1 = CCW, −1 = CW. */
  defaultSpin: 1 | -1;

  /**
   * Linear output scale factor applied after solver clamping.
   * Useful when downstream hardware expects a different unit range.
   * Default: 1.0 (no scaling; outputs remain in [0..1]).
   */
  outputScale: number;

  /**
   * Minimum motor output after scaling.
   * Prevents outputs from going below a safe idle threshold.
   * Default: 0.0.
   */
  outputMin: number;

  /**
   * Maximum motor output after scaling.
   * Hard ceiling for motor commands sent to the link.
   * Default: 1.0.
   */
  outputMax: number;
};

// ─── Parsers ──────────────────────────────────────────────────────────────────

function parseFloatEnv(key: string, fallback: number): number {
  const raw = process.env[key];
  if (raw === undefined || raw === '') return fallback;
  const parsed = parseFloat(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Config error: ${key}="${raw}" is not a valid finite number`);
  }
  return parsed;
}

function parseSpinEnv(key: string, fallback: 1 | -1): 1 | -1 {
  const raw = process.env[key];
  if (raw === undefined || raw === '') return fallback;
  if (raw === '1') return 1;
  if (raw === '-1') return -1;
  throw new Error(`Config error: ${key}="${raw}" must be "1" (CCW) or "-1" (CW)`);
}

function parseBoolEnv(key: string, fallback: boolean): boolean {
  const raw = process.env[key];
  if (raw === undefined || raw === '') return fallback;
  return raw === '1' || raw.toLowerCase() === 'true';
}

// ─── Loader ───────────────────────────────────────────────────────────────────

/**
 * Build a PropulsionConfig from the current process environment.
 * Throws a descriptive Error if any value fails to parse.
 */
export function loadConfig(): PropulsionConfig {
  const cfg: PropulsionConfig = {
    fieldModeEnabled:     parseBoolEnv('FIELD_MODE_ENABLED', false),
    defaultPhaseVelocity: parseFloatEnv('FIELD_PHASE_VELOCITY', 2 * Math.PI),
    defaultSpin:          parseSpinEnv('FIELD_SPIN', 1),
    outputScale:          parseFloatEnv('FIELD_OUTPUT_SCALE', 1.0),
    outputMin:            parseFloatEnv('FIELD_OUTPUT_MIN', 0.0),
    outputMax:            parseFloatEnv('FIELD_OUTPUT_MAX', 1.0),
  };

  if (cfg.outputMin > cfg.outputMax) {
    throw new Error(
      `Config error: FIELD_OUTPUT_MIN (${cfg.outputMin}) must be ≤ FIELD_OUTPUT_MAX (${cfg.outputMax})`,
    );
  }
  if (cfg.outputScale <= 0) {
    throw new Error(
      `Config error: FIELD_OUTPUT_SCALE (${cfg.outputScale}) must be > 0`,
    );
  }

  return cfg;
}
