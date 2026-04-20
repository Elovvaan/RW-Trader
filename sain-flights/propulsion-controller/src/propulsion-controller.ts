/**
 * propulsion-controller.ts
 *
 * Propulsion controller for the lynchpin drone.
 *
 * Version 1 extends the legacy avgLift throttle model with a
 * geometry-driven tangential-field control mode.
 *
 * Legacy path (field mode disabled):
 *   setThrottle(avgLift) → link.setThrottle()
 *
 * Field path (field mode enabled):
 *   setFieldState(partial) → solveField() → link.setActuatorOutputs()
 *   tick()                 → advancePhase() → solveField() → link.setActuatorOutputs()
 *   setThrottle(avgLift)   → updates intensity in fieldState → field path
 *
 * High-level flight-state flow (arm / takeoff / loiter / land / disarm) is
 * preserved unchanged and coexists with both actuation modes.
 */

import { FlightControllerLink } from './flight-controller-link';
import {
  FieldState,
  MotorOutputs,
  advancePhase,
  solveField,
} from './field-solver';
import { loadConfig, PropulsionConfig } from './config';

export type FlightPhase =
  | 'idle'
  | 'arming'
  | 'takeoff'
  | 'loiter'
  | 'land'
  | 'disarming';

export class PropulsionController {
  private readonly link: FlightControllerLink;
  private readonly cfg: PropulsionConfig;

  private fieldState: FieldState;
  private flightPhase: FlightPhase = 'idle';
  private lastTickMs: number | null = null;

  constructor(link: FlightControllerLink, cfg?: PropulsionConfig) {
    this.link = link;
    this.cfg = cfg ?? loadConfig();

    this.fieldState = {
      intensity: 0,
      phase: 0,
      phaseVelocity: this.cfg.defaultPhaseVelocity,
      spin: this.cfg.defaultSpin,
      bias: 0,
      enabled: this.cfg.fieldModeEnabled,
    };
  }

  // ─── High-level flight-state flow (unchanged) ─────────────────────────────

  /**
   * Arm the vehicle and command takeoff.
   * @param altitudeMeters  Target altitude in metres.
   */
  async armAndTakeoff(altitudeMeters: number): Promise<void> {
    this.flightPhase = 'arming';
    await this.link.arm();
    this.flightPhase = 'takeoff';
    await this.link.takeoff(altitudeMeters);
    this.flightPhase = 'loiter';
  }

  /** Command landing and disarm when complete. */
  async landAndDisarm(): Promise<void> {
    this.flightPhase = 'land';
    await this.link.land();
    this.flightPhase = 'disarming';
    await this.link.disarm();
    this.flightPhase = 'idle';
  }

  // ─── Legacy avgLift path ──────────────────────────────────────────────────

  /**
   * Apply a single normalised throttle value.
   *
   * When field mode is disabled this delegates directly to
   * `link.setThrottle(throttle)` (original behaviour).
   *
   * When field mode is enabled this updates the field state intensity
   * (avgLift × 100 → intensity 0..100) and drives the field path, so the
   * existing high/medium/low throttle semantics remain meaningful.
   *
   * @param throttle  Normalised throttle 0 (off) to 1 (full).
   */
  async setThrottle(throttle: number): Promise<void> {
    if (this.fieldState.enabled) {
      this.fieldState = {
        ...this.fieldState,
        intensity: Math.max(0, Math.min(1, throttle)) * 100,
      };
      await this.applyFieldState();
    } else {
      await this.link.setThrottle(throttle);
    }
  }

  // ─── Version-1 field-mode path ────────────────────────────────────────────

  /**
   * Merge a partial FieldState update and immediately apply motor outputs.
   *
   * @param update  Any subset of FieldState fields to overwrite.
   */
  async setFieldState(update: Partial<FieldState>): Promise<void> {
    this.fieldState = { ...this.fieldState, ...update };
    await this.applyFieldState();
  }

  /**
   * Advance the phase by elapsed wall-clock time and re-apply motor outputs.
   *
   * Call this periodically from the flight loop (recommended: every 50 ms).
   * Does nothing when field mode is disabled.
   */
  async tick(): Promise<void> {
    if (!this.fieldState.enabled) return;

    const now = Date.now();
    if (this.lastTickMs !== null) {
      const dt = (now - this.lastTickMs) / 1000;
      this.fieldState = {
        ...this.fieldState,
        phase: advancePhase(
          this.fieldState.phase,
          this.fieldState.phaseVelocity,
          this.fieldState.spin,
          dt,
        ),
      };
    }
    this.lastTickMs = now;
    await this.applyFieldState();
  }

  // ─── Accessors ────────────────────────────────────────────────────────────

  getFlightPhase(): FlightPhase {
    return this.flightPhase;
  }

  getFieldState(): Readonly<FieldState> {
    return { ...this.fieldState };
  }

  // ─── Internals ────────────────────────────────────────────────────────────

  /**
   * Solve the current field state and dispatch scaled motor outputs to the link.
   *
   * Outputs from solveField() are in [0, 1].  Config-driven scaling and
   * clamping are applied before dispatching so the link always receives
   * values within the configured safe operating range.
   */
  private async applyFieldState(): Promise<void> {
    const raw: MotorOutputs = solveField(this.fieldState);
    const scaled = [raw.a, raw.b, raw.c, raw.d].map((v) =>
      Math.max(
        this.cfg.outputMin,
        Math.min(this.cfg.outputMax, v * this.cfg.outputScale),
      ),
    );
    await this.link.setActuatorOutputs(scaled);
  }
}
