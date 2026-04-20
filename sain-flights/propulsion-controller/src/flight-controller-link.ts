/**
 * flight-controller-link.ts
 *
 * Abstract interface for all flight-controller hardware/sim adapters.
 *
 * Existing callers depend on arm/disarm/takeoff/land/setThrottle.
 * Version-1 field-mode adds setActuatorOutputs as an additive extension:
 * adapters that do not yet implement it may throw or fall back gracefully.
 */

export interface FlightControllerLink {
  /** Arm the vehicle. */
  arm(): Promise<void>;

  /** Disarm the vehicle. */
  disarm(): Promise<void>;

  /**
   * Command takeoff to a target altitude.
   * @param altitudeMeters Target altitude above ground in metres.
   */
  takeoff(altitudeMeters: number): Promise<void>;

  /** Command landing at the current position. */
  land(): Promise<void>;

  /**
   * Set a single average throttle (legacy avgLift mode).
   *
   * Maps a normalised 0..1 throttle to the underlying actuator pathway.
   * This path is preserved when field mode is disabled.
   *
   * @param throttle Normalised throttle 0 (off) to 1 (full).
   */
  setThrottle(throttle: number): Promise<void>;

  /**
   * Set direct per-actuator outputs for field-mode control.
   *
   * Outputs are normalised to [0, 1] where 1 represents full throttle.
   * Index order matches the 4-node motor geometry: [A, B, C, D].
   *
   * This method is additive — it does not exist in the legacy interface
   * and must not break callers that do not use it.  Adapters that cannot
   * support direct actuator writes should throw an Error explaining why.
   *
   * @param outputs Array of exactly 4 normalised motor outputs.
   */
  setActuatorOutputs(outputs: number[]): Promise<void>;

  /** Return the current armed status. */
  isArmed(): Promise<boolean>;
}
