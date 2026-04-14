/**
 * adapters/mavlink-adapter.ts
 *
 * MAVLink hardware adapter for FlightControllerLink.
 *
 * All MAVLink command ids used here:
 *   MAV_CMD_COMPONENT_ARM_DISARM  = 400
 *   MAV_CMD_NAV_TAKEOFF           =  22
 *   MAV_CMD_NAV_LAND              =  21
 *   MAV_CMD_DO_SET_ACTUATOR       = 187  (primary field-mode path)
 *
 * Direct motor control — setActuatorOutputs():
 * ─────────────────────────────────────────────
 * Primary:  MAV_CMD_DO_SET_ACTUATOR (id 187, MAVLink 2).
 *   param1 = motor A  (normalised 0..1)
 *   param2 = motor B
 *   param3 = motor C
 *   param4 = motor D
 *   param5 = −1       (unused actuator slot, no-change sentinel)
 *   param6 = −1       (unused actuator slot, no-change sentinel)
 *   param7 = 0        (actuator group 0 = main motors)
 *
 * Fallback: SET_ACTUATOR_CONTROL_TARGET (message id 140) when the autopilot
 *   does not ACK MAV_CMD_DO_SET_ACTUATOR within MAVLINK_ACK_TIMEOUT ms.
 *   controls[0..3] = motors A–D;  controls[4..7] = 0.
 *   group_mlx = 0 (main motor group).
 *
 * Arming/disarming, takeoff, and land are unchanged from the legacy path.
 * setThrottle (legacy avgLift mode) maps a single value to equal outputs
 * across all 4 motors via setActuatorOutputs.
 */

import { FlightControllerLink } from '../flight-controller-link';
import { MavlinkBridge } from '../mavlink-bridge';

// MAVLink command ids
const MAV_CMD_NAV_TAKEOFF = 22;
const MAV_CMD_NAV_LAND = 21;
const MAV_CMD_COMPONENT_ARM_DISARM = 400;
const MAV_CMD_DO_SET_ACTUATOR = 187;

export class MavlinkAdapter implements FlightControllerLink {
  constructor(private readonly bridge: MavlinkBridge) {}

  // ─── Safety & flight-state commands (unchanged from legacy) ───────────────

  async arm(): Promise<void> {
    // param1 = 1 → arm
    const ok = await this.bridge.sendCommandLong(
      MAV_CMD_COMPONENT_ARM_DISARM,
      1, 0, 0, 0, 0, 0, 0,
    );
    if (!ok) {
      throw new Error('MAVLink: ARM command was not acknowledged by autopilot');
    }
  }

  async disarm(): Promise<void> {
    // param1 = 0 → disarm
    const ok = await this.bridge.sendCommandLong(
      MAV_CMD_COMPONENT_ARM_DISARM,
      0, 0, 0, 0, 0, 0, 0,
    );
    if (!ok) {
      throw new Error(
        'MAVLink: DISARM command was not acknowledged by autopilot',
      );
    }
  }

  async takeoff(altitudeMeters: number): Promise<void> {
    // param7 = target altitude above ground in metres
    const ok = await this.bridge.sendCommandLong(
      MAV_CMD_NAV_TAKEOFF,
      0, 0, 0, 0, 0, 0,
      altitudeMeters,
    );
    if (!ok) {
      throw new Error(
        'MAVLink: TAKEOFF command was not acknowledged by autopilot',
      );
    }
  }

  async land(): Promise<void> {
    const ok = await this.bridge.sendCommandLong(
      MAV_CMD_NAV_LAND,
      0, 0, 0, 0, 0, 0, 0,
    );
    if (!ok) {
      throw new Error('MAVLink: LAND command was not acknowledged by autopilot');
    }
  }

  // ─── Throttle / actuator outputs ──────────────────────────────────────────

  /**
   * Legacy single-throttle path.
   * Maps one normalised value to equal outputs across all 4 motors so that
   * existing high-level state logic (takeoff/loiter/land thresholds) continues
   * to work when field mode is disabled.
   */
  async setThrottle(throttle: number): Promise<void> {
    const t = Math.max(0, Math.min(1, throttle));
    await this.setActuatorOutputs([t, t, t, t]);
  }

  /**
   * Field-mode direct motor control.
   *
   * Tries MAV_CMD_DO_SET_ACTUATOR first (cleaner, per-motor ACK path).
   * If the autopilot does not ACK within MAVLINK_ACK_TIMEOUT, falls back to
   * SET_ACTUATOR_CONTROL_TARGET (broadcast, no ACK, widely supported).
   *
   * @param outputs  Exactly 4 normalised motor outputs in [0..1], order A–D.
   */
  async setActuatorOutputs(outputs: number[]): Promise<void> {
    if (outputs.length !== 4) {
      throw new Error(
        `MavlinkAdapter.setActuatorOutputs expects 4 outputs, got ${outputs.length}`,
      );
    }

    const [a, b, c, d] = outputs.map((o) => Math.max(0, Math.min(1, o)));

    // Primary: MAV_CMD_DO_SET_ACTUATOR (id 187)
    //   param5 = −1 (no-change), param6 = −1 (no-change), param7 = 0 (group 0)
    const acked = await this.bridge.sendCommandLong(
      MAV_CMD_DO_SET_ACTUATOR,
      a,   // param1 → motor A
      b,   // param2 → motor B
      c,   // param3 → motor C
      d,   // param4 → motor D
      -1,  // param5 → unused
      -1,  // param6 → unused
      0,   // param7 → actuator group 0
    );

    if (!acked) {
      // Fallback: SET_ACTUATOR_CONTROL_TARGET (message 140)
      // 8-element controls array: indices 0–3 = motors A–D, 4–7 = unused (0).
      await this.bridge.sendActuatorControlTarget(0, [a, b, c, d, 0, 0, 0, 0]);
    }
  }

  async isArmed(): Promise<boolean> {
    return this.bridge.getArmedStatus();
  }
}
