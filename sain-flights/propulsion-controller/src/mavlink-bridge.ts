/**
 * mavlink-bridge.ts
 *
 * Interface for the low-level MAVLink communication bridge.
 *
 * Concrete implementations wrap the actual transport (UDP/serial) and handle
 * MAVLink 2 packet framing, sequencing, and ACK tracking.  The interface
 * deliberately exposes only the primitives the propulsion stack needs so
 * that the adapters remain testable against a mock bridge.
 */

export interface MavlinkBridge {
  /**
   * Send a COMMAND_LONG (message id 76) to the autopilot.
   *
   * @param command    MAV_CMD command id.
   * @param param1..7  Command parameters (unused params should be 0).
   * @returns          True when the autopilot ACKed with MAV_RESULT_ACCEPTED,
   *                   false when it NACKed or the ACK timed out.
   */
  sendCommandLong(
    command: number,
    param1: number,
    param2: number,
    param3: number,
    param4: number,
    param5: number,
    param6: number,
    param7: number,
  ): Promise<boolean>;

  /**
   * Send a SET_ACTUATOR_CONTROL_TARGET (message id 140) broadcast.
   *
   * Used as the fallback actuator-write path when MAV_CMD_DO_SET_ACTUATOR
   * is not acknowledged.
   *
   * @param groupMix  Actuator group: 0 = main motors, 1 = flaps, etc.
   * @param controls  Exactly 8 normalised control values in [0..1].
   *                  Indices 0–3 map to the four main motors.
   */
  sendActuatorControlTarget(groupMix: number, controls: number[]): Promise<void>;

  /**
   * Return the current armed status derived from the most recent HEARTBEAT.
   * Returns false if no heartbeat has been received yet.
   */
  getArmedStatus(): Promise<boolean>;

  /** Open the transport (UDP socket / serial port) and start listening. */
  connect(): Promise<void>;

  /** Flush and close the transport. */
  close(): Promise<void>;
}
