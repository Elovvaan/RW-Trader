/**
 * adapters/sim-adapter.ts
 *
 * Simulator adapter for FlightControllerLink.
 *
 * Logs all commands to stdout.  No network or hardware interaction occurs.
 * Used when RUNTIME=sim (or during tests that do not mock the bridge).
 *
 * setActuatorOutputs logs per-motor values in the same order as the
 * tangential-field geometry: [A, B, C, D].
 */

import { FlightControllerLink } from '../flight-controller-link';

export class SimAdapter implements FlightControllerLink {
  private armed = false;

  async arm(): Promise<void> {
    this.armed = true;
    console.log('[SIM] ARM');
  }

  async disarm(): Promise<void> {
    this.armed = false;
    console.log('[SIM] DISARM');
  }

  async takeoff(altitudeMeters: number): Promise<void> {
    console.log(`[SIM] TAKEOFF → ${altitudeMeters.toFixed(1)} m`);
  }

  async land(): Promise<void> {
    console.log('[SIM] LAND');
  }

  async setThrottle(throttle: number): Promise<void> {
    const clamped = Math.max(0, Math.min(1, throttle));
    console.log(`[SIM] SET_THROTTLE → ${clamped.toFixed(4)}`);
  }

  async setActuatorOutputs(outputs: number[]): Promise<void> {
    if (outputs.length !== 4) {
      throw new Error(
        `SimAdapter.setActuatorOutputs expects 4 outputs, got ${outputs.length}`,
      );
    }
    const [a, b, c, d] = outputs.map((o) =>
      Math.max(0, Math.min(1, o)).toFixed(4),
    );
    console.log(`[SIM] SET_ACTUATORS A=${a} B=${b} C=${c} D=${d}`);
  }

  async isArmed(): Promise<boolean> {
    return this.armed;
  }
}
