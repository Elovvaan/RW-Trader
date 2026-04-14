# Field Mode — Version-1 Tangential-Field Control

## Overview

Version 1 of the lynchpin drone replaces the flat equal-throttle quad model
with a *geometry-driven tangential-field* control model.  Instead of sending
one average-lift value to all four motors, the propulsion controller now
computes a per-motor output derived from a rotating sinusoidal field that
spans the physical node geometry of the drone.

---

## Control Model

### Legacy avgLift mode (unchanged default)

```
setThrottle(avgLift)
  └─► link.setThrottle(avgLift)
        └─► all 4 motors receive the same normalised throttle
```

`avgLift` semantics remain:

| throttle range | flight-state meaning |
|----------------|----------------------|
| 0.0 – 0.3      | land / idle          |
| 0.3 – 0.7      | loiter / hold        |
| 0.7 – 1.0      | takeoff / climb      |

### Version-1 field mode (`FIELD_MODE_ENABLED=true`)

```
FieldState { intensity, phase, phaseVelocity, spin, bias, enabled }
  └─► solveField(fieldState)
        └─► MotorOutputs { a, b, c, d }
              └─► link.setActuatorOutputs([a, b, c, d])
```

#### Per-motor formula

Four motors A/B/C/D sit at 90° spacing.  Their angular offsets are:

| Motor | Position    | Offset θ  |
|-------|-------------|-----------|
| A     | front-left  | 0°        |
| B     | front-right | 90°       |
| C     | rear-right  | 180°      |
| D     | rear-left   | 270°      |

Each motor output is:

```
base      = intensity / 100          (normalise 0..100 → 0..1)
modAmp    = base × 0.30              (modulation depth: 30 % of base)
biasOff   = clamp(bias, -1, 1) × modAmp

output_i  = clamp( base + biasOff + modAmp × sin(phase + θᵢ),  0,  1 )
```

Because `sin` terms sum to zero over all four offsets, total thrust equals
`4 × base` when `bias = 0` — phase rotation does not waste energy.

---

## FieldState fields

| Field          | Type        | Range     | Description                                         |
|----------------|-------------|-----------|-----------------------------------------------------|
| `intensity`    | `number`    | 0 .. 100  | Normalised master lift / energy level.              |
| `phase`        | `number`    | radians   | Current phase angle (advanced by `tick()`).         |
| `phaseVelocity`| `number`    | rad/s     | Phase advance per second.                           |
| `spin`         | `1 \| -1`   | —         | `1` = CCW (positive phase advance), `-1` = CW.      |
| `bias`         | `number`    | -1 .. 1   | −1 = full contraction, +1 = full expansion.         |
| `enabled`      | `boolean`   | —         | `false` → fall back to equal-output flat mode.      |

---

## Environment variables

All variables are optional.  Defaults are shown.

```
FIELD_MODE_ENABLED=false          # true|1 to enable field mode
FIELD_PHASE_VELOCITY=6.2832       # radians/s (default: 2π = 1 rot/s)
FIELD_SPIN=1                      # 1 (CCW) or -1 (CW)
FIELD_OUTPUT_SCALE=1.0            # linear scale on solver outputs
FIELD_OUTPUT_MIN=0.0              # lower clamp after scaling
FIELD_OUTPUT_MAX=1.0              # upper clamp after scaling
```

Full list of all env vars (including MAVLink bridge settings) is in
[`.env.example`](../.env.example).

---

## Sim vs hardware paths

Both paths share the same `PropulsionController` and `solveField()` solver.
Only the `FlightControllerLink` adapter differs.

### Sim (`RUNTIME=sim`)

`SimAdapter` logs every command to stdout:

```
[SIM] ARM
[SIM] TAKEOFF → 10.0 m
[SIM] SET_ACTUATORS A=0.4985 B=0.6485 C=0.5015 D=0.3515
```

No hardware or network interaction occurs.  Safe to use in unit tests.

### Hardware (`RUNTIME=hardware`)

`MavlinkAdapter` + `MavlinkBridgeImpl` communicate with the autopilot over UDP.

**Direct motor control path:**

1. `MAV_CMD_DO_SET_ACTUATOR` (id 187) is sent as a `COMMAND_LONG`.
   - `param1–param4` carry motor A–D outputs (normalised 0..1).
   - `param5 = -1`, `param6 = -1` (unused slots, no-change sentinel).
   - `param7 = 0` (actuator group 0, main motors).
2. If the autopilot does not ACK within `MAVLINK_ACK_TIMEOUT` ms,
   `SET_ACTUATOR_CONTROL_TARGET` (message 140) is broadcast as a fallback.
   - `controls[0..3]` = motors A–D; `controls[4..7] = 0`.
   - `group_mlx = 0` (`group_mlx` is the official MAVLink wire-field name for the actuator group selector).

**Arming / disarming / takeoff / land** use `COMMAND_LONG` with the same
command ids as before — these paths are unchanged.

### Hardware limitations

- `MAV_CMD_DO_SET_ACTUATOR` requires MAVLink 2 and a firmware that supports
  direct actuator overrides (ArduCopter ≥ 4.3 or PX4 ≥ 1.14).
- In SITL (`--sitl` ArduPilot flag) the command is accepted but the effect
  on simulated motors depends on the actuator-mapping configuration.
- If the autopilot requires pre-arm safety checks to pass before accepting
  actuator writes, arm the vehicle first (`armAndTakeoff()`).

---

## Coexistence with legacy behavior

- `setThrottle(avgLift)` still works exactly as before when
  `FIELD_MODE_ENABLED=false`.
- When field mode is enabled, `setThrottle(t)` maps `t × 100` to the field
  state intensity so existing high-level throttle callers continue to work.
- `armAndTakeoff()` and `landAndDisarm()` are unaffected by field mode.

---

## Example usage

```typescript
import { PropulsionController } from './propulsion-controller';
import { SimAdapter } from './adapters/sim-adapter';

const link = new SimAdapter();
const ctrl = new PropulsionController(link);

await ctrl.armAndTakeoff(10);                       // arm + takeoff to 10 m

await ctrl.setFieldState({                          // enable field mode
  enabled: true,
  intensity: 65,
  phaseVelocity: 2 * Math.PI,                       // 1 rot/s
  spin: 1,
  bias: 0,
});

// Flight loop — call every ~50 ms
const tickInterval = setInterval(() => ctrl.tick(), 50);

// … later …
clearInterval(tickInterval);
await ctrl.landAndDisarm();
```
