/**
 * mavlink-bridge-impl.ts
 *
 * Concrete MAVLink 2 bridge over UDP.
 *
 * Packet format (MAVLink 2, §6.2):
 *   [0]     STX              0xFD
 *   [1]     len              payload length in bytes
 *   [2]     incompat_flags   0x00
 *   [3]     compat_flags     0x00
 *   [4]     seq              rolling 0–255
 *   [5]     sysid            this GCS system id
 *   [6]     compid           this GCS component id
 *   [7..9]  msgid            24-bit little-endian message id
 *   [10..N] payload          little-endian field layout (see below)
 *   [N+1,2] checksum         X25 CRC over bytes [1..N], then CRCE appended
 *
 * Messages encoded / decoded here:
 *   COMMAND_LONG               id  76   payload 33 bytes   CRCE 152
 *   COMMAND_ACK                id  77   payload 10 bytes   CRCE  43  (rx only)
 *   SET_ACTUATOR_CONTROL_TARGET id 140  payload 41 bytes   CRCE 168
 *   HEARTBEAT                  id   0   payload  9 bytes   CRCE  50  (rx only)
 *
 * Environment variables consumed (all optional):
 *   MAVLINK_HOST          Autopilot UDP host.      Default: 127.0.0.1
 *   MAVLINK_PORT          Autopilot UDP port.      Default: 14550
 *   MAVLINK_SYSID         This GCS system id.      Default: 255
 *   MAVLINK_COMPID        This GCS component id.   Default: 190
 *   MAVLINK_TARGET_SYSID  Target vehicle sysid.    Default: 1
 *   MAVLINK_TARGET_COMPID Target vehicle compid.   Default: 1
 *   MAVLINK_ACK_TIMEOUT   COMMAND_ACK timeout ms.  Default: 2000
 */

import * as dgram from 'dgram';
import { MavlinkBridge } from './mavlink-bridge';

// ─── Constants ────────────────────────────────────────────────────────────────

const STX = 0xfd;

// Message ids
const MSGID_HEARTBEAT = 0;
const MSGID_COMMAND_LONG = 76;
const MSGID_COMMAND_ACK = 77;
const MSGID_SET_ACTUATOR_CONTROL_TARGET = 140;

// MAV_RESULT values
const MAV_RESULT_ACCEPTED = 0;

// Extra CRC bytes (CRCE) — one per message definition
const CRCE: Record<number, number> = {
  [MSGID_HEARTBEAT]: 50,
  [MSGID_COMMAND_LONG]: 152,
  [MSGID_COMMAND_ACK]: 43,
  [MSGID_SET_ACTUATOR_CONTROL_TARGET]: 168,
};

// Heartbeat base_mode bit indicating vehicle is armed
const MAV_MODE_FLAG_SAFETY_ARMED = 0x80;

// ─── CRC ─────────────────────────────────────────────────────────────────────

/**
 * Compute MAVLink X25 CRC over `data`.
 * The CRC seed is 0xFFFF and the polynomial is the CCITT X25 variant.
 */
function crcX25(data: Buffer | Uint8Array): number {
  let crc = 0xffff;
  for (let i = 0; i < data.length; i++) {
    let tmp = (data[i] as number) ^ (crc & 0xff);
    tmp = (tmp ^ (tmp << 4)) & 0xff;
    crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xffff;
  }
  return crc;
}

// ─── Packet builder ───────────────────────────────────────────────────────────

/**
 * Build a complete MAVLink 2 frame.
 *
 * @param seq      Sequence number (0–255).
 * @param sysid    Sender system id.
 * @param compid   Sender component id.
 * @param msgid    24-bit message id.
 * @param payload  Already-encoded payload buffer.
 * @returns        Full frame as a Buffer including STX and checksum.
 */
function buildFrame(
  seq: number,
  sysid: number,
  compid: number,
  msgid: number,
  payload: Buffer,
): Buffer {
  const len = payload.length;
  // Header is bytes 1..9 (after STX) for CRC computation
  const header = Buffer.allocUnsafe(9);
  header[0] = len;
  header[1] = 0x00; // incompat_flags
  header[2] = 0x00; // compat_flags
  header[3] = seq & 0xff;
  header[4] = sysid & 0xff;
  header[5] = compid & 0xff;
  header[6] = msgid & 0xff;
  header[7] = (msgid >> 8) & 0xff;
  header[8] = (msgid >> 16) & 0xff;

  // CRC covers header + payload + CRCE
  const crceVal = CRCE[msgid];
  if (crceVal === undefined) {
    throw new Error(`buildFrame: unknown msgid ${msgid} — CRCE not registered`);
  }
  const crcInput = Buffer.concat([header, payload, Buffer.from([crceVal])]);
  const crcVal = crcX25(crcInput);

  const frame = Buffer.allocUnsafe(1 + header.length + len + 2);
  let offset = 0;
  frame[offset++] = STX;
  header.copy(frame, offset);
  offset += header.length;
  payload.copy(frame, offset);
  offset += len;
  frame[offset++] = crcVal & 0xff;
  frame[offset] = (crcVal >> 8) & 0xff;

  return frame;
}

// ─── Payload encoders ─────────────────────────────────────────────────────────

/**
 * Encode a COMMAND_LONG payload (33 bytes, little-endian).
 * Field order matches MAVLink wire definition (sorted by field type size).
 *
 * uint16  command      @ offset 28
 * float32 param1..7   @ offsets 0, 4, 8, 12, 16, 20, 24
 * uint8   target_system   @ 30
 * uint8   target_component @ 31
 * uint8   confirmation @ 32
 */
function encodeCommandLong(
  targetSystem: number,
  targetComponent: number,
  command: number,
  confirmation: number,
  param1: number,
  param2: number,
  param3: number,
  param4: number,
  param5: number,
  param6: number,
  param7: number,
): Buffer {
  const buf = Buffer.allocUnsafe(33);
  buf.writeFloatLE(param1, 0);
  buf.writeFloatLE(param2, 4);
  buf.writeFloatLE(param3, 8);
  buf.writeFloatLE(param4, 12);
  buf.writeFloatLE(param5, 16);
  buf.writeFloatLE(param6, 20);
  buf.writeFloatLE(param7, 24);
  buf.writeUInt16LE(command, 28);
  buf[30] = targetSystem & 0xff;
  buf[31] = targetComponent & 0xff;
  buf[32] = confirmation & 0xff;
  return buf;
}

/**
 * Encode a SET_ACTUATOR_CONTROL_TARGET payload (41 bytes, little-endian).
 *
 * uint64  time_usec  @ offset 0
 * float32 controls[8] @ offset 8
 * uint8   group_mlx  @ offset 40
 */
function encodeActuatorControlTarget(
  timeUsec: bigint,
  controls: number[],
  groupMix: number,
): Buffer {
  const buf = Buffer.allocUnsafe(41);
  buf.writeBigUInt64LE(timeUsec, 0);
  for (let i = 0; i < 8; i++) {
    buf.writeFloatLE(controls[i] ?? 0, 8 + i * 4);
  }
  buf[40] = groupMix & 0xff;
  return buf;
}

// ─── Partial packet parser ────────────────────────────────────────────────────

type ParsedFrame = {
  msgid: number;
  payload: Buffer;
};

/**
 * Attempt to extract complete MAVLink 2 frames from `buf`.
 * Returns an array of successfully decoded frames and the unconsumed tail.
 */
function parseFrames(buf: Buffer): { frames: ParsedFrame[]; remainder: Buffer } {
  const frames: ParsedFrame[] = [];
  let pos = 0;

  while (pos < buf.length) {
    // Find next STX
    if (buf[pos] !== STX) {
      pos++;
      continue;
    }
    // Need at least 12 bytes for header + empty payload + 2-byte CRC
    if (buf.length - pos < 12) break;

    const len = buf[pos + 1] as number;
    const frameLen = 12 + len; // STX(1) + header(9) + payload(len) + CRC(2)
    if (buf.length - pos < frameLen) break;

    const msgid =
      (buf[pos + 7] as number) |
      ((buf[pos + 8] as number) << 8) |
      ((buf[pos + 9] as number) << 16);

    const crceVal = CRCE[msgid];
    if (crceVal !== undefined) {
      // Validate checksum: CRC over header(9) + payload(len) + CRCE
      const crcInput = Buffer.concat([
        buf.subarray(pos + 1, pos + 10 + len),
        Buffer.from([crceVal]),
      ]);
      const expected = crcX25(crcInput);
      const got =
        (buf[pos + 10 + len] as number) |
        ((buf[pos + 11 + len] as number) << 8);

      if (expected === got) {
        frames.push({
          msgid,
          payload: Buffer.from(buf.subarray(pos + 10, pos + 10 + len)),
        });
      }
    }

    pos += frameLen;
  }

  return { frames, remainder: Buffer.from(buf.subarray(pos)) };
}

// ─── Bridge implementation ────────────────────────────────────────────────────

type PendingAck = {
  command: number;
  resolve: (accepted: boolean) => void;
  timer: ReturnType<typeof setTimeout>;
};

export class MavlinkBridgeImpl implements MavlinkBridge {
  private readonly host: string;
  private readonly port: number;
  private readonly sysid: number;
  private readonly compid: number;
  private readonly targetSysid: number;
  private readonly targetCompid: number;
  private readonly ackTimeoutMs: number;

  private socket: dgram.Socket | null = null;
  private seq = 0;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private rxBuf: Buffer<any> = Buffer.alloc(0);
  private armed = false;
  private pendingAcks: PendingAck[] = [];

  constructor() {
    this.host = process.env['MAVLINK_HOST'] ?? '127.0.0.1';
    this.port = parseInt(process.env['MAVLINK_PORT'] ?? '14550', 10);
    this.sysid = parseInt(process.env['MAVLINK_SYSID'] ?? '255', 10);
    this.compid = parseInt(process.env['MAVLINK_COMPID'] ?? '190', 10);
    this.targetSysid = parseInt(process.env['MAVLINK_TARGET_SYSID'] ?? '1', 10);
    this.targetCompid = parseInt(
      process.env['MAVLINK_TARGET_COMPID'] ?? '1',
      10,
    );
    this.ackTimeoutMs = parseInt(
      process.env['MAVLINK_ACK_TIMEOUT'] ?? '2000',
      10,
    );
  }

  async connect(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const sock = dgram.createSocket('udp4');

      sock.on('error', (err) => {
        if (this.socket === null) {
          reject(err);
        }
      });

      sock.on('message', (msg) => {
        this.rxBuf = Buffer.concat([this.rxBuf, msg]);
        const { frames, remainder } = parseFrames(this.rxBuf);
        this.rxBuf = remainder;
        for (const frame of frames) {
          this.handleFrame(frame);
        }
      });

      sock.bind(0, () => {
        this.socket = sock;
        resolve();
      });
    });
  }

  async close(): Promise<void> {
    return new Promise<void>((resolve) => {
      if (this.socket === null) {
        resolve();
        return;
      }
      this.socket.close(() => {
        this.socket = null;
        resolve();
      });
    });
  }

  async sendCommandLong(
    command: number,
    param1: number,
    param2: number,
    param3: number,
    param4: number,
    param5: number,
    param6: number,
    param7: number,
  ): Promise<boolean> {
    const payload = encodeCommandLong(
      this.targetSysid,
      this.targetCompid,
      command,
      0, // confirmation
      param1,
      param2,
      param3,
      param4,
      param5,
      param6,
      param7,
    );
    const frame = buildFrame(
      this.nextSeq(),
      this.sysid,
      this.compid,
      MSGID_COMMAND_LONG,
      payload,
    );

    return new Promise<boolean>((resolve) => {
      const timer = setTimeout(() => {
        this.removePendingAck(command);
        resolve(false);
      }, this.ackTimeoutMs);

      this.pendingAcks.push({ command, resolve, timer });
      this.send(frame);
    });
  }

  async sendActuatorControlTarget(
    groupMix: number,
    controls: number[],
  ): Promise<void> {
    if (controls.length !== 8) {
      throw new Error(
        `sendActuatorControlTarget expects 8 controls, got ${controls.length}`,
      );
    }
    const timeUsec = BigInt(Date.now()) * 1000n;
    const payload = encodeActuatorControlTarget(timeUsec, controls, groupMix);
    const frame = buildFrame(
      this.nextSeq(),
      this.sysid,
      this.compid,
      MSGID_SET_ACTUATOR_CONTROL_TARGET,
      payload,
    );
    this.send(frame);
  }

  async getArmedStatus(): Promise<boolean> {
    return this.armed;
  }

  // ─── Internals ──────────────────────────────────────────────────────────────

  private nextSeq(): number {
    const s = this.seq;
    this.seq = (this.seq + 1) & 0xff;
    return s;
  }

  private send(frame: Buffer): void {
    if (this.socket === null) {
      throw new Error('MavlinkBridgeImpl: not connected — call connect() first');
    }
    this.socket.send(frame, this.port, this.host);
  }

  private handleFrame(frame: ParsedFrame): void {
    if (frame.msgid === MSGID_HEARTBEAT) {
      this.handleHeartbeat(frame.payload);
    } else if (frame.msgid === MSGID_COMMAND_ACK) {
      this.handleCommandAck(frame.payload);
    }
  }

  /**
   * Parse HEARTBEAT and update armed status.
   * base_mode byte is at offset 6; bit 7 = MAV_MODE_FLAG_SAFETY_ARMED.
   */
  private handleHeartbeat(payload: Buffer): void {
    if (payload.length < 9) return;
    const baseMode = payload[6] as number;
    this.armed = (baseMode & MAV_MODE_FLAG_SAFETY_ARMED) !== 0;
  }

  /**
   * Parse COMMAND_ACK and resolve the matching pending promise.
   * COMMAND_ACK layout:
   *   uint16 command  @ 0
   *   uint8  result   @ 2
   */
  private handleCommandAck(payload: Buffer): void {
    if (payload.length < 3) return;
    const command = payload.readUInt16LE(0);
    const result = payload[2] as number;

    const idx = this.pendingAcks.findIndex((p) => p.command === command);
    if (idx === -1) return;

    const pending = this.pendingAcks.splice(idx, 1)[0];
    if (pending === undefined) return;
    clearTimeout(pending.timer);
    pending.resolve(result === MAV_RESULT_ACCEPTED);
  }

  private removePendingAck(command: number): void {
    const idx = this.pendingAcks.findIndex((p) => p.command === command);
    if (idx !== -1) {
      this.pendingAcks.splice(idx, 1);
    }
  }
}
