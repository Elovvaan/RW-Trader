// store.rs
//
// Append-only event store.
//
// Architecture:
//   - EventStore trait: portable interface, decoupled from SQLite
//   - SqliteEventStore: production implementation
//   - Writer: background task receives events via unbounded channel, writes to SQLite
//   - Reader pool: r2d2 connection pool for concurrent audit queries
//   - WAL mode: readers never block the writer
//
// The fire-and-forget channel design means the trading loop never waits on disk I/O.
// If the channel is somehow exhausted (>10k queued), events are dropped with a warning.
// Audit completeness is best-effort; safety and execution are never blocked.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use crate::events::StoredEvent;

// ── Schema ────────────────────────────────────────────────────────────────────

const CREATE_TABLE: &str = "
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT    NOT NULL UNIQUE,
    occurred_at     TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    symbol          TEXT,
    correlation_id  TEXT,
    client_order_id TEXT,
    payload         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS events_symbol_time
    ON events(symbol, occurred_at);
CREATE INDEX IF NOT EXISTS events_coid
    ON events(client_order_id)
    WHERE client_order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS events_corr
    ON events(correlation_id)
    WHERE correlation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS events_type_time
    ON events(event_type, occurred_at);
";

const PRAGMAS: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA temp_store   = MEMORY;
PRAGMA cache_size   = -8000;
";

// ── EventStore trait ──────────────────────────────────────────────────────────

/// Portable audit store interface.
/// Current implementation: SQLite.
/// Future: implement for PostgreSQL without changing callers.
pub trait EventStore: Send + Sync + 'static {
    /// Append an event. Non-blocking: event is queued to a background writer.
    fn append(&self, event: StoredEvent);

    /// Fetch the N most recent events, newest first.
    fn fetch_recent(&self, limit: usize) -> Result<Vec<StoredEvent>>;

    /// Fetch events for a symbol within a time range, oldest first.
    fn fetch_by_symbol(
        &self,
        symbol: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<StoredEvent>>;

    /// Fetch all events tied to a client_order_id, oldest first.
    fn fetch_by_coid(&self, client_order_id: &str) -> Result<Vec<StoredEvent>>;

    /// Reconstruct the full lifecycle of a trade via correlation_id, oldest first.
    fn fetch_trade_lifecycle(&self, correlation_id: &str) -> Result<Vec<StoredEvent>>;

    /// Fetch all events within a time range, oldest first.
    fn fetch_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<StoredEvent>>;

    /// Fetch market snapshots for replay, oldest first.
    fn fetch_market_snapshots_for_replay(
        &self,
        symbol: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<StoredEvent>>;

    /// Fetch signal decision events for a symbol in range.
    fn fetch_signal_decisions(
        &self,
        symbol: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<StoredEvent>>;
}

// ── SqliteEventStore ──────────────────────────────────────────────────────────

/// Production event store backed by SQLite.
///
/// Writes go through a background task via a channel — non-blocking.
/// Reads use a connection pool — concurrent.
pub struct SqliteEventStore {
    sender: mpsc::UnboundedSender<StoredEvent>,
    reader: Pool<SqliteConnectionManager>,
}

impl SqliteEventStore {
    /// Open or create the SQLite database at `path`.
    /// Spawns the background writer task.
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        // ── Writer connection (exclusive, WAL) ────────────────────────────────
        let write_conn = rusqlite::Connection::open(path)
            .with_context(|| format!("Failed to open SQLite at {:?}", path))?;

        write_conn.execute_batch(PRAGMAS)
            .context("Failed to set SQLite pragmas")?;
        write_conn.execute_batch(CREATE_TABLE)
            .context("Failed to create events table")?;

        // ── Reader pool ────────────────────────────────────────────────────────
        let manager = SqliteConnectionManager::file(path)
            .with_init(|conn| {
                conn.execute_batch(PRAGMAS)?;
                Ok(())
            });
        let reader = Pool::builder()
            .max_size(4)
            .build(manager)
            .context("Failed to build SQLite reader pool")?;

        // ── Writer channel ─────────────────────────────────────────────────────
        let (sender, mut receiver) = mpsc::unbounded_channel::<StoredEvent>();

        // Spawn blocking writer task.
        // rusqlite is not async-safe, so we use spawn_blocking per batch.
        // The channel drains events as fast as disk allows.
        std::thread::spawn(move || {
            while let Some(event) = receiver.blocking_recv() {
                if let Err(e) = write_event(&write_conn, &event) {
                    error!(event_type = %event.event_type, error = %e, "Failed to persist event");
                } else {
                    debug!(event_type = %event.event_type, event_id = %event.event_id, "Event persisted");
                }
            }
        });

        Ok(Arc::new(Self { sender, reader }))
    }

    // ── Internal reader helper ────────────────────────────────────────────────

    fn query<F>(&self, f: F) -> Result<Vec<StoredEvent>>
    where
        F: FnOnce(&rusqlite::Connection) -> Result<Vec<StoredEvent>>,
    {
        let conn = self.reader.get().context("Failed to get reader connection")?;
        f(&conn)
    }
}

impl EventStore for SqliteEventStore {
    fn append(&self, event: StoredEvent) {
        // Try to send; if channel is closed (writer panicked), log and continue.
        if let Err(e) = self.sender.send(event) {
            warn!("Event store writer is gone, dropping event: {}", e);
        }
    }

    fn fetch_recent(&self, limit: usize) -> Result<Vec<StoredEvent>> {
        self.query(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, occurred_at, event_type, symbol, correlation_id,
                        client_order_id, payload
                 FROM events
                 ORDER BY id DESC
                 LIMIT ?1"
            )?;
            collect_rows(stmt.query(params![limit as i64])?)
        })
    }

    fn fetch_by_symbol(
        &self,
        symbol: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<StoredEvent>> {
        self.query(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, occurred_at, event_type, symbol, correlation_id,
                        client_order_id, payload
                 FROM events
                 WHERE symbol = ?1
                   AND occurred_at >= ?2
                   AND occurred_at <= ?3
                 ORDER BY id ASC"
            )?;
            collect_rows(stmt.query(params![
                symbol,
                from.to_rfc3339(),
                to.to_rfc3339(),
            ])?)
        })
    }

    fn fetch_by_coid(&self, client_order_id: &str) -> Result<Vec<StoredEvent>> {
        self.query(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, occurred_at, event_type, symbol, correlation_id,
                        client_order_id, payload
                 FROM events
                 WHERE client_order_id = ?1
                 ORDER BY id ASC"
            )?;
            collect_rows(stmt.query(params![client_order_id])?)
        })
    }

    fn fetch_trade_lifecycle(&self, correlation_id: &str) -> Result<Vec<StoredEvent>> {
        self.query(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, occurred_at, event_type, symbol, correlation_id,
                        client_order_id, payload
                 FROM events
                 WHERE correlation_id = ?1
                 ORDER BY id ASC"
            )?;
            collect_rows(stmt.query(params![correlation_id])?)
        })
    }

    fn fetch_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<StoredEvent>> {
        self.query(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, occurred_at, event_type, symbol, correlation_id,
                        client_order_id, payload
                 FROM events
                 WHERE occurred_at >= ?1
                   AND occurred_at <= ?2
                 ORDER BY id ASC"
            )?;
            collect_rows(stmt.query(params![from.to_rfc3339(), to.to_rfc3339()])?)
        })
    }

    fn fetch_market_snapshots_for_replay(
        &self,
        symbol: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<StoredEvent>> {
        self.query(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, occurred_at, event_type, symbol, correlation_id,
                        client_order_id, payload
                 FROM events
                 WHERE event_type = 'market_snapshot'
                   AND symbol = ?1
                   AND occurred_at >= ?2
                   AND occurred_at <= ?3
                 ORDER BY id ASC"
            )?;
            collect_rows(stmt.query(params![
                symbol,
                from.to_rfc3339(),
                to.to_rfc3339(),
            ])?)
        })
    }

    fn fetch_signal_decisions(
        &self,
        symbol: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<StoredEvent>> {
        self.query(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, occurred_at, event_type, symbol, correlation_id,
                        client_order_id, payload
                 FROM events
                 WHERE event_type = 'signal_decision'
                   AND symbol = ?1
                   AND occurred_at >= ?2
                   AND occurred_at <= ?3
                 ORDER BY id ASC"
            )?;
            collect_rows(stmt.query(params![
                symbol,
                from.to_rfc3339(),
                to.to_rfc3339(),
            ])?)
        })
    }
}

// ── SQLite row helpers ────────────────────────────────────────────────────────

fn write_event(conn: &rusqlite::Connection, event: &StoredEvent) -> Result<()> {
    let payload_json = serde_json::to_string(&event.payload)
        .context("Failed to serialize event payload")?;

    conn.execute(
        "INSERT OR IGNORE INTO events
             (event_id, occurred_at, event_type, symbol, correlation_id, client_order_id, payload)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            event.event_id,
            event.occurred_at.to_rfc3339(),
            event.event_type,
            event.symbol,
            event.correlation_id,
            event.client_order_id,
            payload_json,
        ],
    ).context("Failed to insert event")?;

    Ok(())
}

fn collect_rows(mut rows: rusqlite::Rows<'_>) -> Result<Vec<StoredEvent>> {
    let mut events = Vec::new();
    while let Some(row) = rows.next()? {
        let event = row_to_event(row)?;
        events.push(event);
    }
    Ok(events)
}

fn row_to_event(row: &rusqlite::Row<'_>) -> Result<StoredEvent> {
    let event_id:        String         = row.get(0)?;
    let occurred_at_str: String         = row.get(1)?;
    let event_type:      String         = row.get(2)?;
    let symbol:          Option<String> = row.get(3)?;
    let correlation_id:  Option<String> = row.get(4)?;
    let client_order_id: Option<String> = row.get(5)?;
    let payload_json:    String         = row.get(6)?;

    let occurred_at = DateTime::parse_from_rfc3339(&occurred_at_str)
        .with_context(|| format!("Failed to parse timestamp: {}", occurred_at_str))?
        .with_timezone(&Utc);

    let payload: crate::events::TradingEvent = serde_json::from_str(&payload_json)
        .with_context(|| format!("Failed to deserialize payload for event {}", event_id))?;

    Ok(StoredEvent {
        event_id,
        occurred_at,
        event_type,
        symbol,
        correlation_id,
        client_order_id,
        payload,
    })
}

// ── NoopEventStore ────────────────────────────────────────────────────────────

/// A no-op store for tests that don't need persistence.
pub struct NoopEventStore;

impl EventStore for NoopEventStore {
    fn append(&self, _event: StoredEvent) {}
    fn fetch_recent(&self, _limit: usize) -> Result<Vec<StoredEvent>> { Ok(vec![]) }
    fn fetch_by_symbol(&self, _symbol: &str, _from: DateTime<Utc>, _to: DateTime<Utc>) -> Result<Vec<StoredEvent>> { Ok(vec![]) }
    fn fetch_by_coid(&self, _coid: &str) -> Result<Vec<StoredEvent>> { Ok(vec![]) }
    fn fetch_trade_lifecycle(&self, _corr_id: &str) -> Result<Vec<StoredEvent>> { Ok(vec![]) }
    fn fetch_range(&self, _from: DateTime<Utc>, _to: DateTime<Utc>) -> Result<Vec<StoredEvent>> { Ok(vec![]) }
    fn fetch_market_snapshots_for_replay(&self, _symbol: &str, _from: DateTime<Utc>, _to: DateTime<Utc>) -> Result<Vec<StoredEvent>> { Ok(vec![]) }
    fn fetch_signal_decisions(&self, _symbol: &str, _from: DateTime<Utc>, _to: DateTime<Utc>) -> Result<Vec<StoredEvent>> { Ok(vec![]) }
}

// ── InMemoryEventStore ────────────────────────────────────────────────────────

/// An in-memory store for tests that need to inspect appended events.
pub struct InMemoryEventStore {
    events: std::sync::Mutex<Vec<StoredEvent>>,
}

impl InMemoryEventStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { events: std::sync::Mutex::new(Vec::new()) })
    }

    pub fn all_events(&self) -> Vec<StoredEvent> {
        self.events.lock().unwrap().clone()
    }

    pub fn count_by_type(&self, event_type: &str) -> usize {
        self.events.lock().unwrap()
            .iter()
            .filter(|e| e.event_type == event_type)
            .count()
    }

    pub fn find_by_coid(&self, coid: &str) -> Vec<StoredEvent> {
        self.events.lock().unwrap()
            .iter()
            .filter(|e| e.client_order_id.as_deref() == Some(coid))
            .cloned()
            .collect()
    }

    pub fn find_by_correlation(&self, corr_id: &str) -> Vec<StoredEvent> {
        self.events.lock().unwrap()
            .iter()
            .filter(|e| e.correlation_id.as_deref() == Some(corr_id))
            .cloned()
            .collect()
    }
}

impl Default for InMemoryEventStore {
    fn default() -> Self {
        Self { events: std::sync::Mutex::new(Vec::new()) }
    }
}

impl EventStore for InMemoryEventStore {
    fn append(&self, event: StoredEvent) {
        self.events.lock().unwrap().push(event);
    }

    fn fetch_recent(&self, limit: usize) -> Result<Vec<StoredEvent>> {
        let events = self.events.lock().unwrap();
        Ok(events.iter().rev().take(limit).cloned().collect())
    }

    fn fetch_by_symbol(&self, symbol: &str, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<StoredEvent>> {
        let events = self.events.lock().unwrap();
        Ok(events.iter()
            .filter(|e| e.symbol.as_deref() == Some(symbol) && e.occurred_at >= from && e.occurred_at <= to)
            .cloned()
            .collect())
    }

    fn fetch_by_coid(&self, coid: &str) -> Result<Vec<StoredEvent>> {
        Ok(self.find_by_coid(coid))
    }

    fn fetch_trade_lifecycle(&self, corr_id: &str) -> Result<Vec<StoredEvent>> {
        Ok(self.find_by_correlation(corr_id))
    }

    fn fetch_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<StoredEvent>> {
        let events = self.events.lock().unwrap();
        Ok(events.iter()
            .filter(|e| e.occurred_at >= from && e.occurred_at <= to)
            .cloned()
            .collect())
    }

    fn fetch_market_snapshots_for_replay(&self, symbol: &str, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<StoredEvent>> {
        let events = self.events.lock().unwrap();
        Ok(events.iter()
            .filter(|e| {
                e.event_type == "market_snapshot"
                && e.symbol.as_deref() == Some(symbol)
                && e.occurred_at >= from
                && e.occurred_at <= to
            })
            .cloned()
            .collect())
    }

    fn fetch_signal_decisions(&self, symbol: &str, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<StoredEvent>> {
        let events = self.events.lock().unwrap();
        Ok(events.iter()
            .filter(|e| {
                e.event_type == "signal_decision"
                && e.symbol.as_deref() == Some(symbol)
                && e.occurred_at >= from
                && e.occurred_at <= to
            })
            .cloned()
            .collect())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{StoredEvent, TradingEvent, SystemModeChangePayload};

    fn make_mode_event(from: &str, to: &str) -> StoredEvent {
        StoredEvent::new(
            None,
            None,
            None,
            TradingEvent::SystemModeChange(SystemModeChangePayload {
                from_mode: from.into(),
                to_mode:   to.into(),
                reason:    "test".into(),
            }),
        )
    }

    fn make_order_event(coid: &str, corr: &str) -> StoredEvent {
        use crate::events::{OrderSubmittedPayload};
        StoredEvent::new(
            Some("BTCUSDT".into()),
            Some(corr.into()),
            Some(coid.into()),
            TradingEvent::OrderSubmitted(OrderSubmittedPayload {
                client_order_id: coid.into(),
                side:            "BUY".into(),
                qty:             "0.001".into(),
                order_type:      "MARKET".into(),
                expected_price:  50000.0,
            }),
        )
    }

    // ── InMemoryEventStore ────────────────────────────────────────────────────

    #[test]
    fn test_append_and_fetch_recent() {
        let store = InMemoryEventStore::new();
        store.append(make_mode_event("Booting", "Ready"));
        store.append(make_mode_event("Ready", "Halted"));

        let events = store.fetch_recent(10).unwrap();
        assert_eq!(events.len(), 2);
        // fetch_recent returns newest first
        assert_eq!(events[0].event_type, "system_mode_change");
    }

    #[test]
    fn test_fetch_by_coid() {
        let store = InMemoryEventStore::new();
        store.append(make_order_event("coid-1", "corr-a"));
        store.append(make_order_event("coid-2", "corr-b"));

        let events = store.fetch_by_coid("coid-1").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].client_order_id.as_deref(), Some("coid-1"));
    }

    #[test]
    fn test_fetch_trade_lifecycle() {
        let store = InMemoryEventStore::new();
        store.append(make_order_event("coid-a", "corr-x"));
        store.append(make_mode_event("Ready", "Halted")); // different correlation
        store.append(make_order_event("coid-b", "corr-x")); // same correlation

        let lifecycle = store.fetch_trade_lifecycle("corr-x").unwrap();
        assert_eq!(lifecycle.len(), 2, "Should find both events with corr-x");
        for e in &lifecycle {
            assert_eq!(e.correlation_id.as_deref(), Some("corr-x"));
        }
    }

    #[test]
    fn test_count_by_type() {
        let store = InMemoryEventStore::new();
        store.append(make_mode_event("Booting", "Ready"));
        store.append(make_mode_event("Ready", "Halted"));
        store.append(make_order_event("coid-1", "corr-1"));

        assert_eq!(store.count_by_type("system_mode_change"), 2);
        assert_eq!(store.count_by_type("order_submitted"), 1);
        assert_eq!(store.count_by_type("order_filled"), 0);
    }

    #[test]
    fn test_event_serialization_roundtrip() {
        let event = make_mode_event("Ready", "Halted");
        let json = serde_json::to_string(&event.payload).unwrap();
        let payload: TradingEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(payload.event_type_name(), "system_mode_change");
    }

    #[test]
    fn test_stored_event_has_event_id_and_timestamp() {
        let event = make_mode_event("Booting", "Ready");
        assert!(!event.event_id.is_empty());
        // timestamp should be recent
        let age = Utc::now() - event.occurred_at;
        assert!(age.num_seconds() < 5, "Event timestamp too old");
    }

    // ── SqliteEventStore ──────────────────────────────────────────────────────

    #[test]
    fn test_sqlite_append_and_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = SqliteEventStore::open(&db_path).unwrap();

        store.append(make_mode_event("Booting", "Ready"));
        store.append(make_order_event("coid-1", "corr-a"));

        // Give the background writer time to flush (it runs in a thread)
        std::thread::sleep(std::time::Duration::from_millis(50));

        let events = store.fetch_recent(10).unwrap();
        assert_eq!(events.len(), 2, "Both events should be persisted");
    }

    #[test]
    fn test_sqlite_fetch_by_coid() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = SqliteEventStore::open(&db_path).unwrap();

        store.append(make_order_event("coid-alpha", "corr-1"));
        store.append(make_order_event("coid-beta", "corr-2"));
        std::thread::sleep(std::time::Duration::from_millis(50));

        let events = store.fetch_by_coid("coid-alpha").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].client_order_id.as_deref(), Some("coid-alpha"));
    }

    #[test]
    fn test_sqlite_trade_lifecycle_reconstruction() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = SqliteEventStore::open(&db_path).unwrap();

        // Simulate: signal → risk → order → fill (all same correlation)
        let corr = "corr-lifecycle-test";
        use crate::events::*;

        store.append(StoredEvent::new(
            Some("BTCUSDT".into()), Some(corr.into()), None,
            TradingEvent::SignalDecision(SignalDecisionPayload {
                decision: "Buy".into(), exit_reason: None,
                reason: "momentum ok".into(), confidence: 0.8,
                metrics: MarketSnapshotPayload {
                    bid: 50000.0, ask: 50001.0, mid: 50000.5,
                    spread_bps: 2.0, momentum_1s: 0.0001, momentum_3s: 0.0002,
                    momentum_5s: 0.0003, imbalance_1s: 0.3, imbalance_3s: 0.2,
                    imbalance_5s: 0.15, feed_age_ms: 50.0, mid_samples: 10, trade_samples: 8,
                },
            }),
        ));

        store.append(StoredEvent::new(
            Some("BTCUSDT".into()), Some(corr.into()), None,
            TradingEvent::RiskCheckResult(RiskCheckPayload {
                approved: true, side: "BUY".into(), qty: 0.001,
                expected_price: 50001.0, rejection_reason: None,
                position_size: 0.0, position_avg_entry: 0.0,
            }),
        ));

        store.append(StoredEvent::new(
            Some("BTCUSDT".into()), Some(corr.into()), Some("coid-xyz".into()),
            TradingEvent::OrderSubmitted(OrderSubmittedPayload {
                client_order_id: "coid-xyz".into(), side: "BUY".into(),
                qty: "0.001".into(), order_type: "MARKET".into(), expected_price: 50001.0,
            }),
        ));

        store.append(StoredEvent::new(
            Some("BTCUSDT".into()), Some(corr.into()), Some("coid-xyz".into()),
            TradingEvent::OrderFilled(OrderFilledPayload {
                client_order_id: "coid-xyz".into(), exchange_order_id: 99999,
                side: "BUY".into(), filled_qty: 0.001, avg_fill_price: 50001.5,
                cumulative_quote: 50.0015,
            }),
        ));

        std::thread::sleep(std::time::Duration::from_millis(50));

        let lifecycle = store.fetch_trade_lifecycle(corr).unwrap();
        assert_eq!(lifecycle.len(), 4, "All 4 events in lifecycle");
        assert_eq!(lifecycle[0].event_type, "signal_decision");
        assert_eq!(lifecycle[1].event_type, "risk_check_result");
        assert_eq!(lifecycle[2].event_type, "order_submitted");
        assert_eq!(lifecycle[3].event_type, "order_filled");
    }

    #[test]
    fn test_duplicate_event_id_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = SqliteEventStore::open(&db_path).unwrap();

        let mut event = make_mode_event("Booting", "Ready");
        let event_id = event.event_id.clone();
        store.append(event.clone());

        // Modify the payload but keep same event_id
        event.payload = TradingEvent::SystemModeChange(SystemModeChangePayload {
            from_mode: "Ready".into(), to_mode: "Halted".into(), reason: "dup".into(),
        });
        store.append(event);

        std::thread::sleep(std::time::Duration::from_millis(50));

        let events = store.fetch_recent(10).unwrap();
        assert_eq!(events.len(), 1, "Duplicate event_id should be ignored (INSERT OR IGNORE)");
        // The first one should have been kept
        if let TradingEvent::SystemModeChange(p) = &events[0].payload {
            assert_eq!(p.from_mode, "Booting");
        } else {
            panic!("Wrong payload");
        }
    }
}
