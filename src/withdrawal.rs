use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use crate::authority::AuthorityMode;
use crate::client::BinanceClient;
use crate::events::{OperatorActionPayload, StoredEvent, TradingEvent};
use crate::store::EventStore;

#[derive(Clone, Debug)]
pub struct WithdrawalConfig {
    pub auto_execute_enabled: bool,
    pub max_withdrawal_amount: f64,
    pub allowed_destinations: HashSet<String>,
    pub cooldown: Duration,
    pub duplicate_window: Duration,
    pub default_fee: f64,
}

impl Default for WithdrawalConfig {
    fn default() -> Self {
        Self {
            auto_execute_enabled: false,
            max_withdrawal_amount: 1_000.0,
            allowed_destinations: HashSet::new(),
            cooldown: Duration::from_secs(300),
            duplicate_window: Duration::from_secs(600),
            default_fee: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WithdrawalStatus {
    Requested,
    Approved,
    Executing,
    Rejected,
    Completed,
    Failed,
}

#[derive(Clone, Debug)]
pub struct WithdrawalProposal {
    pub id: String,
    pub amount: f64,
    pub asset: String,
    pub destination: String,
    pub network: String,
    pub reason: String,
    pub estimated_fee: f64,
    pub created_at: Instant,
    pub status: WithdrawalStatus,
    pub failure_reason: Option<String>,
    fingerprint: String,
}

impl WithdrawalProposal {
    pub fn final_received_amount(&self) -> f64 {
        (self.amount - self.estimated_fee).max(0.0)
    }
}

struct WithdrawalState {
    proposals: HashMap<String, WithdrawalProposal>,
    last_executed_at: Option<Instant>,
    recent_fingerprints: HashMap<String, Instant>,
}

pub struct WithdrawalManager {
    inner: Arc<Mutex<WithdrawalState>>,
    cfg: WithdrawalConfig,
}

impl WithdrawalManager {
    pub fn new(cfg: WithdrawalConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WithdrawalState {
                proposals: HashMap::new(),
                last_executed_at: None,
                recent_fingerprints: HashMap::new(),
            })),
            cfg,
        }
    }

    pub fn allowed_destinations(&self) -> Vec<String> {
        let mut vals: Vec<_> = self.cfg.allowed_destinations.iter().cloned().collect();
        vals.sort();
        vals
    }

    pub async fn proposals(&self) -> Vec<WithdrawalProposal> {
        let g = self.inner.lock().await;
        let mut items: Vec<_> = g.proposals.values().cloned().collect();
        items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        items
    }

    pub async fn create_request(
        &self,
        req: NewWithdrawalRequest,
        kill_switch_active: bool,
        client: Option<&BinanceClient>,
        store: &dyn EventStore,
    ) -> Result<WithdrawalProposal, String> {
        if kill_switch_active {
            return Err("withdrawals are blocked while kill-switch is active".into());
        }
        if req.amount <= 0.0 {
            return Err("amount must be greater than 0".into());
        }
        if req.amount > self.cfg.max_withdrawal_amount {
            return Err(format!(
                "amount exceeds max withdrawal size ({:.8})",
                self.cfg.max_withdrawal_amount
            ));
        }
        if !self.cfg.allowed_destinations.is_empty()
            && !self.cfg.allowed_destinations.contains(req.destination.trim())
        {
            return Err("destination is not in allowed destination list".into());
        }

        if let Some(cl) = client {
            let balances = cl.fetch_balances().await.map_err(|e| format!("balance check failed: {e:#}"))?;
            let free = balances
                .iter()
                .find(|b| b.asset.eq_ignore_ascii_case(req.asset.trim()))
                .and_then(|b| b.free.parse::<f64>().ok())
                .unwrap_or(0.0);
            if free < req.amount {
                return Err(format!(
                    "insufficient balance: free {:.8} {} < requested {:.8}",
                    free,
                    req.asset.trim().to_uppercase(),
                    req.amount
                ));
            }
        }

        let fingerprint = format!(
            "{}|{}|{}|{:.8}|{}",
            req.asset.trim().to_uppercase(),
            req.network.trim().to_uppercase(),
            req.destination.trim(),
            req.amount,
            req.reason.trim().to_lowercase()
        );

        let mut g = self.inner.lock().await;
        let now = Instant::now();
        g.recent_fingerprints
            .retain(|_, t| now.duration_since(*t) <= self.cfg.duplicate_window);

        if let Some(last) = g.last_executed_at {
            if now.duration_since(last) < self.cfg.cooldown {
                return Err(format!(
                    "cooldown active: wait {:.0}s before next withdrawal",
                    (self.cfg.cooldown - now.duration_since(last)).as_secs_f64()
                ));
            }
        }
        if g.proposals.values().any(|p| {
            matches!(p.status, WithdrawalStatus::Requested | WithdrawalStatus::Approved)
                && p.fingerprint == fingerprint
        }) {
            return Err("duplicate request already pending".into());
        }
        if g.recent_fingerprints.contains_key(&fingerprint) {
            return Err("duplicate request blocked by dedup window".into());
        }

        let proposal = WithdrawalProposal {
            id: Uuid::new_v4().to_string(),
            amount: req.amount,
            asset: req.asset.trim().to_uppercase(),
            destination: req.destination.trim().to_string(),
            network: req.network.trim().to_uppercase(),
            reason: req.reason.trim().to_string(),
            estimated_fee: req.estimated_fee.unwrap_or(self.cfg.default_fee),
            created_at: now,
            status: WithdrawalStatus::Requested,
            failure_reason: None,
            fingerprint: fingerprint.clone(),
        };
        g.proposals.insert(proposal.id.clone(), proposal.clone());
        g.recent_fingerprints.insert(fingerprint, now);

        log_withdrawal_stage(store, "requested", &proposal.id, &format!(
            "{} {:.8} to {} via {}",
            proposal.asset, proposal.amount, proposal.destination, proposal.network
        ));
        Ok(proposal)
    }

    pub async fn approve(&self, id: &str, store: &dyn EventStore) -> Result<WithdrawalProposal, String> {
        let mut g = self.inner.lock().await;
        let p = g.proposals.get_mut(id).ok_or_else(|| "proposal not found".to_string())?;
        if p.status != WithdrawalStatus::Requested {
            return Err("only requested proposals can be approved".into());
        }
        p.status = WithdrawalStatus::Approved;
        let out = p.clone();
        drop(g);
        log_withdrawal_stage(store, "approved", id, "authority approved proposal");
        Ok(out)
    }

    pub async fn reject(&self, id: &str, store: &dyn EventStore) -> Result<WithdrawalProposal, String> {
        let mut g = self.inner.lock().await;
        let p = g.proposals.get_mut(id).ok_or_else(|| "proposal not found".to_string())?;
        if p.status != WithdrawalStatus::Requested {
            return Err("only requested proposals can be rejected".into());
        }
        p.status = WithdrawalStatus::Rejected;
        let out = p.clone();
        drop(g);
        log_withdrawal_stage(store, "rejected", id, "authority rejected proposal");
        Ok(out)
    }

    pub async fn execute(
        &self,
        id: &str,
        mode: AuthorityMode,
        kill_switch_active: bool,
        client: Option<&BinanceClient>,
        store: &dyn EventStore,
    ) -> Result<WithdrawalProposal, String> {
        if kill_switch_active {
            return Err("withdrawals are blocked while kill-switch is active".into());
        }

        let mut g = self.inner.lock().await;
        let (asset, amount, destination, network, approved_flow, auto_flow) = {
            let p = g
                .proposals
                .get(id)
                .ok_or_else(|| "proposal not found".to_string())?;
            (
                p.asset.clone(),
                p.amount,
                p.destination.clone(),
                p.network.clone(),
                p.status == WithdrawalStatus::Approved,
                p.status == WithdrawalStatus::Requested
                    && mode == AuthorityMode::Auto
                    && self.cfg.auto_execute_enabled,
            )
        };
        if !approved_flow && !auto_flow {
            return Err("execution requires approved proposal (or AUTO mode + policy)".into());
        }

        if let Some(last) = g.last_executed_at {
            let elapsed = Instant::now().duration_since(last);
            if elapsed < self.cfg.cooldown {
                return Err(format!("cooldown active: wait {:.0}s", (self.cfg.cooldown - elapsed).as_secs_f64()));
            }
        }

        if let Some(p) = g.proposals.get_mut(id) {
            p.status = WithdrawalStatus::Executing;
            p.failure_reason = None;
        }
        drop(g);

        let response = if let Some(cl) = client {
            cl.request_withdrawal(&asset, amount, &destination, &network)
                .await
                .map_err(|e| format!("withdrawal exchange call failed: {e:#}"))?
        } else {
            return Err("real withdrawal client unavailable in this mode".to_string());
        };

        let mut g = self.inner.lock().await;
        let out = if let Some(p) = g.proposals.get_mut(id) {
            p.status = WithdrawalStatus::Completed;
            p.failure_reason = None;
            p.clone()
        } else {
            return Err("proposal missing at finalize step".to_string());
        };
        g.last_executed_at = Some(Instant::now());
        drop(g);

        log_withdrawal_stage(
            store,
            "executed",
            id,
            &format!("exchange withdrawal id={}", response.id),
        );
        info!(proposal_id=%id, withdrawal_id=%response.id, "Withdrawal executed");
        Ok(out)
    }

    pub async fn mark_failed(&self, id: &str, reason: &str, store: &dyn EventStore) {
        let mut g = self.inner.lock().await;
        if let Some(p) = g.proposals.get_mut(id) {
            p.status = WithdrawalStatus::Failed;
            p.failure_reason = Some(reason.to_string());
            warn!(proposal_id=%id, reason=%reason, "Withdrawal execution failed");
        }
        drop(g);
        log_withdrawal_stage(store, "failed", id, reason);
    }
}

#[derive(Clone, Debug)]
pub struct NewWithdrawalRequest {
    pub amount: f64,
    pub asset: String,
    pub destination: String,
    pub network: String,
    pub reason: String,
    pub estimated_fee: Option<f64>,
}

pub fn log_withdrawal_stage(store: &dyn EventStore, stage: &str, proposal_id: &str, reason: &str) {
    store.append(StoredEvent::new(
        None,
        None,
        None,
        TradingEvent::OperatorAction(OperatorActionPayload {
            action: format!("withdrawal_{stage}:{proposal_id}"),
            reason: reason.to_string(),
        }),
    ));
}
