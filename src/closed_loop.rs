use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureType {
    BuildFail,
    RuntimeBlock,
    ProfileDesync,
    NoExecution,
    ApiFail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureSeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl FailureSeverity {
    fn rank(self) -> u8 {
        match self {
            FailureSeverity::Critical => 4,
            FailureSeverity::High => 3,
            FailureSeverity::Medium => 2,
            FailureSeverity::Low => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogSource {
    RailwayBuild,
    Runtime,
    AgentDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedFailure {
    #[serde(rename = "type")]
    pub type_: FailureType,
    pub file: String,
    pub line: u32,
    pub reason: String,
    pub severity: FailureSeverity,
}

#[derive(Debug, Clone, Copy)]
pub struct FailureClassificationMap;

impl FailureClassificationMap {
    pub fn classify(&self, source: LogSource, line: &str) -> Option<(FailureType, FailureSeverity, String)> {
        let lower = line.to_ascii_lowercase();
        match source {
            LogSource::RailwayBuild => {
                if lower.contains("could not compile")
                    || lower.contains("error[e")
                    || lower.contains("error: aborting")
                    || lower.contains("linker")
                    || line.contains(".rs:")
                {
                    return Some((FailureType::BuildFail, FailureSeverity::Critical, line.trim().to_string()));
                }
            }
            LogSource::Runtime => {
                if lower.contains("active_profile")
                    && (lower.contains("mismatch") || lower.contains("desync"))
                {
                    return Some((FailureType::ProfileDesync, FailureSeverity::High, line.trim().to_string()));
                }
                if lower.contains("blocked")
                    || lower.contains("final_live_blocker_reason")
                    || lower.contains("blocked_dirty_state")
                {
                    return Some((FailureType::RuntimeBlock, FailureSeverity::High, line.trim().to_string()));
                }
                if lower.contains("order_sent_to_binance=false")
                    || lower.contains("no_execution")
                    || lower.contains("no_order")
                {
                    return Some((FailureType::NoExecution, FailureSeverity::High, line.trim().to_string()));
                }
                if lower.contains("binance")
                    && (lower.contains("error")
                        || lower.contains("timeout")
                        || lower.contains("429")
                        || lower.contains("-2010")
                        || lower.contains("-1021"))
                {
                    return Some((FailureType::ApiFail, FailureSeverity::Critical, line.trim().to_string()));
                }
            }
            LogSource::AgentDecision => {
                if lower.contains("order_sent_to_binance=false") {
                    return Some((FailureType::NoExecution, FailureSeverity::High, line.trim().to_string()));
                }
                if lower.contains("profile")
                    && (lower.contains("source") || lower.contains("mismatch") || lower.contains("desync"))
                {
                    return Some((FailureType::ProfileDesync, FailureSeverity::High, line.trim().to_string()));
                }
                if lower.contains("gate")
                    || lower.contains("blocked")
                    || lower.contains("final_blocker_reason")
                {
                    return Some((FailureType::RuntimeBlock, FailureSeverity::High, line.trim().to_string()));
                }
                if lower.contains("binance")
                    && (lower.contains("error") || lower.contains("reject") || lower.contains("429"))
                {
                    return Some((FailureType::ApiFail, FailureSeverity::Critical, line.trim().to_string()));
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LogParser {
    pub map: FailureClassificationMap,
}

impl Default for LogParser {
    fn default() -> Self {
        Self { map: FailureClassificationMap }
    }
}

impl LogParser {
    pub fn parse(&self, source: LogSource, logs: &str) -> Vec<ParsedFailure> {
        logs.lines()
            .filter_map(|line| {
                let (type_, severity, reason) = self.map.classify(source, line)?;
                let (file, line_num) = extract_file_and_line(line);
                Some(ParsedFailure {
                    type_,
                    file: file.unwrap_or_else(|| "unknown".to_string()),
                    line: line_num.unwrap_or(0),
                    reason,
                    severity,
                })
            })
            .collect()
    }
}

fn extract_file_and_line(line: &str) -> (Option<String>, Option<u32>) {
    for token in line.split_whitespace() {
        let token = token.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
        if let Some(idx) = token.find(".rs:") {
            let file = token[..idx + 3].to_string();
            let rest = &token[idx + 4..];
            let parsed_line = rest
                .split(':')
                .next()
                .and_then(|v| v.parse::<u32>().ok());
            return (Some(file), parsed_line);
        }
    }
    (None, None)
}

#[derive(Debug, Clone)]
pub struct AutoFixPlan {
    pub reason: String,
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct AutoFixEngine;

impl AutoFixEngine {
    pub fn plan_for(&self, failure: &ParsedFailure) -> AutoFixPlan {
        match failure.type_ {
            FailureType::BuildFail => AutoFixPlan {
                reason: format!("BUILD_FAIL: {}", failure.reason),
                commands: vec![
                    "cargo check".to_string(),
                    "cargo test --quiet".to_string(),
                ],
            },
            FailureType::RuntimeBlock => AutoFixPlan {
                reason: format!("RUNTIME_BLOCK: {}", failure.reason),
                commands: vec![
                    "cargo test npc::diagnostic_tests::proof_paper_mode_all_npc_gates_pass_dispatch_is_sole_blocker -- --exact".to_string(),
                ],
            },
            FailureType::ProfileDesync => AutoFixPlan {
                reason: format!("PROFILE_DESYNC: {}", failure.reason),
                commands: vec![
                    "cargo test profile::tests::persist_and_load_active_profile_round_trip -- --exact".to_string(),
                ],
            },
            FailureType::NoExecution => AutoFixPlan {
                reason: format!("NO_EXECUTION: {}", failure.reason),
                commands: vec![
                    "cargo test npc::diagnostic_tests::proof_simulation_reaches_execute_when_score_passes_threshold -- --exact".to_string(),
                ],
            },
            FailureType::ApiFail => AutoFixPlan {
                reason: format!("API_FAIL: {}", failure.reason),
                commands: vec![
                    "cargo test orders::tests::test_rate_limit_is_transient -- --exact".to_string(),
                ],
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitHubMode {
    PullRequest,
    DirectPush,
}

#[derive(Debug, Clone)]
pub struct GitHubAutomationConfig {
    pub repo_path: PathBuf,
    pub branch_prefix: String,
    pub mode: GitHubMode,
}

#[derive(Debug, Clone)]
pub struct GitHubAutomation {
    pub cfg: GitHubAutomationConfig,
}

impl GitHubAutomation {
    pub fn create_branch(&self, reason: &str) -> Result<String> {
        let slug = slugify(reason);
        let branch = format!("{}{}", self.cfg.branch_prefix, slug);
        run_git(&self.cfg.repo_path, &["checkout", "-B", &branch])?;
        Ok(branch)
    }

    pub fn commit_with_reason(&self, reason: &str) -> Result<()> {
        run_git(&self.cfg.repo_path, &["add", "."])?;
        let status = run_git(&self.cfg.repo_path, &["status", "--porcelain"])?;
        if status.trim().is_empty() {
            return Ok(());
        }
        run_git(&self.cfg.repo_path, &["commit", "-m", reason])?;
        Ok(())
    }

    pub fn push_or_open_pr(&self, branch: &str) -> Result<()> {
        run_git(&self.cfg.repo_path, &["push", "-u", "origin", branch])?;
        if self.cfg.mode == GitHubMode::PullRequest {
            run_shell(&self.cfg.repo_path, "gh pr create --fill")
                .context("failed to create pull request with gh CLI")?;
        }
        Ok(())
    }
}

fn run_git(repo_path: &PathBuf, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo_path)
        .output()
        .with_context(|| format!("failed to run git {:?}", args))?;
    if !out.status.success() {
        bail!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn run_shell(repo_path: &PathBuf, command: &str) -> Result<()> {
    let status = Command::new("sh")
        .arg("-lc")
        .arg(command)
        .current_dir(repo_path)
        .status()
        .with_context(|| format!("failed to run shell command: {command}"))?;
    if !status.success() {
        bail!("command failed: {command}");
    }
    Ok(())
}

fn slugify(reason: &str) -> String {
    const BRANCH_SLUG_LIMIT: usize = 48;
    let mut out = String::new();
    for ch in reason.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if ch == ' ' || ch == '-' || ch == '_' {
            if !out.ends_with('-') {
                out.push('-');
            }
        }
    }
    out.trim_matches('-').chars().take(BRANCH_SLUG_LIMIT).collect()
}

#[derive(Debug, Clone)]
pub struct RailwayDeployer {
    pub repo_path: PathBuf,
    pub deploy_command: String,
}

impl RailwayDeployer {
    pub async fn trigger_deploy(&self) -> Result<()> {
        info!(command = %self.deploy_command, "[CLOSED_LOOP] Triggering Railway deploy");
        run_shell(&self.repo_path, &self.deploy_command)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    pub health_ok: bool,
    pub runtime_stable: bool,
    pub active_profile_correct: bool,
    pub no_blocking_gates: bool,
    pub execution_path_reachable: bool,
    pub order_sent_or_ready: bool,
    pub cycle_count: u64,
}

impl ValidationResult {
    pub fn success(&self) -> bool {
        self.health_ok
            && self.runtime_stable
            && self.active_profile_correct
            && self.no_blocking_gates
            && self.execution_path_reachable
            && self.order_sent_or_ready
    }
}

#[derive(Debug, Clone)]
pub struct ClosedLoopControllerConfig {
    pub expected_profile: String,
    pub health_base_url: String,
    pub max_attempts: u32,
    pub check_interval: Duration,
    pub railway_build_log_path: Option<PathBuf>,
    pub runtime_log_path: Option<PathBuf>,
    pub decision_log_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ClosedLoopController {
    pub cfg: ClosedLoopControllerConfig,
    pub parser: LogParser,
    pub auto_fix: AutoFixEngine,
    pub github: GitHubAutomation,
    pub deployer: RailwayDeployer,
    pub client: reqwest::Client,
}

impl ClosedLoopController {
    pub fn new(
        cfg: ClosedLoopControllerConfig,
        github: GitHubAutomation,
        deployer: RailwayDeployer,
    ) -> Self {
        Self {
            cfg,
            parser: LogParser::default(),
            auto_fix: AutoFixEngine,
            github,
            deployer,
            client: reqwest::Client::new(),
        }
    }

    pub async fn run_until_success(&self) -> Result<()> {
        info!("[CLOSED_LOOP] validation loop active");
        for attempt in 1..=self.cfg.max_attempts {
            let failures = self.collect_failures()?;
            if let Some(top) = failures
                .iter()
                .max_by_key(|f| f.severity.rank()) {
                let plan = self.auto_fix.plan_for(top);
                info!(
                    attempt,
                    failure_type = ?top.type_,
                    reason = %top.reason,
                    "[CLOSED_LOOP] Detected failure, applying auto-fix plan"
                );
                let branch = self.github.create_branch(&plan.reason)?;
                let mut commands_ok = true;
                for cmd in &plan.commands {
                    if let Err(e) = run_shell(&self.github.cfg.repo_path, cmd) {
                        commands_ok = false;
                        warn!(command = %cmd, error = %e, "[CLOSED_LOOP] Auto-fix command failed");
                        break;
                    }
                }
                if !commands_ok {
                    continue;
                }
                self.github.commit_with_reason(&plan.reason)?;
                self.github.push_or_open_pr(&branch)?;
                self.deployer.trigger_deploy().await?;
            } else {
                warn!(attempt, "[CLOSED_LOOP] Validation pending, but no classified failures found");
            }

            let validation = self.validate().await?;
            info!(attempt, success = validation.success(), ?validation, "[CLOSED_LOOP] Validation result");
            if validation.success() {
                info!("[CLOSED_LOOP] success condition reached");
                return Ok(());
            }
            tokio::time::sleep(self.cfg.check_interval).await;
        }
        Err(anyhow!("closed loop reached max attempts without success"))
    }

    fn collect_failures(&self) -> Result<Vec<ParsedFailure>> {
        let mut out = Vec::new();
        if let Some(path) = &self.cfg.railway_build_log_path {
            let logs = fs::read_to_string(path)
                .with_context(|| format!("cannot read railway build logs: {}", path.display()))?;
            out.extend(self.parser.parse(LogSource::RailwayBuild, &logs));
        }
        if let Some(path) = &self.cfg.runtime_log_path {
            let logs = fs::read_to_string(path)
                .with_context(|| format!("cannot read runtime logs: {}", path.display()))?;
            out.extend(self.parser.parse(LogSource::Runtime, &logs));
        }
        if let Some(path) = &self.cfg.decision_log_path {
            let logs = fs::read_to_string(path)
                .with_context(|| format!("cannot read decision logs: {}", path.display()))?;
            out.extend(self.parser.parse(LogSource::AgentDecision, &logs));
        }
        Ok(out)
    }

    async fn validate(&self) -> Result<ValidationResult> {
        let health_url = format!("{}/health", self.cfg.health_base_url.trim_end_matches('/'));
        let status_url = format!("{}/agent/status", self.cfg.health_base_url.trim_end_matches('/'));

        let health_resp = self.client.get(&health_url).send().await?;
        let health_ok = health_resp.status().is_success();

        let status_resp = self.client.get(&status_url).send().await?;
        let status_json: Value = status_resp.json().await?;
        let parsed = parse_agent_status(&status_json, &self.cfg.expected_profile);
        Ok(ValidationResult {
            health_ok,
            runtime_stable: parsed.runtime_stable,
            active_profile_correct: parsed.active_profile_correct,
            no_blocking_gates: parsed.no_blocking_gates,
            execution_path_reachable: parsed.execution_path_reachable,
            order_sent_or_ready: parsed.order_sent_or_ready,
            cycle_count: parsed.cycle_count,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedAgentStatus {
    runtime_stable: bool,
    active_profile_correct: bool,
    no_blocking_gates: bool,
    execution_path_reachable: bool,
    order_sent_or_ready: bool,
    cycle_count: u64,
}

fn parse_agent_status(v: &Value, expected_profile: &str) -> ParsedAgentStatus {
    let active_profile = v.get("active_profile").and_then(Value::as_str).unwrap_or_default();
    let mode = v.get("mode").and_then(Value::as_str).unwrap_or_default();
    let state = v.get("state").and_then(Value::as_str).unwrap_or_default();
    let blocker = v.get("final_live_blocker_reason").and_then(Value::as_str).unwrap_or_default();
    let pipeline = v.get("pipeline_state").and_then(Value::as_str).unwrap_or_default();
    let final_decision = v.get("final_decision").and_then(Value::as_str).unwrap_or_default();
    let order_sent = v.get("ORDER_SENT_TO_BINANCE").and_then(Value::as_bool).unwrap_or(false);
    let live_ready = v.get("live_ready").and_then(Value::as_bool).unwrap_or(false);
    let cycle_count = v.get("cycle_count").and_then(Value::as_u64).unwrap_or(0);

    let runtime_stable = !matches!(mode, "Halted" | "Booting")
        && !state.eq_ignore_ascii_case("Recovery");
    let active_profile_correct = active_profile.eq_ignore_ascii_case(expected_profile);
    let no_blocking_gates = blocker.trim().is_empty() || blocker.eq_ignore_ascii_case("none");
    let execution_path_reachable = !pipeline.to_ascii_lowercase().contains("blocked")
        && !final_decision.to_ascii_lowercase().contains("blocked");
    let order_sent_or_ready = order_sent || live_ready;

    ParsedAgentStatus {
        runtime_stable,
        active_profile_correct,
        no_blocking_gates,
        execution_path_reachable,
        order_sent_or_ready,
        cycle_count,
    }
}

pub fn maybe_spawn_from_env(expected_profile: String, web_base_url: Option<String>) {
    let enabled = std::env::var("CLOSED_LOOP_EXECUTION_ENABLED")
        .ok()
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !enabled {
        warn!("[CLOSED_LOOP] CLOSED_LOOP_EXECUTION_ENABLED=false, controller disabled");
        return;
    }
    let Some(base_url) = web_base_url else {
        warn!("[CLOSED_LOOP] Cannot start: no web base URL resolved");
        return;
    };

    let repo_path = PathBuf::from(
        std::env::var("CLOSED_LOOP_REPO_PATH").unwrap_or_else(|_| ".".to_string()),
    );
    let mode = if std::env::var("CLOSED_LOOP_GITHUB_MODE")
        .ok()
        .map(|v| v.eq_ignore_ascii_case("direct"))
        .unwrap_or(false)
    {
        GitHubMode::DirectPush
    } else {
        GitHubMode::PullRequest
    };
    let github = GitHubAutomation {
        cfg: GitHubAutomationConfig {
            repo_path: repo_path.clone(),
            branch_prefix: std::env::var("CLOSED_LOOP_BRANCH_PREFIX")
                .unwrap_or_else(|_| "auto-fix/".to_string()),
            mode,
        },
    };
    let deployer = RailwayDeployer {
        repo_path: repo_path.clone(),
        deploy_command: std::env::var("CLOSED_LOOP_RAILWAY_DEPLOY_CMD")
            .unwrap_or_else(|_| "railway up --ci".to_string()),
    };
    let cfg = ClosedLoopControllerConfig {
        expected_profile,
        health_base_url: base_url,
        max_attempts: std::env::var("CLOSED_LOOP_MAX_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(10),
        check_interval: Duration::from_secs(
            std::env::var("CLOSED_LOOP_CHECK_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(10),
        ),
        railway_build_log_path: std::env::var("CLOSED_LOOP_RAILWAY_BUILD_LOG")
            .ok()
            .map(PathBuf::from),
        runtime_log_path: std::env::var("CLOSED_LOOP_RUNTIME_LOG")
            .ok()
            .map(PathBuf::from),
        decision_log_path: std::env::var("CLOSED_LOOP_AGENT_LOG")
            .ok()
            .map(PathBuf::from),
    };

    let controller = ClosedLoopController::new(cfg, github, deployer);
    tokio::spawn(async move {
        if let Err(e) = controller.run_until_success().await {
            warn!(error = %e, "[CLOSED_LOOP] loop ended without success");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_build_fail_with_file_and_line() {
        let parser = LogParser::default();
        let logs = "error[E0425]: cannot find value `x` in this scope\n --> src/reconciler.rs:1466:9";
        let failures = parser.parse(LogSource::RailwayBuild, logs);
        assert!(!failures.is_empty());
        let build = failures
            .iter()
            .find(|f| f.file == "src/reconciler.rs")
            .expect("expected build failure with file context");
        assert_eq!(build.type_, FailureType::BuildFail);
        assert_eq!(build.line, 1466);
    }

    #[test]
    fn parses_no_execution_from_agent_decisions() {
        let parser = LogParser::default();
        let logs = "ORDER_SENT_TO_BINANCE=false side=BUY qty=0.001";
        let failures = parser.parse(LogSource::AgentDecision, logs);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].type_, FailureType::NoExecution);
    }

    #[test]
    fn validation_accepts_ready_state_without_order_sent() {
        let v = json!({
            "active_profile": "SWING",
            "mode": "Ready",
            "state": "Order Working",
            "final_live_blocker_reason": "",
            "pipeline_state": "live_ready",
            "final_decision": "execute",
            "ORDER_SENT_TO_BINANCE": false,
            "live_ready": true,
            "cycle_count": 22
        });
        let parsed = parse_agent_status(&v, "SWING");
        assert!(parsed.runtime_stable);
        assert!(parsed.active_profile_correct);
        assert!(parsed.no_blocking_gates);
        assert!(parsed.execution_path_reachable);
        assert!(parsed.order_sent_or_ready);
    }
}
