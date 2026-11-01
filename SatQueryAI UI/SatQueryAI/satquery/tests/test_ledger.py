"""Tests for AuditLedgerAgent (Agent 3)."""

import os
import sqlite3
import tempfile

import pytest

from core.config import SatQueryConfig
from core.models import AgentResponse, AgentStatus, OutputType
from core.ledger import (
    AuditLedgerAgent,
    tool_calculate_calibrated_confidence,
    tool_commit_ledger_entry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_config(temp_db):
    return SatQueryConfig(mode="mock", db_path=temp_db)


@pytest.fixture
def ledger(mock_config):
    return AuditLedgerAgent(config=mock_config)


# ---------------------------------------------------------------------------
# Calibrated Confidence Tests
# ---------------------------------------------------------------------------

class TestCalibratedConfidence:
    def test_empty_scores(self):
        assert tool_calculate_calibrated_confidence([]) == 0.0

    def test_single_perfect_score(self):
        result = tool_calculate_calibrated_confidence([1.0])
        assert result == 1.0

    def test_single_zero_score(self):
        result = tool_calculate_calibrated_confidence([0.0])
        assert result == 0.0

    def test_mixed_scores(self):
        result = tool_calculate_calibrated_confidence([0.9, 0.7, 0.5])
        assert 0.5 < result < 1.0

    def test_snr_degradation(self):
        # Same scores, lower SNR should give lower confidence
        high_snr = tool_calculate_calibrated_confidence([0.8, 0.8], sensor_snr=30.0)
        low_snr = tool_calculate_calibrated_confidence([0.8, 0.8], sensor_snr=10.0)
        assert low_snr < high_snr

    def test_result_clamped(self):
        result = tool_calculate_calibrated_confidence([1.0, 1.0, 1.0])
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Commit Ledger Entry Tests
# ---------------------------------------------------------------------------

class TestCommitLedgerEntry:
    def test_commit_returns_uuid(self, temp_db):
        entry_id = tool_commit_ledger_entry(
            session_id="test-session",
            trace_data={"agent_name": "TestAgent", "query": "test"},
            db_path=temp_db,
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) == 36  # UUID length

    def test_entry_persisted_in_db(self, temp_db):
        tool_commit_ledger_entry(
            session_id="sess-001",
            trace_data={
                "agent_name": "VQAAgent",
                "query": "How many buildings?",
                "confidence": 0.85,
                "status": "success",
            },
            db_path=temp_db,
        )

        conn = sqlite3.connect(temp_db)
        rows = conn.execute(
            "SELECT agent_name, query, confidence FROM audit_ledger"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "VQAAgent"
        assert rows[0][1] == "How many buildings?"
        assert rows[0][2] == 0.85

    def test_schema_auto_created(self, temp_db):
        """DB table is created on first commit."""
        tool_commit_ledger_entry(
            session_id="s1",
            trace_data={"agent_name": "Test"},
            db_path=temp_db,
        )

        conn = sqlite3.connect(temp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()

        table_names = [t[0] for t in tables]
        assert "audit_ledger" in table_names

    def test_multiple_entries(self, temp_db):
        for i in range(5):
            tool_commit_ledger_entry(
                session_id=f"sess-{i}",
                trace_data={"agent_name": f"Agent{i}"},
                db_path=temp_db,
            )

        conn = sqlite3.connect(temp_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM audit_ledger"
        ).fetchone()[0]
        conn.close()

        assert count == 5


# ---------------------------------------------------------------------------
# Agent Run Tests
# ---------------------------------------------------------------------------

class TestLedgerAgent:
    @pytest.mark.asyncio
    async def test_run_commits_entries(self, ledger):
        responses = [
            AgentResponse(
                agent="TestAgent1",
                status=AgentStatus.SUCCESS,
                result={"answer": "test"},
                confidence=0.9,
                execution_time_ms=42.0,
                output_type=OutputType.MOCK,
            ),
            AgentResponse(
                agent="TestAgent2",
                status=AgentStatus.SUCCESS,
                result={"answer": "test2"},
                confidence=0.7,
                execution_time_ms=55.0,
                output_type=OutputType.MOCK,
            ),
        ]

        response = await ledger.run(
            session_id="audit-test",
            query="Test query",
            intent="vqa",
            agent_responses=responses,
        )

        assert response.status == AgentStatus.SUCCESS
        result = response.result
        assert result["entries_committed"] == 2
        assert len(result["entry_ids"]) == 2
        assert 0.0 <= result["calibrated_confidence"] <= 1.0
