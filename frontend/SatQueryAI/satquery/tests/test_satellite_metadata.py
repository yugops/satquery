"""
SatQuery AI — Tests for Satellite Metadata (TLE) Client
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from core.config import SatQueryConfig
from core.external.satellite_metadata import get_tle
from core.models import OutputType, TLEResult
from core.validator import GeoValidatorAgent


@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def real_config():
    return SatQueryConfig(mode="real", db_path=":memory:")


class TestSatelliteMetadata:
    def test_get_tle_mock(self, mock_config):
        tle = get_tle(60133, config=mock_config)
        assert tle is not None
        assert isinstance(tle, TLEResult)
        assert tle.norad_id == 60133
        assert tle.name == "SATELLITE-60133"
        assert tle.line1.startswith("1 60133")
        assert tle.line2.startswith("2 60133")
        assert tle.output_type == OutputType.MOCK

    def test_get_tle_invalid_id(self, mock_config):
        assert get_tle(0, config=mock_config) is None
        assert get_tle(-5, config=mock_config) is None
        assert get_tle("not_an_int", config=mock_config) is None  # type: ignore

    def test_get_tle_mocked_http_success(self, real_config):
        mock_payload = {
            "@context": "https://schema.org",
            "@type": "Observation",
            "satelliteId": 60133,
            "name": "NUSAT-44",
            "date": "2026-08-27T08:00:00.000Z",
            "line1": "1 60133U 24119E   26239.33333333  .00001234  00000+0  12345-4 0  9991",
            "line2": "2 60133  97.4567 123.4567 0012345  45.6789 314.5678 15.12345678 12345",
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tle = get_tle(60133, config=real_config)
            assert tle is not None
            assert isinstance(tle, TLEResult)
            assert tle.norad_id == 60133
            assert tle.name == "NUSAT-44"
            assert tle.epoch == "2026-08-27T08:00:00.000Z"
            assert tle.output_type == OutputType.RULE_BASED

    def test_get_tle_http_error_returns_none(self, real_config):
        with patch("urllib.request.urlopen", side_effect=Exception("HTTP 404")):
            tle = get_tle(999999, config=real_config)
            assert tle is None

    @pytest.mark.asyncio
    async def test_validator_populates_satellite_tle(self, mock_config):
        agent = GeoValidatorAgent(config=mock_config)
        resp = await agent.run(norad_id=25544)
        assert resp.result is not None
        assert resp.result.satellite_tle is not None
        assert resp.result.satellite_tle.norad_id == 25544

    @pytest.mark.asyncio
    async def test_validator_without_norad_id_leaves_tle_none(self, mock_config):
        agent = GeoValidatorAgent(config=mock_config)
        resp = await agent.run()
        assert resp.result is not None
        assert resp.result.satellite_tle is None
