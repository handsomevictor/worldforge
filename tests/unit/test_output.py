"""Unit tests for SimulationResult and output adapters."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from worldforge.output.result import SimulationResult


# ============================================================
# SimulationResult access
# ============================================================

@pytest.fixture
def sample_data():
    return {
        "purchases": [
            {"user_id": "1", "amount": 50.0, "timestamp": 1},
            {"user_id": "2", "amount": 30.0, "timestamp": 2},
        ],
        "metrics": [
            {"n_users": 10, "gmv": 80.0, "timestamp": 1},
        ],
    }


@pytest.fixture
def result(sample_data):
    return SimulationResult(
        data=sample_data,
        metadata={"name": "test", "seed": 42, "steps": 10},
    )


class TestSimulationResultAccess:
    def test_getitem(self, result):
        rows = result["purchases"]
        assert len(rows) == 2

    def test_contains(self, result):
        assert "purchases" in result
        assert "nonexistent" not in result

    def test_keys(self, result):
        assert sorted(result.keys()) == ["metrics", "purchases"]

    def test_to_dict(self, result):
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "purchases" in d
        assert len(d["purchases"]) == 2

    def test_to_dict_returns_copy(self, result):
        d1 = result.to_dict()
        d2 = result.to_dict()
        assert d1 is not d2

    def test_metadata_access(self, result):
        assert result.metadata["name"] == "test"
        assert result.metadata["seed"] == 42

    def test_repr(self, result):
        r = repr(result)
        assert "SimulationResult" in r
        assert "purchases" in r


# ============================================================
# summary()
# ============================================================

class TestSummary:
    def test_summary_contains_probe_names(self, result):
        s = result.summary()
        assert "purchases" in s
        assert "metrics" in s

    def test_summary_contains_record_counts(self, result):
        s = result.summary()
        assert "2" in s  # purchases has 2 records

    def test_summary_contains_metadata(self, result):
        s = result.summary()
        assert "test" in s


# ============================================================
# to_pandas()
# ============================================================

class TestToPandas:
    def test_returns_dict_of_dataframes(self, result):
        pd = pytest.importorskip("pandas")
        dfs = result.to_pandas()
        assert isinstance(dfs, dict)
        assert "purchases" in dfs
        assert isinstance(dfs["purchases"], pd.DataFrame)

    def test_dataframe_rows_match(self, result):
        pd = pytest.importorskip("pandas")
        dfs = result.to_pandas()
        assert len(dfs["purchases"]) == 2

    def test_dataframe_columns(self, result):
        pd = pytest.importorskip("pandas")
        dfs = result.to_pandas()
        assert "user_id" in dfs["purchases"].columns
        assert "amount" in dfs["purchases"].columns

    def test_empty_probe_gives_empty_df(self):
        pd = pytest.importorskip("pandas")
        r = SimulationResult(data={"empty": []})
        dfs = r.to_pandas()
        assert dfs["empty"].empty


# ============================================================
# to_json()
# ============================================================

class TestToJson:
    def test_writes_json_files(self, result):
        with tempfile.TemporaryDirectory() as tmpdir:
            result.to_json(tmpdir)
            files = os.listdir(tmpdir)
            assert "purchases.json" in files
            assert "metrics.json" in files

    def test_json_is_valid(self, result):
        with tempfile.TemporaryDirectory() as tmpdir:
            result.to_json(tmpdir)
            path = os.path.join(tmpdir, "purchases.json")
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 2
            assert data[0]["user_id"] == "1"

    def test_creates_directory_if_missing(self, result):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            result.to_json(nested)
            assert os.path.exists(nested)


# ============================================================
# to_csv()
# ============================================================

class TestToCsv:
    def test_writes_csv_files(self, result):
        pytest.importorskip("pandas")
        with tempfile.TemporaryDirectory() as tmpdir:
            result.to_csv(tmpdir)
            files = os.listdir(tmpdir)
            assert "purchases.csv" in files

    def test_csv_has_headers(self, result):
        pytest.importorskip("pandas")
        with tempfile.TemporaryDirectory() as tmpdir:
            result.to_csv(tmpdir)
            path = os.path.join(tmpdir, "purchases.csv")
            with open(path) as f:
                header = f.readline()
            assert "user_id" in header
            assert "amount" in header
