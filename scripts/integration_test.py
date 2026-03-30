#!/usr/bin/env python3
"""
Automated integration test script for QLib management platform.

Tests all API endpoints for frontend-backend integration (联调测试).
Runs against a live backend server at http://localhost:8000.

Usage:
    # Start backend first, then run:
    python scripts/integration_test.py [--base-url http://localhost:8000]
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTest:
    """Integration test runner for QLib management API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.token: Optional[str] = None
        self.results: list = []
        self.created_resources: Dict[str, list] = {
            "experiments": [],
            "configs": [],
            "factors": [],
            "factor_groups": [],
            "users": [],
        }

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _record(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append({"name": name, "passed": passed, "detail": detail})
        emoji = "✅" if passed else "❌"
        logger.info(f"  {emoji} {name}: {status} {detail}")

    def _get(self, path: str, params: dict = None) -> requests.Response:
        return self.session.get(self._url(path), headers=self._headers(), params=params, timeout=30)

    def _post(self, path: str, data: Any = None) -> requests.Response:
        return self.session.post(self._url(path), headers=self._headers(), json=data, timeout=30)

    def _put(self, path: str, data: Any = None) -> requests.Response:
        return self.session.put(self._url(path), headers=self._headers(), json=data, timeout=30)

    def _delete(self, path: str) -> requests.Response:
        return self.session.delete(self._url(path), headers=self._headers(), timeout=30)

    def _post_form(self, path: str, data: dict) -> requests.Response:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return self.session.post(self._url(path), data=data, headers=headers, timeout=30)

    # =========================================================================
    # Test Module 1: Health & Root
    # =========================================================================
    def test_health(self):
        logger.info("\n📋 Module: Health Check")
        try:
            r = self._get("/health")
            self._record("GET /health", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("GET /health", False, str(e))

        try:
            r = self._get("/")
            self._record("GET /", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("GET /", False, str(e))

    # =========================================================================
    # Test Module 2: Authentication
    # =========================================================================
    def test_auth(self):
        logger.info("\n🔐 Module: Authentication")

        # AUTH-01: Login with admin
        try:
            r = self._post_form("/api/auth/token", {
                "username": "admin",
                "password": "admin123"
            })
            if r.status_code == 200:
                data = r.json()
                self.token = data.get("access_token")
                self._record("AUTH-01 Login admin", True, "token obtained")
            else:
                self._record("AUTH-01 Login admin", False, f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            self._record("AUTH-01 Login admin", False, str(e))

        # AUTH-02: Bad password
        try:
            r = self._post_form("/api/auth/token", {
                "username": "admin",
                "password": "wrongpassword"
            })
            self._record("AUTH-02 Bad password", r.status_code == 401, f"status={r.status_code}")
        except Exception as e:
            self._record("AUTH-02 Bad password", False, str(e))

        # AUTH-03: Get current user info
        try:
            r = self._get("/api/auth/users/me")
            self._record("AUTH-03 Get user info", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("AUTH-03 Get user info", False, str(e))

        # AUTH-04: Register new user (email verification skipped)
        try:
            r = self._post("/api/auth/register", {
                "username": "test_integration_user",
                "email": "test@example.com",
                "full_name": "Test User",
                "password": "testpass123",
                "role": "developer"
            })
            if r.status_code == 200:
                user_data = r.json()
                self._record("AUTH-04 Register user", True, f"user_id={user_data.get('id')}")
            elif r.status_code == 400:
                # User already exists from previous test run - this is acceptable
                self._record("AUTH-04 Register user", True, "user already exists (expected on re-run)")
            else:
                self._record("AUTH-04 Register user", False, f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            self._record("AUTH-04 Register user", False, str(e))

        # AUTH-05: Login with new user
        try:
            r = self._post_form("/api/auth/token", {
                "username": "test_integration_user",
                "password": "testpass123"
            })
            self._record("AUTH-05 Login new user", r.status_code == 200, f"status={r.status_code}")
            # Switch back to admin token
            r2 = self._post_form("/api/auth/token", {"username": "admin", "password": "admin123"})
            if r2.status_code == 200:
                self.token = r2.json().get("access_token")
        except Exception as e:
            self._record("AUTH-05 Login new user", False, str(e))

        # AUTH-06: List users (admin only)
        try:
            r = self._get("/api/auth/users")
            self._record("AUTH-06 List users (admin)", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("AUTH-06 List users (admin)", False, str(e))

        # AUTH-07: Unauthenticated access
        try:
            old_token = self.token
            self.token = None
            r = self._get("/api/experiments/")
            self._record("AUTH-07 Unauth access blocked", r.status_code == 401, f"status={r.status_code}")
            self.token = old_token
        except Exception as e:
            self._record("AUTH-07 Unauth access blocked", False, str(e))

    # =========================================================================
    # Test Module 3: Configs
    # =========================================================================
    def test_configs(self):
        logger.info("\n⚙️ Module: Configs")

        # CFG-01: List configs
        try:
            r = self._get("/api/configs/")
            self._record("CFG-01 List configs", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("CFG-01 List configs", False, str(e))

        # CFG-02: Create config
        try:
            config_yaml = """qlib_init:
  provider_uri: "~/.qlib/qlib_data/cn_data"
  region: cn
task:
  model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
      loss: mse
      colsample_bytree: 0.8
  dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: Alpha158
        module_path: qlib.contrib.data.handler
        kwargs:
          start_time: "2020-01-01"
          end_time: "2023-12-31"
          instruments: csi300
  backtest:
    start_time: "2023-01-01"
    end_time: "2023-12-31"
"""
            r = self._post("/api/configs/", {
                "name": "test_integration_config",
                "description": "Integration test config template",
                "content": config_yaml,
                "type": "experiment_template"
            })
            if r.status_code == 200:
                config_id = r.json().get("id")
                self.created_resources["configs"].append(config_id)
                self._record("CFG-02 Create config", True, f"id={config_id}")
            else:
                self._record("CFG-02 Create config", False, f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            self._record("CFG-02 Create config", False, str(e))

        # CFG-03: Get config by ID
        if self.created_resources["configs"]:
            try:
                cid = self.created_resources["configs"][0]
                r = self._get(f"/api/configs/{cid}")
                self._record("CFG-03 Get config", r.status_code == 200, f"name={r.json().get('name') if r.status_code == 200 else 'N/A'}")
            except Exception as e:
                self._record("CFG-03 Get config", False, str(e))

        # CFG-04: Update config
        if self.created_resources["configs"]:
            try:
                cid = self.created_resources["configs"][0]
                r = self._put(f"/api/configs/{cid}", {
                    "description": "Updated integration test config"
                })
                self._record("CFG-04 Update config", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("CFG-04 Update config", False, str(e))

    # =========================================================================
    # Test Module 4: Experiments
    # =========================================================================
    def test_experiments(self):
        logger.info("\n🧪 Module: Experiments")

        # EXP-01: List experiments
        try:
            r = self._get("/api/experiments/")
            self._record("EXP-01 List experiments", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("EXP-01 List experiments", False, str(e))

        # EXP-02: Create experiment
        try:
            r = self._post("/api/experiments/", {
                "name": "test_integration_experiment",
                "description": "Integration test experiment",
                "config": {
                    "model": "LGBModel",
                    "dataset": "CSI300",
                    "start_time": "2020-01-01",
                    "end_time": "2023-12-31"
                }
            })
            if r.status_code == 200:
                exp_id = r.json().get("id")
                self.created_resources["experiments"].append(exp_id)
                self._record("EXP-02 Create experiment", True, f"id={exp_id}")
            else:
                self._record("EXP-02 Create experiment", False, f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            self._record("EXP-02 Create experiment", False, str(e))

        # EXP-03: Get experiment by ID
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._get(f"/api/experiments/{eid}")
                self._record("EXP-03 Get experiment", r.status_code == 200, f"status_field={r.json().get('status') if r.status_code == 200 else 'N/A'}")
            except Exception as e:
                self._record("EXP-03 Get experiment", False, str(e))

        # EXP-04: Run experiment (creates task)
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._post(f"/api/experiments/{eid}/run")
                if r.status_code == 200:
                    task_id = r.json().get("task_id")
                    self._record("EXP-04 Run experiment", True, f"task_id={task_id}")
                else:
                    self._record("EXP-04 Run experiment", False, f"status={r.status_code} body={r.text[:200]}")
            except Exception as e:
                self._record("EXP-04 Run experiment", False, str(e))

        # EXP-05: Get experiment logs
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._get(f"/api/experiments/{eid}/logs")
                self._record("EXP-05 Get logs", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("EXP-05 Get logs", False, str(e))

        # EXP-06: Get experiment analysis
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._get(f"/api/experiments/{eid}/analysis")
                self._record("EXP-06 Get analysis", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("EXP-06 Get analysis", False, str(e))

        # EXP-07: Signal analysis
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._get(f"/api/experiments/{eid}/analysis/signal")
                self._record("EXP-07 Signal analysis", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("EXP-07 Signal analysis", False, str(e))

        # EXP-08: Portfolio analysis
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._get(f"/api/experiments/{eid}/analysis/portfolio")
                self._record("EXP-08 Portfolio analysis", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("EXP-08 Portfolio analysis", False, str(e))

        # EXP-09: Backtest analysis
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._get(f"/api/experiments/{eid}/analysis/backtest")
                self._record("EXP-09 Backtest analysis", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("EXP-09 Backtest analysis", False, str(e))

    # =========================================================================
    # Test Module 5: Factors
    # =========================================================================
    def test_factors(self):
        logger.info("\n📊 Module: Factors")

        # FAC-01: List factor groups
        try:
            r = self._get("/api/factors/groups")
            self._record("FAC-01 List factor groups", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("FAC-01 List factor groups", False, str(e))

        # FAC-02: List factors
        try:
            r = self._get("/api/factors/")
            self._record("FAC-02 List factors", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("FAC-02 List factors", False, str(e))

        # FAC-03: Create factor
        try:
            r = self._post("/api/factors/", {
                "name": "test_integration_factor",
                "description": "Integration test factor",
                "formula": "($close - Ref($close, 1)) / Ref($close, 1)",
                "type": "momentum"
            })
            if r.status_code == 200:
                fid = r.json().get("id")
                self.created_resources["factors"].append(fid)
                self._record("FAC-03 Create factor", True, f"id={fid}")
            else:
                self._record("FAC-03 Create factor", False, f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            self._record("FAC-03 Create factor", False, str(e))

        # FAC-04: Update factor
        if self.created_resources["factors"]:
            try:
                fid = self.created_resources["factors"][0]
                r = self._put(f"/api/factors/{fid}", {
                    "description": "Updated integration test factor"
                })
                self._record("FAC-04 Update factor", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("FAC-04 Update factor", False, str(e))

        # FAC-05: Get factor group with factors
        try:
            r = self._get("/api/factors/groups")
            if r.status_code == 200 and r.json():
                group_id = r.json()[0].get("id")
                r2 = self._get(f"/api/factors/groups/{group_id}/factors")
                self._record("FAC-05 Get group with factors", r2.status_code == 200, f"status={r2.status_code}")
            else:
                self._record("FAC-05 Get group with factors", True, "no groups to test")
        except Exception as e:
            self._record("FAC-05 Get group with factors", False, str(e))

    # =========================================================================
    # Test Module 6: Data
    # =========================================================================
    def test_data(self):
        logger.info("\n📈 Module: Data")

        # DATA-01: Get stock data list
        try:
            r = self._get("/api/data/", params={"page": 1, "per_page": 10})
            self._record("DATA-01 Get stock data", r.status_code == 200, f"total={r.json().get('total') if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("DATA-01 Get stock data", False, str(e))

        # DATA-02: Get stock codes
        try:
            r = self._get("/api/data/stock-codes")
            self._record("DATA-02 Get stock codes", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("DATA-02 Get stock codes", False, str(e))

        # DATA-03: Get instruments
        try:
            r = self._get("/api/data/instruments", params={"market": "all"})
            self._record("DATA-03 Get instruments", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("DATA-03 Get instruments", False, str(e))

        # DATA-04: Stock data pagination
        try:
            r = self._get("/api/data/", params={"page": 1, "per_page": 5})
            if r.status_code == 200:
                data = r.json()
                r2 = self._get("/api/data/", params={"page": 2, "per_page": 5})
                self._record("DATA-04 Pagination", r2.status_code == 200, f"page1={len(data.get('data', []))} items")
            else:
                self._record("DATA-04 Pagination", False, f"status={r.status_code}")
        except Exception as e:
            self._record("DATA-04 Pagination", False, str(e))

        # DATA-05: Data alignment
        try:
            r = self._post("/api/data/align", {"mode": "auto", "date": "2024-01-01"})
            self._record("DATA-05 Data alignment", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("DATA-05 Data alignment", False, str(e))

    # =========================================================================
    # Test Module 7: Models
    # =========================================================================
    def test_models(self):
        logger.info("\n🤖 Module: Models")

        # MOD-01: List models
        try:
            r = self._get("/api/models/", params={"page": 1, "per_page": 10})
            self._record("MOD-01 List models", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("MOD-01 List models", False, str(e))

    # =========================================================================
    # Test Module 8: Tasks
    # =========================================================================
    def test_tasks(self):
        logger.info("\n📋 Module: Tasks")

        # TASK-01: List tasks
        try:
            r = self._get("/api/tasks/")
            self._record("TASK-01 List tasks", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("TASK-01 List tasks", False, str(e))

        # TASK-02: Get tasks by experiment
        if self.created_resources["experiments"]:
            try:
                eid = self.created_resources["experiments"][0]
                r = self._get(f"/api/tasks/experiment/{eid}")
                self._record("TASK-02 Tasks by experiment", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("TASK-02 Tasks by experiment", False, str(e))

    # =========================================================================
    # Test Module 9: Training API
    # =========================================================================
    def test_training(self):
        logger.info("\n🏋️ Module: Training API")

        # TRAIN-01: List training tasks
        try:
            r = self._get("/api/train/tasks")
            self._record("TRAIN-01 List train tasks", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("TRAIN-01 List train tasks", False, str(e))

        # TRAIN-02: Create training task
        try:
            r = self._post("/api/train/train", {
                "model": "LGBModel",
                "dataset": "CSI300"
            })
            self._record("TRAIN-02 Create train task", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("TRAIN-02 Create train task", False, str(e))

    # =========================================================================
    # Test Module 10: Benchmarks
    # =========================================================================
    def test_benchmarks(self):
        logger.info("\n📏 Module: Benchmarks")

        try:
            r = self._get("/api/benchmarks/")
            self._record("BENCH-01 List benchmarks", r.status_code == 200, f"count={len(r.json()) if r.status_code == 200 else 'N/A'}")
        except Exception as e:
            self._record("BENCH-01 List benchmarks", False, str(e))

    # =========================================================================
    # Test Module 11: Monitoring
    # =========================================================================
    def test_monitoring(self):
        logger.info("\n📡 Module: Monitoring")

        try:
            r = self._get("/api/monitoring/health")
            self._record("MON-01 Health check", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("MON-01 Health check", False, str(e))

        try:
            r = self._get("/api/monitoring/service-status")
            self._record("MON-02 Service status", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            self._record("MON-02 Service status", False, str(e))

    # =========================================================================
    # Test Module 12: Admin (user management)
    # =========================================================================
    def test_admin(self):
        logger.info("\n👑 Module: Admin")

        # ADM-01: Create user via admin
        try:
            r = self._post("/api/auth/users", {
                "username": "test_admin_created_user",
                "email": "admintest@example.com",
                "full_name": "Admin Test User",
                "password": "admintest123",
                "role": "viewer"
            })
            if r.status_code == 200:
                uid = r.json().get("id")
                self.created_resources["users"].append(uid)
                self._record("ADM-01 Create user", True, f"id={uid}")
            elif r.status_code == 400:
                # User already exists from previous test run
                self._record("ADM-01 Create user", True, "user already exists (expected on re-run)")
            else:
                self._record("ADM-01 Create user", False, f"status={r.status_code} body={r.text[:200]}")
        except Exception as e:
            self._record("ADM-01 Create user", False, str(e))

        # ADM-02: Update user
        if self.created_resources["users"]:
            try:
                uid = self.created_resources["users"][0]
                r = self._put(f"/api/auth/users/{uid}", {"role": "developer"})
                self._record("ADM-02 Update user", r.status_code == 200, f"status={r.status_code}")
            except Exception as e:
                self._record("ADM-02 Update user", False, str(e))

    # =========================================================================
    # Cleanup
    # =========================================================================
    def cleanup(self):
        logger.info("\n🧹 Cleanup")

        # Delete created resources in reverse dependency order
        for fid in self.created_resources["factors"]:
            try:
                self._delete(f"/api/factors/{fid}")
                logger.info(f"  Deleted factor {fid}")
            except Exception:
                pass

        for eid in self.created_resources["experiments"]:
            try:
                self._delete(f"/api/experiments/{eid}")
                logger.info(f"  Deleted experiment {eid}")
            except Exception:
                pass

        for cid in self.created_resources["configs"]:
            try:
                self._delete(f"/api/configs/{cid}")
                logger.info(f"  Deleted config {cid}")
            except Exception:
                pass

        for uid in self.created_resources["users"]:
            try:
                self._delete(f"/api/auth/users/{uid}")
                logger.info(f"  Deleted user {uid}")
            except Exception:
                pass

    # =========================================================================
    # Run all tests
    # =========================================================================
    def run_all(self):
        logger.info("=" * 60)
        logger.info("QLib Management Platform - Integration Test")
        logger.info(f"Target: {self.base_url}")
        logger.info("=" * 60)

        start_time = time.time()

        self.test_health()
        self.test_auth()
        self.test_configs()
        self.test_experiments()
        self.test_factors()
        self.test_data()
        self.test_models()
        self.test_tasks()
        self.test_training()
        self.test_benchmarks()
        self.test_monitoring()
        self.test_admin()
        self.cleanup()

        elapsed = time.time() - start_time
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])
        total = len(self.results)

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total: {total}  Passed: {passed}  Failed: {failed}")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Pass rate: {passed/total*100:.1f}%" if total > 0 else "No tests run")

        if failed > 0:
            logger.info("\nFailed tests:")
            for r in self.results:
                if not r["passed"]:
                    logger.info(f"  ❌ {r['name']}: {r['detail']}")

        logger.info("=" * 60)
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="QLib Integration Tests")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Backend API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up created resources after tests"
    )
    args = parser.parse_args()

    # Check server is reachable
    try:
        r = requests.get(f"{args.base_url}/health", timeout=5)
        if r.status_code != 200:
            logger.error(f"Server at {args.base_url} returned status {r.status_code}")
            sys.exit(1)
    except requests.ConnectionError:
        logger.error(f"Cannot connect to server at {args.base_url}")
        logger.error("Please start the backend server first:")
        logger.error("  cd backend && uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    tester = IntegrationTest(args.base_url)
    success = tester.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
