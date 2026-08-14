"""Successful child payload used by the isolated-process worker test."""

import json
import os


print(
    json.dumps(
        {
            "ok": True,
            "result": {"device": "test GPU", "pid": os.getpid()},
        }
    ),
    flush=True,
)
