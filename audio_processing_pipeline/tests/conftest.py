# contains pytest fixtures to stop and start the API during tests
import pytest
import asyncio
import sys
import os
import time
import subprocess
import signal
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Replace event_loop fixture with event_loop_policy fixture
@pytest.fixture(scope="session")
def event_loop_policy():
    """Return an event loop policy for test session."""
    return asyncio.DefaultEventLoopPolicy()

# Let pytest_asyncio create the event loop using our policy
# The fixture is automatically used without needing to be defined

@pytest.fixture(scope="session")
def api_server():
    """Start the API server as a subprocess and stop it after tests."""
    # Start the server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "src.api.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Used to kill the process group
    )

    # Wait for the server to start
    time.sleep(2)  # Give the server time to start

    # Check if the server started successfully
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        raise RuntimeError(f"Server failed to start: {stderr.decode()}")

    print("API server started for testing")

    yield server_process

    # Stop the server after tests
    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    server_process.wait()
    print("API server stopped")