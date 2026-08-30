# experiments/fedavg/server/run_server.py

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "experiments.fedAvg.server.api:app",
        host="0.0.0.0",
        port=8001,
        access_log=False,
    )