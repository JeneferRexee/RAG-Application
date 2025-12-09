# run.py
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server on 0.0.0.0:9126")
    uvicorn.run(
        "worker:app",  # points to app object in worker.py
        host="172.17.200.243",
        port=9127,
        reload=True,       # only for development
        access_log=False,
        workers=1          # increase in production if needed
    )
