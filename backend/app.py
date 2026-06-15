from flask import Flask, jsonify
import redis
import os
import requests

app = Flask(__name__)

redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_password = os.getenv("REDIS_PASSWORD") or None

r = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    decode_responses=True
)

@app.route("/api/ping")
def ping():
    try:
        r.incr("ping_count")
        count = r.get("ping_count")
        print(f"received /api/ping, count={count}", flush=True)
        return jsonify({"status": "ok", "count": count})
    except Exception as e:
        print(f"redis error: {e}", flush=True)
        return jsonify({"status": "ok", "redis": "unavailable"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)