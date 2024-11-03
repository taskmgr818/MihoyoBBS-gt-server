import json
import os
import time
from crack import Crack
import predict

from flask import Flask, request, jsonify

app = Flask(__name__)


stats_file = "stats.json"

if not os.path.exists(stats_file):
    with open(stats_file, "w") as f:
        json.dump({"total": 0, "success": 0, "failure": 0, "error": 0}, f)


def read_stats():
    with open(stats_file, "r") as f:
        return json.load(f)


def update_stats(msg):
    stats = read_stats()
    stats[msg] += 1
    stats["total"] += 1
    with open(stats_file, "w") as f:
        json.dump(stats, f)


@app.route("/statistic", methods=["get"])
def statistic():
    stats = read_stats()
    return f"调用{stats['total']}次，成功{stats['success']}次，失败{stats['failure']}次，错误{stats['error']}次"


@app.route("/", methods=["POST"])
def handle_post():
    data = request.get_json()
    if "gt" in data and "challenge" in data:
        crack = Crack(data["gt"], data["challenge"])
        crack.get_type()
        crack.get_c_s()
        crack.ajax()
        for retry in range(6):
            a = []
            type, image = crack.get_pic(retry)
            t = time.time()
            if type == "nine":
                points = predict.nine(image)
                for x, y in points:
                    a.append(f"{x}_{y}")
            elif type == "icon":
                points = predict.icon(image)
                for x, y in points:
                    left = round(x / 333 * 10000)
                    top = round(y / 333 * 10000)
                    a.append(f"{left}_{top}")
            else:
                update_stats("error")
                return jsonify({"status": "error", "msg": "unsupported type"})
            wait_time = t + 2 - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            res = eval(crack.verify(a))
            if res["data"]["result"] == "success":
                update_stats("success")
                return jsonify(
                    {"status": "success", "validate": res["data"]["validate"]}
                )
        update_stats("failure")
        return jsonify({"status": "failed"})
    else:
        return (
            jsonify({"status": "error", "message": "Missing 'gt' or 'challenge'"}),
            400,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10721)
