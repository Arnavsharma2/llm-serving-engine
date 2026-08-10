import http from "k6/http";
import { check } from "k6";
import exec from "k6/execution";

export const options = {
  scenarios: {
    open_loop: {
      executor: "constant-arrival-rate",
      rate: Number(__ENV.ARRIVAL_RATE || 2),
      timeUnit: "1s",
      duration: __ENV.DURATION || "2m",
      preAllocatedVUs: 16,
      maxVUs: 128,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<10000"],
  },
};

// Deterministic right-skewed lengths; phase-0's Python harness remains canonical.
function logNormalLength(median, sigma, minimum, maximum, index) {
  const pseudo = ((index * 9301 + 49297) % 233280) / 233280;
  const z = Math.sqrt(-2 * Math.log(Math.max(pseudo, 1e-6))) * Math.cos(2 * Math.PI * pseudo);
  return Math.max(minimum, Math.min(maximum, Math.round(median * Math.exp(sigma * z))));
}

export default function () {
  const promptLength = logNormalLength(128, 0.9, 8, 1024, exec.scenario.iterationInTest + 1);
  const outputLength = logNormalLength(64, 0.7, 4, 256, exec.scenario.iterationInTest + 17);
  const payload = JSON.stringify({
    model: "tiny-random",
    prompt: "x".repeat(promptLength),
    max_tokens: outputLength,
    temperature: 0,
  });
  const response = http.post(`${__ENV.BASE_URL || "http://localhost:8000"}/v1/completions`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  check(response, { "completion succeeds": (item) => item.status === 200 });
}
