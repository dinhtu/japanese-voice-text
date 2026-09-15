import config from "@/config";
import type { EvaluateResponse } from "@/types/pronunciation";

/** Error carrying the HTTP status so the UI can tell apart user vs server faults. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * POST /api/pronunciation/evaluate — send the target text and a WAV recording.
 */
export async function evaluatePronunciation(
  text: string,
  audio: Blob,
  signal?: AbortSignal,
): Promise<EvaluateResponse> {
  const formData = new FormData();
  formData.append("text", text);
  formData.append("audio", new File([audio], "recording.wav", { type: "audio/wav" }));

  let response: Response;
  try {
    response = await fetch(`${config.apiUrl}/api/pronunciation/evaluate`, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(
      `Không kết nối được tới API (${config.apiUrl}). Kiểm tra backend đã chạy chưa.`,
      0,
    );
  }

  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new ApiError(detail || `Yêu cầu thất bại (HTTP ${response.status})`, response.status);
  }

  return (await response.json()) as EvaluateResponse;
}
