import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express, { type Request, type Response } from "express";
import { GoogleGenAI, Type, type Part, type Schema } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const MODEL_NAME = "gemini-3.8-flash";
const TTS_MODEL_NAME = "gemini-3.8-flash-lite-tts";
const DEFAULT_TTS_VOICE = "Kore";

type AiRlcRequestKind =
  | "patrol_text"
  | "patrol_audio"
  | "rlc_console_text"
  | "rlc_console_audio"
  | "call_request";

interface AiRlcRequestBody {
  kind?: AiRlcRequestKind;
  userPrompt?: string;
  systemInstruction?: string;
  audioBase64?: string;
  mimeType?: string;
  includeAudio?: boolean;
  voiceName?: string;
}

const DIGIT_WORDS_SV: Record<string, string> = {
  "0": "noll",
  "1": "ett",
  "2": "två",
  "3": "tre",
  "4": "fyra",
  "5": "fem",
  "6": "sex",
  "7": "sju",
  "8": "åtta",
  "9": "nio",
};

function formatTwoDigitPairSvServer(pairStr: string): string {
  if (!pairStr || pairStr.length !== 2) return pairStr;
  if (pairStr[0] === "0") {
    return `noll ${DIGIT_WORDS_SV[pairStr[1]] || pairStr[1]}`;
  }
  return pairStr;
}

function formatSwedishRadioSpeechServer(rawText: string): string {
  if (!rawText) return "";
  let txt = String(rawText);
  txt = txt.replace(/\b9-([0-9])\b/g, (_m, d) => `nio ${DIGIT_WORDS_SV[d] || d}`);
  txt = txt.replace(/\b(\d{2})-(\d{2})\s*(\d{2})\b/g, (_m, p1, p2, p3) => {
    return `${formatTwoDigitPairSvServer(p1)} ${formatTwoDigitPairSvServer(p2)} ${formatTwoDigitPairSvServer(p3)}`;
  });
  txt = txt.replace(/\b(97|34|24)(\d{2})(\d{2})\b/g, (_m, p1, p2, p3) => {
    return `${formatTwoDigitPairSvServer(p1)} ${formatTwoDigitPairSvServer(p2)} ${formatTwoDigitPairSvServer(p3)}`;
  });
  return txt;
}

async function synthesizeSwedishRadioSpeech(
  ai: GoogleGenAI,
  text: string,
  voiceName = DEFAULT_TTS_VOICE
): Promise<{ audioBase64: string; audioMimeType: string } | null> {
  const spokenText = formatSwedishRadioSpeechServer(text).trim();
  if (!spokenText) return null;

  const ttsResult = await ai.models.generateContent({
    model: TTS_MODEL_NAME,
    contents: [
      {
        role: "user",
        parts: [
          {
            text: spokenText,
            speechMetadata: {
              style:
                "Lugn, saklig och tydlig svensk polisradiooperatör som talar på svenska över Rakel-radio med kort, professionell radiodisciplin.",
            },
          },
        ],
      },
    ],
    config: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: {
            voiceName: voiceName || DEFAULT_TTS_VOICE,
          },
        },
      },
      abortSignal: AbortSignal.timeout(12000),
    },
  });

  const parts = ttsResult.candidates?.[0]?.content?.parts || [];
  for (const part of parts) {
    if (part.inlineData?.data) {
      return {
        audioBase64: part.inlineData.data,
        audioMimeType: part.inlineData.mimeType || "audio/wav",
      };
    }
  }
  return null;
}

function getGeminiClient(apiVersion?: string): GoogleGenAI {
  const apiKey = (process.env.GEMINI_API_KEY || "").trim();
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY är inte konfigurerad på servern.");
  }
  return new GoogleGenAI({
    apiKey,
    ...(apiVersion ? { apiVersion } : {}),
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function getResponseSchemaForKind(kind: AiRlcRequestKind): Schema {
  if (kind === "rlc_console_text") {
    return {
      type: Type.OBJECT,
      properties: {
        responderCallsign: { type: Type.STRING },
        spokenResponse: { type: Type.STRING },
      },
      required: ["responderCallsign", "spokenResponse"],
    };
  }
  if (kind === "rlc_console_audio") {
    return {
      type: Type.OBJECT,
      properties: {
        transcript: { type: Type.STRING },
        responderCallsign: { type: Type.STRING },
        spokenResponse: { type: Type.STRING },
      },
      required: ["transcript", "responderCallsign", "spokenResponse"],
    };
  }
  if (kind === "patrol_audio") {
    return {
      type: Type.OBJECT,
      properties: {
        transcript: { type: Type.STRING },
        shouldReply: { type: Type.BOOLEAN },
        responderLabel: { type: Type.STRING },
        spokenResponse: { type: Type.STRING },
        sdsFollowUp: { type: Type.STRING },
        newStatus: { type: Type.STRING },
      },
      required: ["transcript", "shouldReply", "spokenResponse"],
    };
  }
  return {
    type: Type.OBJECT,
    properties: {
      shouldReply: { type: Type.BOOLEAN },
      responderLabel: { type: Type.STRING },
      spokenResponse: { type: Type.STRING },
      sdsFollowUp: { type: Type.STRING },
      newStatus: { type: Type.STRING },
    },
    required: ["shouldReply", "spokenResponse"],
  };
}

function extractStatusAndMessage(err: unknown): { status: number; message: string } {
  const anyErr = err as { status?: number; code?: number; message?: string } | undefined;
  const rawMessage = anyErr?.message || String(err || "Okänt serverfel");
  const statusCandidate = Number(anyErr?.status || anyErr?.code);
  if (Number.isInteger(statusCandidate) && statusCandidate >= 400 && statusCandidate < 600) {
    return { status: statusCandidate, message: rawMessage };
  }
  const match = rawMessage.match(/\b(400|401|403|404|429|500|502|503)\b/);
  if (match) {
    return { status: Number(match[1]), message: rawMessage };
  }
  return { status: 500, message: rawMessage };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "15mb" }));

  const allowedOriginsEnv = (process.env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  app.use("/api", (req: Request, res: Response, next) => {
    const reqOrigin = req.headers.origin;
    if (allowedOriginsEnv.length === 0 || allowedOriginsEnv.includes("*")) {
      res.setHeader("Access-Control-Allow-Origin", "*");
    } else if (reqOrigin && allowedOriginsEnv.includes(reqOrigin)) {
      res.setHeader("Access-Control-Allow-Origin", reqOrigin);
      res.setHeader("Vary", "Origin");
    }
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    if (req.method === "OPTIONS") {
      res.status(204).end();
      return;
    }
    next();
  });

  app.get("/api/health", (_req: Request, res: Response) => {
    res.json({ ok: true });
  });

  app.post("/api/ai-rlc", async (req: Request, res: Response) => {
    try {
      const body = (req.body || {}) as AiRlcRequestBody;
      const kind: AiRlcRequestKind = body.kind || "patrol_text";
      const userPrompt = typeof body.userPrompt === "string" ? body.userPrompt.trim() : "";
      const systemInstruction = typeof body.systemInstruction === "string" ? body.systemInstruction : "";

      if (!userPrompt) {
        res.status(400).json({ error: "Saknar userPrompt i anropet." });
        return;
      }

      const ai = getGeminiClient();
      const parts: Part[] = [{ text: userPrompt }];
      if (typeof body.audioBase64 === "string" && body.audioBase64.length > 0) {
        parts.push({
          inlineData: {
            mimeType: body.mimeType || "audio/wav",
            data: body.audioBase64,
          },
        });
      }

      const responseSchema = getResponseSchemaForKind(kind);

      const candidateModels: Array<{ model: string; timeoutMs: number }> = [
        { model: MODEL_NAME, timeoutMs: 8000 },
        { model: "gemini-4-hydrogen", timeoutMs: 15000 },
        { model: "gemini-3.1-flash-lite", timeoutMs: 15000 },
      ];
      let result: Awaited<ReturnType<typeof ai.models.generateContent>> | null = null;
      let lastErr: unknown = null;

      for (const cfg of candidateModels) {
        try {
          result = await ai.models.generateContent({
            model: cfg.model,
            contents: [{ role: "user", parts }],
            config: {
              ...(systemInstruction ? { systemInstruction } : {}),
              responseMimeType: "application/json",
              responseSchema,
              abortSignal: AbortSignal.timeout(cfg.timeoutMs),
            },
          });
          break;
        } catch (modelErr) {
          lastErr = modelErr;
          const { status } = extractStatusAndMessage(modelErr);
          if (status === 400 || status === 401 || status === 403) {
            throw modelErr;
          }
        }
      }

      if (!result) {
        throw lastErr || new Error("Alla Gemini-modeller misslyckades eller tog för lång tid.");
      }

      const rawText = (result.text ?? "").trim();
      if (!rawText) {
        res.status(502).json({ error: "Tomt svar från Gemini (inga candidates)" });
        return;
      }

      const cleanedJson = rawText.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
      const parsed = JSON.parse(cleanedJson);

      let audioBase64: string | null = null;
      let audioMimeType: string | null = null;
      if (
        body.includeAudio !== false &&
        parsed?.shouldReply !== false &&
        typeof parsed?.spokenResponse === "string" &&
        parsed.spokenResponse.trim().length > 0
      ) {
        try {
          const ttsAi = getGeminiClient("v1beta");
          const ttsAudio = await synthesizeSwedishRadioSpeech(
            ttsAi,
            parsed.spokenResponse,
            typeof body.voiceName === "string" && body.voiceName.trim() ? body.voiceName.trim() : DEFAULT_TTS_VOICE
          );
          if (ttsAudio) {
            audioBase64 = ttsAudio.audioBase64;
            audioMimeType = ttsAudio.audioMimeType;
          }
        } catch (ttsErr) {
          console.warn("Server /api/ai-rlc TTS fallback to browser speechSynthesis:", ttsErr);
        }
      }

      res.json({
        ...parsed,
        audioBase64,
        audioMimeType,
      });
    } catch (err) {
      const { status, message } = extractStatusAndMessage(err);
      console.error("Server /api/ai-rlc error:", err);
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/ai-rlc-tts", async (req: Request, res: Response) => {
    try {
      const body = (req.body || {}) as { text?: string; voiceName?: string };
      const text = typeof body.text === "string" ? body.text.trim() : "";
      if (!text) {
        res.status(400).json({ error: "Saknar text i TTS-anropet." });
        return;
      }
      const ttsAi = getGeminiClient("v1beta");
      const ttsAudio = await synthesizeSwedishRadioSpeech(
        ttsAi,
        text,
        typeof body.voiceName === "string" && body.voiceName.trim() ? body.voiceName.trim() : DEFAULT_TTS_VOICE
      );
      if (!ttsAudio) {
        res.status(502).json({ error: "Kunde inte generera ljud från Gemini TTS." });
        return;
      }
      res.json(ttsAudio);
    } catch (err) {
      const { status, message } = extractStatusAndMessage(err);
      console.error("Server /api/ai-rlc-tts error:", err);
      res.status(status).json({ error: message });
    }
  });

  const distDir = path.join(__dirname, "dist");
  const distIndex = path.join(distDir, "index.html");
  const isDev = process.env.NODE_ENV === "development" || !fs.existsSync(distIndex);

  if (isDev) {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(distDir));
    app.get("/manifest.json", (_req: Request, res: Response) => {
      const target = fs.existsSync(path.join(distDir, "manifest.json"))
        ? path.join(distDir, "manifest.json")
        : path.join(__dirname, "manifest.json");
      res.sendFile(target);
    });
    app.get("/sw.js", (_req: Request, res: Response) => {
      const target = fs.existsSync(path.join(distDir, "sw.js"))
        ? path.join(distDir, "sw.js")
        : path.join(__dirname, "sw.js");
      res.type("application/javascript").sendFile(target);
    });
    app.use((_req: Request, res: Response) => {
      res.sendFile(distIndex);
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Rakel simulator server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
