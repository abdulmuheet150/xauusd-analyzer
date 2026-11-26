import "dotenv/config";
import express, { Request, Response } from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ALLOWED_MIME_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/webp",
]);

type ChecklistStatus = "DETECTED" | "MISSING" | "UNCLEAR" | "NOT_APPLICABLE";
type TradeDecision = "BUY" | "SELL" | "NO TRADE";

interface ChecklistItem {
  id: string;
  step: string;
  label: string;
  status: ChecklistStatus;
  detail: string;
}

interface AnalysisResponse {
  decision: TradeDecision;
  screenshotSufficient: boolean;
  detectedAsset: string;
  detectedTimeframes: string;
  reason: string;
  checklist: ChecklistItem[];
}

interface UploadedImageInput {
  mimeType: string;
  data: string;
}

const SYSTEM_INSTRUCTION = `You are a strict, disciplined XAUUSD trading chart analyst.
Analyze ONLY what is visibly present in the uploaded screenshot(s) according to this exact 9-rule strategy:

1. Analyze the 15-minute (15m / M15) timeframe first.
2. Look for a Breaker Block on the 15m chart.
3. If there is no clear Breaker Block, check for a relevant Support/Resistance level or Order Block on the 15m chart.
4. Look for a valid break of that level, preferably with an FVG (Fair Value Gap).
5. Check whether Fibonacci 50% aligns with the setup when it can be determined from the screenshot.
6. After the broken level is tapped/retested, analyze the 5-minute (5m / M5) timeframe.
7. For a BUY setup, require a green/bullish 5-minute confirmation candle after the retest.
8. For a SELL setup, require a red/bearish 5-minute confirmation candle after the retest.
9. If the required conditions are not clearly visible, return NO TRADE instead of guessing.

Critical Rules:
- Do NOT invent missing chart information, hidden candles, unprinted timeframes, or unshown price levels.
- Do NOT claim certainty, high probability, or guaranteed profit.
- Do NOT give a BUY or SELL decision merely because the market looks bullish or bearish.
- If the screenshot quality is blurry, heavily cropped, unreadable, not a price chart, for an asset clearly different from XAUUSD/GOLD, or does not clearly show BOTH the 15-minute structure/retest AND the 5-minute post-retest confirmation candle (for example, only a single timeframe is shown without split-screen or a second chart), explicitly state that the screenshot is insufficient in the reason and return NO TRADE with screenshotSufficient set to false.
- On Step 2 vs Step 3: If a clear 15m Breaker Block is detected, mark "breaker_block" as DETECTED and "sr_or_ob" as NOT_APPLICABLE (or DETECTED if also visibly confluent). If no clear Breaker Block is found, mark "breaker_block" as MISSING and evaluate "sr_or_ob" as DETECTED or MISSING.
- On Step 5 ("fib_50"): Only mark DETECTED if a Fibonacci retracement or clear swing range visibly shows 50% alignment with the zone. If no Fib tool or unambiguous swing high/low is determinable from the screenshot, mark it as NOT_APPLICABLE and state that Fibonacci 50% could not be determined from the screenshot. If a Fib is clearly visible and the setup fails 50% alignment, mark it as MISSING.
- Populate all 7 checklist items in exact order with concise, factual 1-sentence observations grounded strictly in the image.`;

const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    decision: {
      type: Type.STRING,
      enum: ["BUY", "SELL", "NO TRADE"],
      description: "Final trade decision strictly following all 9 strategy rules.",
    },
    screenshotSufficient: {
      type: Type.BOOLEAN,
      description:
        "True only if image quality and timeframe visibility are sufficient to verify both 15m structure and 5m confirmation.",
    },
    detectedAsset: {
      type: Type.STRING,
      description:
        "Asset symbol visible on the chart (e.g., 'XAUUSD', 'GOLD', or 'Not visible').",
    },
    detectedTimeframes: {
      type: Type.STRING,
      description:
        "Timeframe(s) visibly shown on the chart(s) (e.g., '15m & 5m', '15m only', 'Not visible').",
    },
    reason: {
      type: Type.STRING,
      description:
        "Concise factual explanation of the decision. If insufficient, explicitly state that the screenshot is insufficient. Never claim certainty or guaranteed profit.",
    },
    checklist: {
      type: Type.ARRAY,
      description: "Exactly 7 strategy checklist items in order.",
      items: {
        type: Type.OBJECT,
        properties: {
          id: {
            type: Type.STRING,
            enum: [
              "timeframe_15m",
              "breaker_block",
              "sr_or_ob",
              "level_break_fvg",
              "fib_50",
              "retest_tap",
              "confirmation_5m",
            ],
          },
          step: { type: Type.STRING },
          label: { type: Type.STRING },
          status: {
            type: Type.STRING,
            enum: ["DETECTED", "MISSING", "UNCLEAR", "NOT_APPLICABLE"],
          },
          detail: { type: Type.STRING },
        },
        required: ["id", "step", "label", "status", "detail"],
      },
    },
  },
  required: [
    "decision",
    "screenshotSufficient",
    "detectedAsset",
    "detectedTimeframes",
    "reason",
    "checklist",
  ],
};

function enforceStrategyGuardrails(parsed: AnalysisResponse): AnalysisResponse {
  const byId = new Map(parsed.checklist.map((item) => [item.id, item]));

  const tf15 = byId.get("timeframe_15m")?.status;
  const breaker = byId.get("breaker_block")?.status;
  const srOrOb = byId.get("sr_or_ob")?.status;
  const levelBreak = byId.get("level_break_fvg")?.status;
  const retest = byId.get("retest_tap")?.status;
  const confirm5m = byId.get("confirmation_5m")?.status;

  const hasKeyLevel = breaker === "DETECTED" || srOrOb === "DETECTED";
  const allMandatoryMet =
    parsed.screenshotSufficient &&
    tf15 === "DETECTED" &&
    hasKeyLevel &&
    levelBreak === "DETECTED" &&
    retest === "DETECTED" &&
    confirm5m === "DETECTED";

  if (!allMandatoryMet && parsed.decision !== "NO TRADE") {
    return {
      ...parsed,
      decision: "NO TRADE",
      reason: `${parsed.reason} Because one or more mandatory 15m/5m conditions were missing or unclear, the final decision is NO TRADE.`,
    };
  }

  return parsed;
}

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("MISSING_API_KEY");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

async function analyzeWithRetry(
  ai: GoogleGenAI,
  images: UploadedImageInput[]
): Promise<AnalysisResponse> {
  const imageParts = images.map((img) => ({
    inlineData: {
      mimeType: img.mimeType === "image/jpg" ? "image/jpeg" : img.mimeType,
      data: img.data,
    },
  }));

  const promptText =
    images.length > 1
      ? "Analyze these 2 uploaded XAUUSD chart screenshots together according to the exact 9-rule strategy. Identify which image shows 15m and which shows 5m (if visible), verify every condition strictly, and return the structured JSON response."
      : "Analyze this uploaded XAUUSD chart screenshot according to the exact 9-rule strategy. Verify whether both 15m structure/retest and 5m confirmation candle are clearly visible, check every condition strictly, and return the structured JSON response.";

  let lastError: unknown;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: "gemini-4-hydrogen",
        contents: {
          parts: [...imageParts, { text: promptText }],
        },
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
          responseMimeType: "application/json",
          responseSchema: RESPONSE_SCHEMA,
        },
      });

      const rawText = (response.text ?? "").trim();
      if (!rawText) {
        throw new Error("EMPTY_MODEL_RESPONSE");
      }

      const parsed = JSON.parse(rawText) as AnalysisResponse;
      return enforceStrategyGuardrails(parsed);
    } catch (error) {
      lastError = error;
      const msg = error instanceof Error ? error.message : String(error);
      const isTransient =
        msg.includes("503") ||
        msg.includes("500") ||
        msg.includes("UNAVAILABLE") ||
        msg.includes("ECONNRESET") ||
        msg.includes("ETIMEDOUT");
      if (!isTransient || attempt === 2) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 600 * (attempt + 1)));
    }
  }

  throw lastError;
}

const ALLOWED_ORIGINS = new Set([
  "https://localhost",
  "http://localhost",
  "capacitor://localhost",
]);

async function startServer() {
  const app = express();
  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (
      origin &&
      (ALLOWED_ORIGINS.has(origin) ||
        origin === process.env.APP_URL ||
        origin.startsWith("http://localhost:") ||
        origin.startsWith("https://localhost:"))
    ) {
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Vary", "Origin");
      res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    }
    if (req.method === "OPTIONS") {
      res.sendStatus(204);
      return;
    }
    next();
  });
  app.use(express.json({ limit: "25mb" }));

  app.get("/api/health", (_req: Request, res: Response) => {
    res.json({ ok: true });
  });

  app.post("/api/analyze", async (req: Request, res: Response) => {
    try {
      const body = req.body as { images?: UploadedImageInput[] };
      const images = body?.images;

      if (!Array.isArray(images) || images.length === 0 || images.length > 2) {
        res.status(400).json({
          error: "Please upload 1 or 2 chart screenshot images.",
        });
        return;
      }

      for (const img of images) {
        if (
          !img ||
          typeof img.mimeType !== "string" ||
          typeof img.data !== "string" ||
          !ALLOWED_MIME_TYPES.has(img.mimeType.toLowerCase()) ||
          img.data.length < 32
        ) {
          res.status(400).json({
            error:
              "Unsupported or invalid image format. Please upload a PNG, JPG, or WEBP screenshot.",
          });
          return;
        }
        // Reject abnormally large single base64 strings (~18MB decoded)
        if (img.data.length > 24_000_000) {
          res.status(413).json({
            error: "Screenshot file is too large. Please upload a smaller image.",
          });
          return;
        }
      }

      const ai = getAiClient();
      const result = await analyzeWithRetry(ai, images);
      res.json(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error("Chart analysis error:", message);

      if (message === "MISSING_API_KEY") {
        res.status(500).json({
          error:
            "Gemini API key is not configured. Please check the Settings > Secrets panel.",
        });
        return;
      }

      if (
        message.includes("API_KEY_INVALID") ||
        message.includes("PERMISSION_DENIED") ||
        message.includes("403")
      ) {
        res.status(403).json({
          error:
            "API key permission denied or invalid. Please verify your key in Settings > Secrets.",
        });
        return;
      }

      if (message.includes("RESOURCE_EXHAUSTED") || message.includes("429")) {
        res.status(429).json({
          error:
            "Rate limit reached. Please wait a moment and try analyzing the chart again.",
        });
        return;
      }

      res.status(500).json({
        error:
          "Unable to complete chart analysis right now. Please try again with a clear chart screenshot.",
      });
    }
  });

  if (process.env.NODE_ENV === "production") {
    const distDir = path.resolve(__dirname, "dist");
    app.use(express.static(distDir));
    app.get("/{*splat}", (_req: Request, res: Response) => {
      const indexFile = path.join(distDir, "index.html");
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send("Build output not found.");
      }
    });
  } else {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  const PORT = Number(process.env.PORT) || 3000;
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
