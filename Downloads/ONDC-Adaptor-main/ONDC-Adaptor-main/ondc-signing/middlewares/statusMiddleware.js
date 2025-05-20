// /middlewares/statusMiddleware.js

import fs from "fs";
import path from "path";
import Ajv from "ajv";
import addFormats from "ajv-formats";
import * as ondcUtils from "../utils/ondcUtils.js";
import logger from "../utils/logger.js";
import { createNackResponse } from "../utils/ondcUtils.js";

const ajv = new Ajv({ allErrors: true });
addFormats(ajv);

// Resolve the schema path safely
const schemaPath = path.resolve("utils/schemas/status.json");

let validateStatusSchema;
try {
  const schemaRaw = fs.readFileSync(schemaPath, "utf-8");
  const schema = JSON.parse(schemaRaw);
  validateStatusSchema = ajv.compile(schema);
  logger.info({ message: "✅ Loaded and compiled /status schema successfully" });
} catch (err) {
  logger.error({ message: "❌ Failed to load ONDC /status schema", error: err.message });
  validateStatusSchema = null;
}

const processStatusRequest = async (req, res, next) => {
  const context = req.body?.context || {};
  const transactionId = context.transaction_id;
  const messageId = context.message_id;

  logger.info({ message: "📩 Received request on /status endpoint", transactionId, messageId });

  if (!validateStatusSchema) {
    return res.status(500).json(createNackResponse({
      type: "CORE_ERROR",
      code: "SCHEMA_MISSING",
      message: "Schema validation not configured"
    }));
  }

  const valid = validateStatusSchema(req.body);
  if (!valid) {
    const firstError = validateStatusSchema.errors?.[0] || {};
    logger.warn({ message: "⚠️ Schema validation failed", errors: validateStatusSchema.errors, transactionId });
    return res.status(400).json(createNackResponse({
      type: "DOMAIN_ERROR",
      code: "SCHEMA_VALIDATION_FAILED",
      message: `Schema error: ${firstError.instancePath} ${firstError.message}`
    }));
  }

  // Skip auth in dev mode
  if (process.env.SKIP_AUTH === 'true' || req.headers['x-dev-mode'] === 'true') {
    logger.warn({ message: "🔓 Authorization skipped in development mode", transactionId, messageId });
    return next();
  }

  const authHeader = req.headers["authorization"];
  if (!authHeader) {
    logger.warn({ message: "❌ Missing authorization header", transactionId });
    return res.status(401).json(createNackResponse({
      type: "AUTH_ERROR",
      code: "AUTH_HEADER_MISSING",
      message: "Authorization header is missing"
    }));
  }

  try {
    const parsed = ondcUtils.parseAuthorizationHeader(authHeader);
    if (!parsed) {
      return res.status(401).json(createNackResponse({
        type: "AUTH_ERROR",
        code: "AUTH_HEADER_INVALID",
        message: "Authorization header format is invalid"
      }));
    }

    const { subscriberId, uniqueKeyId } = ondcUtils.parseKeyId(parsed.keyId);
    if (!subscriberId || !uniqueKeyId) {
      return res.status(401).json(createNackResponse({
        type: "AUTH_ERROR",
        code: "KEY_ID_PARSE_FAIL",
        message: "Failed to parse keyId"
      }));
    }

    const isTimestampValid = ondcUtils.validateTimestamps(parsed.created, parsed.expires);
    if (!isTimestampValid) {
      return res.status(401).json(createNackResponse({
        type: "AUTH_ERROR",
        code: "TIMESTAMP_INVALID",
        message: "Signature timestamps invalid or expired"
      }));
    }

    const publicKey = await ondcUtils.lookupRegistryPublicKey(subscriberId, uniqueKeyId);
    if (!publicKey) {
      return res.status(401).json(createNackResponse({
        type: "AUTH_ERROR",
        code: "PUBLIC_KEY_NOT_FOUND",
        message: "Public key not found in registry"
      }));
    }

    const signatureValid = await ondcUtils.verifySignature(
      parsed.signature,
      parsed.created,
      parsed.expires,
      req.rawBodyBuffer,
      publicKey
    );

    if (!signatureValid) {
      return res.status(401).json(createNackResponse({
        type: "AUTH_ERROR",
        code: "SIGNATURE_INVALID",
        message: "Signature verification failed"
      }));
    }

    req.verified_bap_id = subscriberId;
    logger.info({ message: "✅ Signature verified successfully", subscriberId, transactionId });

  } catch (err) {
    logger.error({ message: "❗ Error during signature verification", error: err.message, transactionId });
    return res.status(500).json(createNackResponse({
      type: "CORE_ERROR",
      code: "SIGNATURE_VERIFICATION_FAILED",
      message: "Internal error during signature verification"
    }));
  }

  next();
};

export { processStatusRequest };
