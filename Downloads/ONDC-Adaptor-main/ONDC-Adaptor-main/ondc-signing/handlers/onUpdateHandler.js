const axios = require("axios");
const _ = require("lodash");
const logger = require("../utils/logger");
const { sendAck } = require("../utils/sendResponse");
const { getValue, setValue } = require("../utils/cache");

// --- OpenCart Configuration ---
const OPENCART_BASE_URL = process.env.OPENCART_BASE_URL;
const OPENCART_API_USER = process.env.OPENCART_API_USERNAME;
const OPENCART_API_KEY = process.env.OPENCART_API_KEY;
const OPENCART_API_TIMEOUT = parseInt(process.env.OPENCART_API_TIMEOUT || '5000');

// --- ONDC Status to OpenCart Status ID Mapping ---
const ONDC_TO_OPENCART_STATUS_MAP = {
    "Packed": parseInt(process.env.OPENCART_PACKED_STATUS_ID || '3'),
    "Shipped": parseInt(process.env.OPENCART_SHIPPED_STATUS_ID || '5'),
    "Out-for-delivery": parseInt(process.env.OPENCART_OFD_STATUS_ID || '15'),
    "Delivered": parseInt(process.env.OPENCART_DELIVERED_STATUS_ID || '7'),
    "Cancelled": parseInt(process.env.OPENCART_CANCELLED_STATUS_ID || '8'),
};

const getOpenCartStatusId = (ondcStatus) => {
    const statusId = ONDC_TO_OPENCART_STATUS_MAP[ondcStatus];
    if (statusId === undefined) {
        logger.warn({ message: "No OpenCart status ID mapping found for ONDC status", ondcStatus });
        return null;
    }
    return statusId;
};

const loginToOpenCart = async (transactionId) => {
    if (!OPENCART_API_USER || !OPENCART_API_KEY || !OPENCART_BASE_URL) {
        logger.error({ message: "Missing OpenCart API credentials or Base URL.", transactionId });
        return null;
    }

    try {
        logger.debug({ message: "Attempting OpenCart API login", transactionId });
        const response = await axios.post(`${OPENCART_BASE_URL}?route=api/login`, {
            username: OPENCART_API_USER,
            key: OPENCART_API_KEY,
        }, { timeout: OPENCART_API_TIMEOUT });

        const apiToken = response?.data?.api_token;
        if (apiToken) {
            logger.debug({ message: "OpenCart API login successful", transactionId });
            return apiToken;
        } else {
            logger.error({ message: "OpenCart login failed: No token in response", responseData: response?.data, transactionId });
            return null;
        }
    } catch (error) {
        logger.error({
            message: "OpenCart login request failed",
            error: error.isAxiosError ? error.toJSON?.() : error.message,
            transactionId
        });
        return null;
    }
};



const onUpdateHandler = async (req, res) => {
    const { body } = req;
    const context = body.context || {};
    const message = body.message || {};
    const transactionId = context.transaction_id;
    const messageId = context.message_id;
    const orderId = message.order?.id;

    const ackResponse = sendAck();
    const cacheKey = `on_update_ack:${transactionId}:${messageId}`;

    try {
        const cachedAck = await getValue(cacheKey);
        if (cachedAck) {
            logger.warn({ message: "Duplicate /on_update received. Sending cached ACK.", transactionId, messageId });
            return res.status(200).json(ackResponse);
        }
    } catch (cacheError) {
        logger.error({ message: 'Cache GET error for /on_update', error: cacheError.message, transactionId, messageId });
    }

    logger.info({ message: "Sending ACK for /on_update", transactionId, messageId, orderId });
    res.status(200).json(ackResponse);

    setImmediate(async () => {
        try {
            logger.info({ message: "Processing /on_update internally", transactionId, messageId, orderId });

            const apiToken = await loginToOpenCart(transactionId);
            if (!apiToken) return;

            const updated = await updateOpenCartOrderStatus(apiToken, orderId, message.order, transactionId);
            if (updated) {
                await setValue(cacheKey, true, 3600); // Prevent re-processing for 1 hour
                logger.info({ message: "Order update cached to prevent duplicate handling", transactionId, messageId });
            }
        } catch (err) {
            logger.error({ message: "Unhandled error during /on_update processing", error: err.message, transactionId, messageId });
        }
    });
};

module.exports = onUpdateHandler;
