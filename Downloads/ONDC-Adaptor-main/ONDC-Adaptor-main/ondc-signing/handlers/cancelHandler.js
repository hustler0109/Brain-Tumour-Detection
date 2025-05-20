import mysql from "mysql2/promise";
import logger from "../utils/logger.js";
import { sendAck } from "../utils/sendResponse.js";
import { getValue, setValue } from "../utils/cache.js";

const dbConfig = {
  host: process.env.MYSQL_HOST,
  user: process.env.MYSQL_USER,
  password: process.env.MYSQL_PASS,
  database: process.env.MYSQL_DB,
};

const OPENCART_CANCELED_STATUS_ID = parseInt(process.env.OPENCART_CANCELLED_STATUS_ID || '7');

const cancelHandler = async (req, res) => {
  const { body } = req;
  const context = body.context || {};
  const message = body.message || {};
  const transactionId = context.transaction_id;
  const messageId = context.message_id;
  const orderId = message.order_id;
  const cancellationReason = message.cancellation_reason_id || "Not provided";

  const ackResponse = sendAck({
    transaction_id: transactionId,
    message_id: messageId,
    action: "on_cancel",
    timestamp: new Date().toISOString()
  });

  const cacheKey = `on_cancel_ack:${transactionId}:${messageId}`;

  try {
    const cachedAck = await getValue(cacheKey);
    if (cachedAck) {
      logger.warn({ message: "Duplicate /on_cancel received. Sending cached ACK.", transactionId, messageId });
      return res.status(200).json(ackResponse);
    }
  } catch (err) {
    logger.error({ message: "Error reading ACK cache", err: err.message });
  }

  res.status(200).json(ackResponse);

  setImmediate(async () => {
    let connection;
    try {
      if (!orderId) {
        logger.error({ message: "Missing order ID in cancel payload", transactionId });
        return;
      }

      logger.info({ message: "Processing cancellation", transactionId, orderId });

      connection = await mysql.createConnection(dbConfig);

      await connection.execute(
        `UPDATE oc_order SET order_status_id = ? WHERE order_id = ?`,
        [OPENCART_CANCELED_STATUS_ID, orderId]
      );

      await connection.execute(
        `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
         VALUES (?, ?, ?, ?, NOW())`,
        [orderId, OPENCART_CANCELED_STATUS_ID, 0, `Cancellation reason: ${cancellationReason}`]
      );

      await setValue(cacheKey, true, 3600);
      logger.info({ message: "Cancellation recorded and ACK cached", orderId, transactionId });

    } catch (error) {
      logger.error({ message: "Error processing cancellation", error: error.message, transactionId, orderId });
    } finally {
      if (connection) {
        try {
          await connection.end();
        } catch (err) {
          logger.error({ message: "Error closing DB connection", error: err.message });
        }
      }
    }
  });
};

export default cancelHandler;
