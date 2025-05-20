import mysql from 'mysql2/promise';
import { OrderStatus } from "../constants.js";
import { sendAck } from "../utils/sendResponse.js";
import logger from "../utils/logger.js";

const dbConfig = {
  host: process.env.MYSQL_HOST,
  user: process.env.MYSQL_USER,
  password: process.env.MYSQL_PASS,
  database: process.env.MYSQL_DB,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  connectTimeout: 10000,
  maxRetries: 3,
  retryDelay: 1000
};

const connectWithRetry = async (config, retries = config.maxRetries) => {
  try {
    return await mysql.createConnection(config);
  } catch (error) {
    if (retries > 0) {
      await new Promise(resolve => setTimeout(resolve, config.retryDelay));
      return connectWithRetry(config, retries - 1);
    }
    throw error;
  }
};

const statusHandler = async (req, res) => {
  let connection;
  try {
    const orderId = req.body?.message?.order_id;
    if (!orderId) {
      return res.status(400).json({
        message: { ack: { status: "NACK" } },
        error: { code: "40000", message: "Missing order_id" }
      });
    }

    connection = await connectWithRetry(dbConfig);

    const [rows] = await connection.execute(
      'SELECT order_status_id FROM oc_order WHERE order_id = ?',
      [orderId]
    );

    if (rows.length === 0) {
      return res.status(404).json({
        message: { ack: { status: "NACK" } },
        error: { code: "40400", message: "Order not found" }
      });
    }

    const statusId = rows[0].order_status_id;
    const [statusDetails] = await connection.execute(
      'SELECT name FROM oc_order_status WHERE order_status_id = ?',
      [statusId]
    );

    res.status(200).json({
      message: {
        ack: { status: "ACK" },
        order: {
          id: orderId,
          status: {
            id: statusId,
            name: statusDetails[0]?.name || "Unknown"
          }
        }
      }
    });
  } catch (error) {
    logger.error("Status Handler Error:", error);
    res.status(500).json({
      message: { ack: { status: "NACK" } },
      error: {
        code: "50001",
        message: "Internal server error",
        details: error.message
      }
    });
  } finally {
    if (connection) await connection.end();
  }
};

export default statusHandler;

