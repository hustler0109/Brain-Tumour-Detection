import axios from "axios";
import _ from "lodash";
import logger from "../utils/logger.js";
import { sendAck } from "../utils/sendResponse.js";
import { getValue, setValue } from "../utils/cache.js";
import { OrderStatus } from "../constants.js";
import { validateSchema } from "../utils/schemaValidator.js";
import mysql from 'mysql2/promise';

// --- OpenCart Configuration ---
const OPENCART_BASE_URL = process.env.OPENCART_API_URL;
const OPENCART_API_USER = process.env.OPENCART_USERNAME;
const OPENCART_API_KEY = process.env.OPENCART_KEY;
const DEFAULT_ORDER_STATUS_ID = '2';
const API_TOKEN_CACHE_KEY = process.env.OPENCART_KEY;

// --- MySQL Configuration ---
const dbConfig = {
  host: process.env.MYSQL_HOST,
  user: process.env.MYSQL_USER,
  password: process.env.MYSQL_PASS,
  database: process.env.MYSQL_DB,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  connectTimeout: 10000,
  // Add retry configuration
  maxRetries: 3,
  retryDelay: 1000
};

// --- In-memory BAP Order Store ---
let bapOrders = {};

export const addBapOrder = (orderId, context, orderDetails) => {
  if (bapOrders[orderId]) {
    console.warn(`[BAP Store] Attempted to add duplicate OrderID: ${orderId}`);
    return;
  }
  console.log(`[BAP Store] Storing initiated order: ${orderId}`);
  bapOrders[orderId] = {
    originalOrderDetails: JSON.parse(JSON.stringify(orderDetails)),
    originalContext: JSON.parse(JSON.stringify(context)),
    status: OrderStatus.CONFIRM_SENT,
    onConfirmReceivedPayload: null,
    ackNackSent: null,
    nackReason: null,
    lastUpdatedAt: Date.now()
  };
};

export const getBapOrder = (orderId) => bapOrders[orderId];

export const resetBapOrder = (orderId) => {
  delete bapOrders[orderId];
};

// Map ONDC states to OpenCart order_status_id
const statusMapping = {
  'Processing': 2,
  'Shipped': 3,
  'Canceled': 7,
  'Complete': 5,
  'Denied': 8,
  'Canceled Reversal': 9,
  'Failed': 10,
  'Refunded': 11,
  'Reversed': 12,
  'Chargeback': 13,
  'Pending': 1,
  'Voided': 16
};

// Helper function to retry database connection
const connectWithRetry = async (config, retries = config.maxRetries) => {
  try {
    console.log(`Attempting MySQL connection (attempts remaining: ${retries})`);
    return await mysql.createConnection(config);
  } catch (error) {
    if (retries > 0) {
      console.log(`Connection failed, retrying in ${config.retryDelay}ms...`);
      await new Promise(resolve => setTimeout(resolve, config.retryDelay));
      return connectWithRetry(config, retries - 1);
    }
    throw error;
  }
};

// --- onConfirmHandler ---
const onConfirmHandler = async (req, res) => {
  let connection;
  try {
    console.log("MySQL Connection Config:", {
      host: process.env.MYSQL_HOST,
      user: process.env.MYSQL_USER,
      database: process.env.MYSQL_DB
    });

    // Create connection with retry
    connection = await connectWithRetry(dbConfig);
    console.log("MySQL connection successful");

    // Get the order status from the request
    const orderState = req.body?.message?.order?.state || 'Processing';
    const orderId = req.body?.message?.order?.id;

    // Get the corresponding OpenCart status ID
    const opencartStatusId = statusMapping[orderState] || 2;

    // Execute query to get status details
    const [rows] = await connection.execute(
      'SELECT os.order_status_id, os.name, os.language_id FROM oc_order_status os WHERE os.order_status_id = ?',
      [opencartStatusId]
    );
    
    console.log("Query executed successfully, found", rows.length, "rows");

    if (rows.length === 0) {
      return res.status(404).json({
        message: { ack: { status: "NACK" } },
        error: { code: "40400", message: "Order status not found" }
      });
    }

    // Respond with success
    res.status(200).json({
      message: { 
        ack: { status: "ACK" },
        order: {
          id: orderId,
          state: orderState,
          status_details: rows[0]
        }
      }
    });

  } catch (error) {
    console.error("Detailed DB Error:", {
      message: error.message,
      code: error.code,
      errno: error.errno,
      sqlState: error.sqlState,
      sqlMessage: error.sqlMessage,
      stack: error.stack
    });
    
    // Determine appropriate error message based on error type
    let errorMessage = "Database error occurred";
    if (error.code === 'ECONNREFUSED' || error.code === 'ECONNRESET') {
      errorMessage = "Could not connect to database. Please ensure MySQL server is running.";
    } else if (error.code === 'ER_ACCESS_DENIED_ERROR') {
      errorMessage = "Database access denied. Please check credentials.";
    } else if (error.code === 'ER_BAD_DB_ERROR') {
      errorMessage = "Database does not exist.";
    }
    
    res.status(500).json({
      message: { ack: { status: "NACK" } },
      error: { 
        code: "50001", 
        message: errorMessage,
        details: {
          code: error.code,
          errno: error.errno,
          sqlState: error.sqlState
        }
      }
    });
  } finally {
    if (connection) {
      try {
        await connection.end();
        console.log("MySQL connection closed");
      } catch (err) {
        console.error("Error closing MySQL connection:", err.message);
      }
    }
  }
};

export default onConfirmHandler;