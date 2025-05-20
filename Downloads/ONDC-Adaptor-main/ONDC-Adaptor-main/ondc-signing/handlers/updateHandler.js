// import mysql from "mysql2/promise";
// import axios from "axios"; // Added axios for API calls
// import FormData from "form-data"; // Added FormData for API calls
// import logger from "../utils/logger.js";
// import { sendAck } from "../utils/sendResponse.js";
// import { getValue, setValue } from "../utils/cache.js";

// // --- MySQL Configuration ---
// const dbConfig = {
//   host: process.env.MYSQL_HOST,
//   user: process.env.MYSQL_USER,
//   password: process.env.MYSQL_PASS,
//   database: process.env.MYSQL_DB,
// };

// // --- OpenCart API Configuration ---
// // THESE ENVIRONMENT VARIABLES MUST BE SET AND CORRECTLY POINT TO YOUR OPENCART API
// const OPENCART_API_URL = process.env.OPENCART_API_URL; // Your OpenCart API base URL (e.g., http://localhost/opencartsite/index.php?route=api)
// const OPENCART_API_USERNAME = process.env.OPENCART_API_USERNAME; // Your OpenCart API username
// const OPENCART_API_KEY = process.env.OPENCART_API_KEY; // Your OpenCart API key

// // --- OpenCart Status, Reason, Action IDs (Based on provided SQL) ---
// // VERIFY these IDs match your specific OpenCart installation if it differs from the SQL dumps.
// const OPENCART_STATUS_IDS = {
//     PENDING: 1,
//     PROCESSING: 2,
//     SHIPPED: 3,
//     COMPLETE: 5,
//     CANCELED: 7,
//     DENIED: 8,
//     CANCELED_REVERSAL: 9,
//     FAILED: 10,
//     REFUNDED: 11,
//     REVERSED: 12,
//     CHARGEBACK: 13,
//     PROCESSED: 15,
//     VOIDED: 16,
//     EXPIRED: 14,
//     // Add any other custom order statuses you have if not in the SQL
// };

// const OPENCART_RETURN_REASON_IDS = {
//     DEAD_ON_ARRIVAL: 1,
//     RECEIVED_WRONG_ITEM: 2,
//     ORDER_ERROR: 3,
//     FAULTY: 4,
//     OTHER: 5,
//     // Add any other custom return reasons you have if not in the SQL
// };

// const OPENCART_RETURN_ACTION_IDS = {
//     REFUNDED: 1,
//     CREDIT_ISSUED: 2,
//     REPLACEMENT_SENT: 3,
//     // Add any other custom return actions you have if not in the SQL
// };

// const OPENCART_RETURN_STATUS_IDS = {
//     PENDING: 1,
//     AWAYTING_PRODUCTS: 2,
//     COMPLETE: 3,
//     // Add any other custom return statuses you have if not in the SQL
// };


// // --- Helper to get OpenCart API Token ---
// const getApiToken = async () => {
//     const loginData = new FormData();
//     loginData.append("username", OPENCART_API_USERNAME);
//     loginData.append("key", OPENCART_API_KEY);

//     try {
//         // Standard OpenCart API login route
//         const loginResponse = await axios.post(
//             `${OPENCART_API_URL}/login`, // Assuming API_URL ends before /login
//             loginData,
//             { headers: loginData.getHeaders() }
//         );
//         if (loginResponse.data && loginResponse.data.api_token) {
//             logger.info("OpenCart API login successful.");
//             return loginResponse.data.api_token;
//         } else {
//             logger.error("OpenCart API login failed: No token received.", loginResponse.data);
//             throw new Error("OpenCart API login failed");
//         }
//     } catch (error) {
//         logger.error({ message: "Error during OpenCart API login", error: error.message, stack: error.stack });
//         throw new Error("Failed to obtain OpenCart API token");
//     }
// };


// // --- ORDER-Level Update (e.g., Cancellation) ---
// // Assumes update_target 'order' means full order update/cancellation
// const handleOrderUpdate = async (connection, orderId, order, transactionId) => {
//   try {
//     const cancelReasonTag = order.tags?.find(t => t.code === "CANCELLATION_REASON");
//     const cancelCommentTag = order.tags?.find(t => t.code === "CANCELLATION_REASON_COMMENT");

//     const reason = cancelReasonTag?.value || "Order update received";
//     const comment = cancelCommentTag?.value || "";
//     const historyComment = `Order updated: ${reason}${comment ? `: ${comment}` : ''}`;

//     // --- Determine New Order Status ---
//     // Map incoming order state (based on presence of cancellation tags) to a specific oc_order_status_id.
//     let newOrderStatusId = null;
//      if (cancelReasonTag) {
//          newOrderStatusId = OPENCART_STATUS_IDS.CANCELED; // Map to Canceled status from provided SQL
//      }
//      // Add more conditions here for other potential order-level states from payload
//      // Example: if (order.state === 'complete') newOrderStatusId = OPENCART_STATUS_IDS.COMPLETE;


//     logger.info({ message: `Processing ORDER update. New status ID determined: ${newOrderStatusId}`, orderId, transactionId });

//     // Always add history entry for the update event
//      await connection.execute(
//        `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//         VALUES (?, ?, ?, ?, NOW())`,
//        [orderId, newOrderStatusId !== null ? newOrderStatusId : 0, 0, historyComment] // Use new status if mapped, otherwise 0 or current
//      );

//     if (newOrderStatusId !== null) {
//          // Update main order status if a specific status was determined
//          await connection.execute(
//            `UPDATE oc_order SET order_status_id = ? WHERE order_id = ?`,
//            [newOrderStatusId, orderId]
//          );
//          logger.info({ message: `Updated main order status to ${newOrderStatusId}`, orderId, transactionId });
//     } else {
//         logger.warn({ message: "ORDER update received but no specific order_status_id mapped from payload", orderId, transactionId, payload: order });
//     }


//     logger.info({ message: "Processed ORDER update handler", orderId, transactionId });

//   } catch (error) {
//     logger.error({ message: "Error handling ORDER update", orderId, transactionId, error: error.message, stack: error.stack });
//     throw error; // Re-throw to be caught by the main handler's try/catch
//   }
// };


// // --- ITEM-Level Cancellation/Return ---
// // Handles item updates by calling the OpenCart API to create/update return requests.
// const handleItemUpdate = async (connection, orderId, fulfillments, transactionId) => {
//   try {
//     // Get API token
//     const apiToken = await getApiToken();

//     // Assuming item updates relevant to returns/cancellations come within fulfillments structure
//     for (const fulfillment of fulfillments) {
//       const returnTags = fulfillment.tags?.find(t => t.code === "return_request")?.list || [];
//       const itemIdTag = returnTags.find(t => t.code === "item_id"); // Item ID from tags
//       const returnReasonTag = returnTags.find(t => t.code === "RETURN_REASON"); // Return Reason Code
//       const returnCommentTag = returnTags.find(t => t.code === "RETURN_COMMENT"); // Comment
//       const returnQuantityTag = returnTags.find(t => t.code === "return_quantity"); // Quantity to return


//       // Try to get item ID (product ID) from payload
//       const productId = fulfillment.items?.[0]?.id || itemIdTag?.value;

//       if (!productId) {
//         logger.warn({ message: "Skipping item update processing for fulfillment due to missing product ID", orderId, transactionId, fulfillmentId: fulfillment.id });
//         continue; // Skip this fulfillment entry if no product ID
//       }

//       // --- Extract Return Details ---
//       const reasonComment = returnCommentTag?.value || "";
//       const returnQuantity = parseInt(returnQuantityTag?.value, 10) || 1; // Default to 1 if not specified
//       const incomingReasonCode = returnReasonTag?.value; // The reason code from the platform


//       // --- Map incoming reason code to OpenCart return_reason_id ---
//       // Map based on your platform's codes and the 'name' column in oc_return_reason table
//       let returnReasonId = OPENCART_RETURN_REASON_IDS.OTHER; // Defaulting to 'Other'
//       if (incomingReasonCode === 'Dead On Arrival') returnReasonId = OPENCART_RETURN_REASON_IDS.DEAD_ON_ARRIVAL;
//       if (incomingReasonCode === 'Received Wrong Item') returnReasonId = OPENCART_RETURN_REASON_IDS.RECEIVED_WRONG_ITEM;
//       if (incomingReasonCode === 'Order Error') returnReasonId = OPENCART_RETURN_REASON_IDS.ORDER_ERROR;
//       if (incomingReasonCode === 'Faulty, please supply details') returnReasonId = OPENCART_RETURN_REASON_IDS.FAULTY;
//       // Add more mappings here based on your platform's codes and OpenCart's return_reason_id names


//       // --- Map incoming status/action to OpenCart return_status_id and return_action_id ---
//       // Map based on your platform's states/actions and the 'name' column in oc_return_status/oc_return_action tables
//       // Assuming the item update implies a 'Pending' return status and 'Refund' action by default
//       let returnStatusId = OPENCART_RETURN_STATUS_IDS.PENDING; // Defaulting to 'Pending'
//       let returnActionId = OPENCART_RETURN_ACTION_IDS.REFUNDED; // Defaulting to 'Refunded'
//        // You might need more sophisticated logic here based on fulfillment.state or other tags
//        // Example: if (fulfillment.state === 'return_completed') returnStatusId = OPENCART_RETURN_STATUS_IDS.COMPLETE;


//       logger.info(`🟡 Processing item update for product ${productId} (Order ID: ${orderId}). Preparing data for OpenCart Return API.`);
//       logger.debug(`Return details: ReasonCode=${incomingReasonCode} (Mapped ID: ${returnReasonId}), Quantity=${returnQuantity}, Comment="${reasonComment}", Mapped Status ID: ${returnStatusId}, Mapped Action ID: ${returnActionId}`);

//       // --- Prepare FormData for OpenCart API ---
//       const formData = new FormData();
//       formData.append("order_id", orderId); // Order ID is required by the API
//       // The API likely needs customer, product, model, etc.
//       // We need to fetch these from the DB first, similar to the previous version,
//       // as the incoming payload might not have all details required by the API.
//       const [orderRows] = await connection.execute(`SELECT customer_id, firstname, lastname, email, telephone, date_added FROM oc_order WHERE order_id = ?`, [orderId]);
//       const [productRows] = await connection.execute(`SELECT product_id, name, model FROM oc_order_product WHERE order_id = ? AND product_id = ? LIMIT 1`, [orderId, productId]); // Fetch from order_product using order_id and product_id


//       if (orderRows.length === 0 || productRows.length === 0) {
//            logger.error({ message: `Could not retrieve original order or product details for return API call. order_id: ${orderId}, product_id: ${productId}`, transactionId });
//            // Log to order history that return creation failed
//            if (connection && orderId) {
//               try {
//                   await connection.execute(
//                      `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//                       VALUES (?, ?, ?, ?, NOW())`,
//                      [orderId, 0, 0, `Failed to create return for product ${productId}: Could not retrieve order/product details.`]
//                   );
//               } catch (historyError) {
//                   logger.error({ message: "Failed to log return creation failure in history", orderId, transactionId, historyError: historyError.message });
//               }
//            }
//            continue; // Cannot call API without details
//       }
//        const orderDetails = orderRows[0];
//        const productDetails = productRows[0];

//       // Append data to FormData for the API call (using parameters from friend's code)
//       formData.append("customer_id", orderDetails.customer_id);
//       formData.append("firstname", orderDetails.firstname);
//       formData.append("lastname", orderDetails.lastname);
//       formData.append("email", orderDetails.email);
//       formData.append("telephone", orderDetails.telephone);
//       formData.append("product", productDetails.name); // Use product name from order_product
//       formData.append("model", productDetails.model); // Use model from order_product
//       formData.append("product_id", productDetails.product_id); // Include product_id as well
//       formData.append("quantity", returnQuantity); // Quantity from payload
//       formData.append("opened", 1); // Assuming 'opened' status (1 for Opened) - VERIFY THIS ID with your API/Module
//       formData.append("return_reason_id", returnReasonId); // Mapped reason ID
//       formData.append("return_action_id", returnActionId); // Mapped action ID
//       formData.append("return_status_id", returnStatusId); // Initial status ID
//       formData.append("comment", reasonComment);
//       // Note: Friend's code snippet did NOT include 'date_ordered' in FormData.
//       // If your API requires it, you'll need to add:
//       // formData.append("date_ordered", orderDetails.date_added ? new Date(orderDetails.date_added).toISOString().split('T')[0] : '0000-00-00');


//       // --- Call OpenCart API to add Return ---
//       try {
//           // ** IMPORTANT: VERIFY THIS API ROUTE AND EXPECTED PARAMETERS **
//           // Using the route from your friend's code: api/return/add
//           const apiResponse = await axios.post(
//               `${OPENCART_API_URL}/return/add&api_token=${apiToken}`,
//               formData,
//               { headers: formData.getHeaders() }
//           );

//           logger.info({ message: "OpenCart Return API response", orderId, transactionId, productId, apiResponse: apiResponse.data });

//           // You might want to check the API response data here for success/failure indicators
//           // and potentially log a specific history entry based on the API's response.
//           if (apiResponse.data && apiResponse.data.success) {
//              logger.info({ message: `Successfully created return entry for product ${productId} via API.`, orderId });
//           } else {
//              logger.warn({ message: `OpenCart Return API call did not report success for product ${productId}.`, orderId, productId, responseData: apiResponse.data });
//               // Log non-success API response to history
//               if (connection && orderId) {
//                   try {
//                       const responseMessage = JSON.stringify(apiResponse.data);
//                       await connection.execute(
//                          `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//                           VALUES (?, ?, ?, ?, NOW())`,
//                          [orderId, 0, 0, `Return API call for product ${productId} did not succeed. Response: ${responseMessage}`]
//                       );
//                   } catch (historyError) {
//                       logger.error({ message: "Failed to log non-success API response in history", orderId, transactionId, historyError: historyError.message });
//                   }
//                }
//           }


//       } catch (apiError) {
//           logger.error({ message: "Error calling OpenCart Return API", orderId, transactionId, productId, error: apiError.message, stack: apiError.stack, apiResponseData: apiError.response?.data });
//           // Log to order history that return creation failed via API
//            if (connection && orderId) {
//               try {
//                   const errorMessage = apiError.response?.data?.error || apiError.message;
//                   await connection.execute(
//                      `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//                       VALUES (?, ?, ?, ?, NOW())`,
//                      [orderId, 0, 0, `Failed to create return for product ${productId} via API: ${errorMessage}`]
//                   );
//               } catch (historyError) {
//                   logger.error({ message: "Failed to log API error in history", orderId, transactionId, historyError: historyError.message });
//               }
//            }
//           // Decide how else to handle API errors - re-throwing might stop processing other items
//           // For now, we log and continue processing other items if any.
//       }
//     }

//     // Note: No logic here to update the main order status based on item returns.
//     // If you need to update the main order status when ALL items are returned,
//     // you would need a separate check after processing all items, potentially
//     // querying the oc_return table for this order and checking their statuses.

//   } catch (error) {
//     logger.error({ message: "Error handling ITEM (Return) update process", orderId, transactionId, error: error.message, stack: error.stack });
//     throw error; // Re-throw to be caught by the main handler's try/catch
//   }
// };


// // --- FULFILLMENT-Level Update ---
// // Updates main order status and history based on fulfillment state.
// const handleFulfillmentUpdate = async (connection, orderId, fulfillments, transactionId) => {
//   try {
//     for (const fulfillment of fulfillments) {
//       const fulfillmentId = fulfillment.id || "unknown";
//       const state = fulfillment.state || "updated"; // e.g., 'packed', 'shipped', 'delivered'
//       const comment = `Fulfillment ${fulfillmentId} status: ${state}`;

//       // --- Determine New Order Status ---
//       // Map incoming fulfillment state to an OpenCart order_status_id.
//       // **EXAMPLE MAPPING:** Update this based on your OpenCart -> Platform state mapping
//       let newOrderStatusId = null;
//       // Convert incoming state to uppercase for case-insensitive comparison
//       const stateUpperCase = state.toUpperCase();

//       switch(stateUpperCase) {
//           case 'PACKED':
//               newOrderStatusId = OPENCART_STATUS_IDS.PROCESSING; // Assuming packed means processing - VERIFY THIS ID
//               break;
//           case 'ORDER-PICKED-UP':
//           case 'SHIPPED':
//               newOrderStatusId = OPENCART_STATUS_IDS.SHIPPED; // Map to Shipped - VERIFY THIS ID
//               break;
//           case 'DELIVERED':
//               newOrderStatusId = OPENCART_STATUS_IDS.COMPLETE; // Map to Complete - VERIFY THIS ID
//               break;
//           case 'CANCELLED':
//               newOrderStatusId = OPENCART_STATUS_IDS.CANCELED; // Map to Canceled - VERIFY THIS ID
//               break;
//            case 'RTO_INITIATED':
//            case 'RTO_DELIVERED':
//                newOrderStatusId = OPENCART_STATUS_IDS.CHARGEBACK; // Assuming RTO maps to Chargeback or similar 'Returned' status - VERIFY THIS ID
//                break;
//           // Add more cases as needed based on your platform's fulfillment states
//           default:
//                logger.warn({ message: `Mapping for unknown fulfillment state "${state}". Uppercase: "${stateUpperCase}" is not defined.`, orderId, transactionId, fulfillmentId });
//                // Optionally set a default status or skip main status update
//                // newOrderStatusId = OPENCART_STATUS_IDS.PROCESSING; // Example default
//       }


//       logger.info(`🟦 Processing fulfillment status: order=${orderId}, fulfillment=${fulfillmentId}, status=${state}. Mapped status ID: ${newOrderStatusId}`);

//       // Always add history entry for the fulfillment event
//        await connection.execute(
//           `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//            VALUES (?, ?, ?, ?, NOW())`,
//           [orderId, newOrderStatusId !== null ? newOrderStatusId : 0, 0, comment] // Use mapped status if available, otherwise 0 or current
//        );

//       if (newOrderStatusId !== null) {
//            // Update main order status if a specific status was mapped
//            await connection.execute(
//              `UPDATE oc_order SET order_status_id = ? WHERE order_id = ?`,
//              [newOrderStatusId, orderId]
//            );
//             logger.info({ message: `Updated main order status to ${newOrderStatusId}`, orderId, transactionId });
//       } else {
//           logger.warn({ message: `Fulfillment update received but no specific order_status_id mapped for state "${state}"`, orderId, transactionId, fulfillmentId });
//       }
//     }

//   } catch (error) {
//     logger.error({ message: "Error handling FULFILLMENT update", orderId, transactionId, error: error.message, stack: error.stack });
//     throw error;
//   }
// };

// // --- PAYMENT-Level Update ---
// // Updates main order status and history based on payment status.
// const handlePaymentUpdate = async (connection, orderId, payments, transactionId) => {
//   try {
//     logger.info("👉 Payments received in handler:", payments);

//     for (const payment of payments) {
//       const type = payment.type || "unknown"; // e.g., "ONORDER", "PREORDER"
//       const status = payment.status || "initiated"; // e.g., "PAID", "REFUNDED", "PARTIALLY_REFUNDED"
//       const amount = payment.params?.amount || 0; // Amount related to this specific payment event
//       const comment = `Payment type ${type}, Amount ₹${amount}, Status: ${status}`;

//       // --- Determine New Order Status ---
//       // Map incoming payment status to an OpenCart order_status_id.
//       // **EXAMPLE MAPPING:** Update this based on your OpenCart -> Platform status mapping
//       // Convert incoming status to uppercase for case-insensitive comparison
//       const statusUpperCase = status.toUpperCase();
//       let newOrderStatusId = null;

//        switch(statusUpperCase) {
//            case 'PAID':
//                // If an order was pending and payment is now confirmed
//                newOrderStatusId = OPENCART_STATUS_IDS.PROCESSED; // Assuming PAID maps to Processed - VERIFY THIS ID
//                break;
//            case 'REFUNDED': // Full refund
//                newOrderStatusId = OPENCART_STATUS_IDS.REFUNDED; // Map to Refunded - VERIFY THIS ID
//                break;
//            case 'PARTIALLY_REFUNDED':
//                // Map to a specific 'Partially Refunded' status if you have one, or log only
//                // newOrderStatusId = YOUR_PARTIAL_REFUND_STATUS_ID; // VERIFY THIS ID
//                break;
//            // Add more cases as needed based on your platform's payment statuses
//            default:
//                logger.warn({ message: `Payment update received but no specific order_status_id mapped for status "${status}". Uppercase: "${statusUpperCase}"`, orderId, transactionId });
//                 // Optionally set a default status or skip main status update
//                 // newOrderStatusId = OPENCART_STATUS_IDS.PROCESSING; // Example default
//        }

//       logger.info(`🟢 Processing payment/refund update: order=${orderId}, type=${type}, amount=${amount}, status=${status}. Mapped status ID: ${newOrderStatusId}`);

//        // Always add history entry for the payment event
//        await connection.execute(
//           `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//            VALUES (?, ?, ?, ?, NOW())`,
//           [orderId, newOrderStatusId !== null ? newOrderStatusId : 0, 0, comment] // Use mapped status if available, otherwise 0 or current
//        );

//        // Update main order status ONLY if a specific status was mapped
//        if (newOrderStatusId !== null) {
//            await connection.execute(
//              `UPDATE oc_order SET order_status_id = ? WHERE order_id = ?`,
//              [newOrderStatusId, orderId]
//            );
//            logger.info({ message: `Updated main order status to ${newOrderStatusId}`, orderId, transactionId });
//        } else {
//            logger.warn({ message: `Payment update received but no specific order_status_id mapped for status "${status}"`, orderId, transactionId });
//        }
//     }

//   } catch (error) {
//     logger.error({ message: "Error handling PAYMENT update", orderId, transactionId, error: error.message, stack: error.stack });
//     throw error;
//   }
// };


// // --- MAIN UPDATE HANDLER ---
// const updateHandler = async (req, res) => {
//   const { body } = req;
//   const context = body.context || {};
//   const message = body.message || {};
//   const updateTarget = message.update_target;
//   const orderId = message.order?.id;
//   const transactionId = context.transaction_id;
//   const messageId = context.message_id;

//   // Prepare the ACK response
//   const ackResponse = sendAck({
//     transaction_id: transactionId,
//     message_id: messageId,
//     action: "on_update",
//     timestamp: new Date().toISOString(),
//   });

//   const cacheKey = `on_update_ack:${transactionId}:${messageId}`;

//   // Check cache for idempotency BEFORE processing
//   try {
//     const cachedAck = await getValue(cacheKey);
//     if (cachedAck) {
//       logger.info({ message: "ACK already sent for this message", transactionId, messageId });
//       return res.status(200).json(ackResponse);
//     }
//   } catch (err) {
//     // Log cache error but continue processing in case cache is down
//     logger.error({ message: "Error reading ACK cache", err: err.message, transactionId, messageId });
//   }

//   // Send ACK response immediately
//   res.status(200).json(ackResponse);

//   // Process the update asynchronously
//   setImmediate(async () => {
//     let connection; // Declare connection variable outside try block
//     try {
//       if (!orderId) {
//         logger.error({ message: "Missing order ID in update payload", transactionId, messageId, updateTarget });
//         return; // Cannot process without order ID
//       }

//       logger.info({ message: "Handling update", updateTarget, transactionId, orderId, messageId });

//       // Establish DB connection once for this message processing
//       connection = await mysql.createConnection(dbConfig);

//       // Dispatch based on update target
//       switch (updateTarget) {
//         case "order":
//           // Handles full order updates/cancellations
//           await handleOrderUpdate(connection, orderId, message.order, transactionId);
//           break;

//         case "item":
//           // Handles item updates, mapping to OpenCart Returns via API
//           // Note: Item updates often come within fulfillments structure in some payloads
//           await handleItemUpdate(connection, orderId, message.order?.fulfillments || [], transactionId);
//           break;

//         case "fulfillment":
//            // Handles fulfillment updates, updating main order status
//            await handleFulfillmentUpdate(connection, orderId, message.order?.fulfillments || [], transactionId);
//           break;

//         case "payment":
//            // Handles payment updates, updating main order status
//            await handlePaymentUpdate(connection, orderId, message.order?.payments || [], transactionId);
//           break;

//         default:
//           logger.warn({
//             message: "Unknown or unsupported update_target",
//             updateTarget,
//             transactionId,
//             orderId,
//             messageId
//           });
//            // Log history entry for unknown updates if connection is available
//            if (connection) {
//              try {
//                 await connection.execute(
//                    `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//                     VALUES (?, ?, ?, ?, NOW())`,
//                    [orderId, 0, 0, `Unknown update_target "${updateTarget}" received.`] // Use status 0 or current if available
//                 );
//              } catch (historyError) {
//                 logger.error({ message: "Failed to log unknown update target in history", orderId, transactionId, historyError: historyError.message });
//              }
//            }
//       }

//       // Cache the successful processing for 1 hour (3600 seconds)
//       await setValue(cacheKey, true, 3600);
//       logger.info({ message: "Processed /update and cached ACK", transactionId, orderId, messageId, updateTarget });

//     } catch (error) {
//       // Log any errors during asynchronous processing
//       logger.error({
//         message: "Error processing /update asynchronously",
//         error: error.message,
//         transactionId,
//         orderId,
//         messageId,
//         updateTarget,
//         stack: error.stack
//       });
//        // Log a history entry about the processing error if connection and orderId exist
//        if (connection && orderId) {
//           try {
//               await connection.execute(
//                  `INSERT INTO oc_order_history (order_id, order_status_id, notify, comment, date_added)
//                   VALUES (?, ?, ?, ?, NOW())`,
//                  [orderId, 0, 0, `Error processing update_target "${updateTarget}": ${error.message}`] // Use status 0 or current if available
//               );
//           } catch (historyError) {
//               logger.error({ message: "Failed to log error in order history", orderId, transactionId, historyError: historyError.message });
//           }
//        }
//     } finally {
//       // Ensure the database connection is closed
//       if (connection) {
//         try {
//           await connection.end();
//           // logger.info({ message: "Database connection closed", transactionId, orderId, messageId });
//         } catch (closeError) {
//           logger.error({ message: "Error closing database connection", error: closeError.message, transactionId, orderId, messageId });
//         }
//       }
//     }
//   });
// };

// export default updateHandler;




// handlers/onUpdateHandler.js

import axios from "axios";
import logger from "../utils/logger.js";
import { sendAck, sendNack } from "../utils/sendResponse.js";
import { getValue, setValue } from "../utils/cache.js";

// Map ONDC fulfillment state codes to OpenCart status IDs
const FULFILLMENT_STATE_TO_STATUS_ID = {
  "Order-delivered": 5,       // COMPLETE
  "Order-picked-up": 3,       // SHIPPED
  "Cancelled": 7,             // CANCELED
  "Out-for-delivery": 2,      // PROCESSING
};

const OPENCART_API_URL = process.env.OPENCART_API_URL;
const OPENCART_API_USERNAME = process.env.OPENCART_API_USERNAME;
const OPENCART_API_KEY = process.env.OPENCART_API_KEY;

const onUpdateHandler = async (req, res) => {
  const { context = {}, message = {} } = req.body;
  const transactionId = context.transaction_id;
  const messageId = context.message_id;
  const orderId = message.order?.id;
  const fulfillments = message.order?.fulfillments;

  const cacheKey = `on_update_ack:${transactionId}:${messageId}`;
  const ackResponse = sendAck({
    transaction_id: transactionId,
    message_id: messageId,
    action: "on_update",
    timestamp: new Date().toISOString(),
  });

  try {
    const cachedAck = await getValue(cacheKey);
    if (cachedAck) {
      logger.warn({ message: "Duplicate /on_update received. Sending cached ACK.", transactionId });
      return res.status(200).json(ackResponse);
    }
  } catch (err) {
    logger.error({ message: "Error reading ACK cache", error: err.message });
  }

  if (!orderId) {
    logger.error({ message: "Missing order ID in on_update payload", transactionId });
    return res.status(400).json({
      message: { ack: { status: "NACK" } },
      error: { code: "40000", message: "Missing order.id in message" },
    });
  }

  res.status(200).json(ackResponse);

  setImmediate(async () => {
    try {
      const mainFulfillment = fulfillments?.[0];
      const stateCode = mainFulfillment?.state?.descriptor?.code;
      const statusId = FULFILLMENT_STATE_TO_STATUS_ID[stateCode];

      if (!statusId) {
        logger.warn({ message: "Unknown fulfillment state code", stateCode });
        return;
      }

      // Optional: Login to API if needed
      const form = new FormData();
      form.append("username", OPENCART_API_USERNAME);
      form.append("key", OPENCART_API_KEY);

      const loginRes = await axios.post(`${OPENCART_API_URL}/login`, form, {
        headers: form.getHeaders(),
      });

      const token = loginRes.data.api_token;

      const updateForm = new FormData();
      updateForm.append("order_status_id", statusId);
      updateForm.append("notify", "0");
      updateForm.append("comment", `Status updated via /on_update to '${stateCode}'`);

      await axios.post(
        `${OPENCART_API_URL}/order/history&api_token=${token}&order_id=${orderId}`,
        updateForm,
        { headers: updateForm.getHeaders() }
      );

      await setValue(cacheKey, true, 3600);
      logger.info({ message: "on_update handled and order updated", orderId, transactionId });
    } catch (err) {
      logger.error({ message: "Error processing /on_update", error: err.message, transactionId });
    }
  });
};

export default onUpdateHandler;
