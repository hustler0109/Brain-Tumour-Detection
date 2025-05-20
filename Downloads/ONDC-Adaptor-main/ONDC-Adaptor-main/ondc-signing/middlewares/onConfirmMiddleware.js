import logger from "../utils/logger.js";
import axios from 'axios';
import FormData from 'form-data';

export default async function onConfirmMiddleware(req, res, next) {
    const { body } = req;
    const transactionId = body?.context?.transaction_id;
    const messageId = body?.context?.message_id;
    const orderId = body?.message?.order?.id;
    const orderState = body?.message?.order?.state;

    logger.info({ 
        message: "Middleware triggered for /on_confirm",
        transactionId,
        messageId,
        orderId,
        orderState,
        timestamp: new Date().toISOString()
    });

    try {
        const authCookie = req.cookies;
        logger.debug({ message: "Cookies received", cookies: authCookie });

        // Extract opencart api_token from cookies
        const token = authCookie?.api_token;
        const isTokenPresent = !!token;

        if(!isTokenPresent) {
            try {
                const loginData = new FormData();
                loginData.append("username", process.env.OPENCART_USERNAME);
                loginData.append("key", process.env.OPENCART_KEY);

                const loginResponse = await axios.post(
                    `${process.env.OPENCART_API_URL}/index.php?route=api/login`,
                    loginData,
                    {
                        timeout: 5000,
                        headers: {
                            ...loginData.getHeaders()
                        }
                    }
                );

                if (!loginResponse.data?.api_token) {
                    logger.error({ 
                        message: "OpenCart login failed - no token in response",
                        transactionId,
                        messageId
                    });
                    req.isValidRequest = false;
                    return next();
                }

                const apiToken = loginResponse.data.api_token;
                logger.info({ 
                    message: "Login success. API Token received",
                    transactionId,
                    messageId
                });

                res.cookie("api_token", apiToken, { httpOnly: true, maxAge: 3600000 });
                req.cookies.api_token = apiToken;
            } catch (error) {
                logger.error({ 
                    message: "OpenCart login failed", 
                    error: error.message,
                    response: error.response?.data,
                    transactionId,
                    messageId
                });
                req.isValidRequest = false;
                return next();
            }
        }

        // Extract api_token from cookies
        const apiToken = authCookie?.api_token;
        
        // Check if .env has necessary credentials
        const hasCredentials = process.env.OPENCART_USERNAME && process.env.OPENCART_KEY;

        // Validation logic
        const isValidRequest = !!(apiToken && hasCredentials);

        req.isValidRequest = isValidRequest;

        if (isValidRequest) {
            logger.info('Valid request. Proceeding...');
        } else {
            logger.warn('Invalid request. Missing Opencart token or env credentials.');
        }

        // Log request details
        logger.debug({
            message: "Request details for /on_confirm",
            headers: req.headers,
            body: {
                context: body?.context,
                message: {
                    order: {
                        id: orderId,
                        state: orderState,
                        items: body?.message?.order?.items?.length || 0
                    }
                }
            }
        });

        next();
    } catch (error) {
        logger.error({ 
            message: "Middleware Error", 
            error: error.message,
            transactionId,
            messageId
        });
        req.isValidRequest = false;
        next();
    }
}
