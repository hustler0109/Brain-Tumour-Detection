import axios from 'axios';
import { signRequest } from './cryptoUtils.js';
import config from '../utils/config.js';
import logger from '../utils/logger.js';

const makeAuthenticatedRequest = async (url, payload, options = {}) => {
  try {
    const { authHeader } = signRequest(payload);

    const headers = {
      'Content-Type': 'application/json',
      'Authorization': authHeader,
      ...options.headers
    };

    const response = await axios({
      method: options.method || 'POST',
      url,
      data: payload,
      headers,
      timeout: options.timeout || 30000
    });

    return response.data;
  } catch (error) {
    logger.error('Error making authenticated request', { error: error.message });
    throw error;
  }
};

export { makeAuthenticatedRequest };
