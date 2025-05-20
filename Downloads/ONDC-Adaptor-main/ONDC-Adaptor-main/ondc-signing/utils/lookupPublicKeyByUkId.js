// utils/registryLookup.js
import axios from 'axios';
import config from './config.js';
import logger from './logger.js';

export const lookupPublicKeyByUkId = async (ukId) => {
  try {
    // Make registry request
    const response = await axios.post(
      `${config.ondc.registryUrl}/lookup`,
      {
        subscriber_id: config.ondc.subscriberId,
        ukId,
        domain: config.ondc.domain,
        country: config.ondc.country,
        city: config.ondc.city || "std:080",
        type: "BPP"
      },
      {
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.ondc.authToken}`
        },
        timeout: 5000
      }
    );

    // Validate response
    if (!response.data || !Array.isArray(response.data)) {
      logger.warn('Invalid response format from registry', { 
        status: response.status,
        data: response.data 
      });
      return null;
    }

    // Find matching subscriber
    const subscriberData = response.data.find(entry => entry.ukId === ukId);
    if (!subscriberData) {
      logger.warn('Subscriber not found in registry response', { 
        subscriberId: config.ondc.subscriberId, 
        ukId,
        availableUkIds: response.data.map(d => d.ukId)
      });
      return null;
    }

    // Validate required fields
    if (!subscriberData.signing_public_key) {
      logger.warn('Subscriber found but missing signing_public_key', { 
        subscriberId: config.ondc.subscriberId, 
        ukId 
      });
      return null;
    }

    logger.info('Found public key in registry', { 
      ukId, 
      subscriber_id: config.ondc.subscriberId 
    });
    return subscriberData.signing_public_key;

  } catch (error) {
    if (error.code === 'ECONNABORTED') {
      logger.error('Registry lookup timed out', { 
        subscriberId: config.ondc.subscriberId, 
        ukId,
        timeout: 5000 
      });
    } else if (error.response) {
      logger.error('Registry lookup failed with response', {
        status: error.response.status,
        data: error.response.data,
        subscriberId: config.ondc.subscriberId,
        ukId
      });
    } else {
      logger.error('Registry lookup failed', {
        error: error.message,
        subscriberId: config.ondc.subscriberId,
        ukId
      });
    }
    return null;
  }
};
