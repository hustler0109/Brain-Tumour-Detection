import axios from 'axios';
import NodeCache from 'node-cache';
import logger from '../utils/logger.js';
import config from '../utils/config.js';

// Cache with 5 minute TTL and 1 minute check period
const registryCache = new NodeCache({ 
  stdTTL: 300,  // 5 minutes
  checkperiod: 60,  // 1 minute
  useClones: false // Store references to avoid memory issues
});

const REGISTRY_TIMEOUT = 5000; // 5 seconds

const lookupSubscriber = async (subscriberId, ukId) => {
  const cacheKey = `${subscriberId}:${ukId}`;
  
  try {
    // Check cache first
    const cachedData = registryCache.get(cacheKey);
    if (cachedData) {
      logger.debug('Retrieved subscriber from cache', { subscriberId, ukId });
      return cachedData;
    }

    // Validate inputs
    if (!ukId) {
      logger.warn('Invalid ukId provided to lookupSubscriber');
      return null;
    }

    // Make registry request
    const response = await axios.post(
      `${config.ondc.registryUrl}/lookup`,
      {
        subscriber_id: subscriberId,
        ukId,
        domain: config.ondc.domain,
        country: config.ondc.country,
        city: config.ondc.city || "std:080",
        type: "BPP"
      },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: REGISTRY_TIMEOUT
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
        subscriberId, 
        ukId,
        availableUkIds: response.data.map(d => d.ukId)
      });
      return null;
    }

    // Validate required fields
    if (!subscriberData.signing_public_key) {
      logger.warn('Subscriber found but missing signing_public_key', { 
        subscriberId, 
        ukId 
      });
      return null;
    }

    // Cache the result
    registryCache.set(cacheKey, subscriberData);
    logger.info('Subscriber data cached successfully', { 
      subscriberId, 
      ukId,
      cacheTTL: registryCache.getTtl(cacheKey)
    });

    return subscriberData;

  } catch (error) {
    if (error.code === 'ECONNABORTED') {
      logger.error('Registry lookup timed out', { 
        subscriberId, 
        ukId,
        timeout: REGISTRY_TIMEOUT 
      });
    } else if (error.response) {
      logger.error('Registry lookup failed with response', {
        status: error.response.status,
        data: error.response.data,
        subscriberId,
        ukId
      });
    } else {
      logger.error('Registry lookup failed', {
        error: error.message,
        subscriberId,
        ukId
      });
    }
    return null;
  }
};

// Clear cache for a specific subscriber
const clearSubscriberCache = (subscriberId, ukId) => {
  const cacheKey = `${subscriberId}:${ukId}`;
  registryCache.del(cacheKey);
  logger.debug('Cleared subscriber cache', { subscriberId, ukId });
};

// Clear entire cache
const clearRegistryCache = () => {
  registryCache.flushAll();
  logger.info('Cleared entire registry cache');
};

export { 
  lookupSubscriber,
  clearSubscriberCache,
  clearRegistryCache
};
