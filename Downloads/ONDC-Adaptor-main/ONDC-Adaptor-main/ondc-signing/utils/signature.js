// /utils/signature.js
import crypto from 'crypto';
import config from './config.js'; // Assuming your private key is in config

const privateKey = config.ondc.signingPrivateKey; // Make sure this is your actual private key

const Signer = {
  signPayload: (payload) => {
    try {
      const signer = crypto.createSign('SHA256'); // Ensure this matches the ONDC signing algorithm
      signer.update(JSON.stringify(payload));
      const signature = signer.sign(privateKey, 'base64'); // Ensure output format matches ONDC requirements
      return signature;
    } catch (error) {
      console.error('Error signing payload:', error);
      return null;
    }
  },
};

export default Signer;