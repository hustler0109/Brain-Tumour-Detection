import axios from "axios";
import { signRequest } from "../auth/cryptoUtils.js"; 

export default async function searchHandler(req, res) {
  console.log("search handler executed");

  try {
    const { context } = req.body;
    const isValidRequest = req.isValidRequest;
    console.log('isValidRequest', isValidRequest);

    // 1. Basic request body validation
    if (!isValidRequest) {
      console.warn("NACK: Request failed middleware validation");
      return res.status(200).json({
        response: {
          context: {
            ...context,
            action: "on_search",
            bpp_id: context?.bpp_id,
            bpp_uri: context?.bpp_uri,
          },
          message: {
            ack: {
              status: "NACK",
            },
          },
          error: {
            type: "AUTH_ERROR",
            code: "10002",
            message: "Authentication failed or missing OpenCart API token/environment variables.",
          },
        },
      });
    }

    if (!context || !context.transaction_id || !context.message_id || !context.bap_id || !context.bap_uri) {
      console.log("NACK: Missing mandatory context parameters");
      return res.status(200).json({
        response: {
          context: {
            ...context,
            action: "on_search",
            bpp_id: req.body?.context?.bpp_id,
            bpp_uri: req.body?.context?.bpp_uri,
          },
          message: {
            ack: {
              status: "NACK",
            },
          },
          error: {
            type: "REQUEST_ERROR",
            code: "10001",
            message: "Missing mandatory parameters in the /search request context.",
          },
        },
      });
    }

    // Send initial ACK response
    const ackResponse = {
      response: {
        context: {
          ...context,
          action: "on_search",
          bpp_id: req.body?.context?.bpp_id,
          bpp_uri: req.body?.context?.bpp_uri,
        },
        message: {
          ack: {
            status: "ACK",
          },
        },
      },
    };

    // Send ACK response first
    res.status(200).json(ackResponse);

    // Then process the request asynchronously
    setImmediate(async () => {
      try {
        const ondcRequest = req.body;
        console.log("\n\n\n request body ", ondcRequest, "-------------------------");

        // Sign the request
        const { signature, digest, authHeader } = signRequest(ondcRequest);

        // Get the BAP URI from the request context
        // const bapUri = context.bap_uri;
        // if (!bapUri) {
        //   console.error("BAP URI not found in request context");
        //   return;
        // }

        // Send the signed request to the adapter's /on_search endpoint
        if (!process.env.ADAPTOR_SITE) {
          console.error("ADAPTOR_SITE environment variable is not set");
          return;
        }

        const adapterUrl = `${process.env.ADAPTOR_SITE.replace(/\/$/, '')}/api/on_search`;
        console.log("Sending request to adapter:", adapterUrl);

        await axios.post(
          adapterUrl,
          ondcRequest,
          {
            headers: {
              "Content-Type": "application/json",
              Authorization: authHeader,
              Digest: digest,
              Signature: signature,
              "x-dev-mode": "true",
            },
          }
        ).catch(error => {
          if (error.code === 'ECONNREFUSED') {
            console.error(`Connection refused to ${adapterUrl}. Make sure the adapter server is running.`);
          } else {
            console.error("Error sending request to adapter:", error.message);
          }
        });

      } catch (error) {
        console.error("Error in async processing:", error);
        // Log the error but don't try to send another response
      }
    });

  } catch (error) {
    console.error("Error in search handler:", error);
    // Only send error response if we haven't sent a response yet
    if (!res.headersSent) {
      res.status(500).json({ 
        error: {
          type: "CORE-ERROR",
          code: "50000",
          message: "An error occurred while processing the search request"
        }
      });
    }
  }
}
