import ConfirmHandler from '../handlers/confirmHandler.js';
import express from 'express';
const confirmRouter = express.Router();

confirmRouter.post('/confirm', async (req, res) => {
  try {
    const order = req.body?.message?.order;

    if (!order || !order.items || !order.billing || !order.fulfillments) {
      return res.status(400).json({ success: false, error: 'Missing required fields in ONDC order payload' });
    }

    const transformedOrder = {
      items: order.items.map(item => ({
        id: item.id,
        quantity: item.quantity,
        options: item.options || {}
      })),
      billing: {
        name: order.billing.name,
        email: order.billing.email,
        phone: order.billing.phone,
        address: `${order.billing.address.building}, ${order.billing.address.locality}` || "N/A"
      },
      fulfillments: order.fulfillments
    };

    const service = new ConfirmHandler();
    const result = await service.createOrder(transformedOrder);

    if (result.success) {
      res.status(200).json({ success: true, orderId: result.orderId });
    } else {
      res.status(500).json({ success: false, error: result.error });
    }

  } catch (err) {
    console.error('❌ confirm handler error:', err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

export default confirmRouter;
