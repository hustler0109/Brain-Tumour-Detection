// /routes/onCancelRouter.js

import express from 'express';
import onCancelMiddleware from '../middlewares/onCancelMiddleware.js'; 
import handleOnCancel from '../handlers/onCancelHandler.js';

const onCancelRouter = express.Router();

onCancelRouter.post('/on_cancel', onCancelMiddleware, handleOnCancel);

export default onCancelRouter;
