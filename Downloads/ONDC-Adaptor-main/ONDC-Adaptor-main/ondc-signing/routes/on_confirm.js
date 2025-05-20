import express from 'express';
const onConfirmRouter = express.Router();

import onConfirmHandler from '../handlers/onconfirmHandler.js';

import onConfirmMiddleware from '../middlewares/onConfirmMiddleware.js';


// import authenticateSnpRequest, { validateOnConfirmSchema } from '../middlewares/onconfirmMiddleware.js'; 
// onConfirmRouter.post('/on_confirm', authenticateSnpRequest, validateOnConfirmSchema, onConfirmHandler);

// Set the route to match the API endpoint
onConfirmRouter.post('/on_confirm', onConfirmMiddleware, onConfirmHandler);

export default onConfirmRouter;
