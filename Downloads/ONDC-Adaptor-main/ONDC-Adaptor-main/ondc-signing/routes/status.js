// /routes/statusRouter.js

import express from "express";
import { processStatusRequest } from "../middlewares/statusMiddleware.js";
import statusHandler from "../handlers/statusHandler.js";

const statusRouter = express.Router();

statusRouter.post("/status", processStatusRequest, statusHandler);

export default statusRouter;
