"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.createSensor = void 0;
var _object = require("./object");
var _resizeObserver = require("./resizeObserver");
/**
 * Created by hustcc on 18/7/5.
 * Contract: i@hust.cc
 */

/**
 * sensor strategies
 */
// export const createSensor = createObjectSensor;
var createSensor = typeof ResizeObserver !== 'undefined' ? _resizeObserver.createSensor : _object.createSensor;
exports.createSensor = createSensor;